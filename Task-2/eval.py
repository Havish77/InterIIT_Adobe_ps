# ============================================================================
# INFERENCE & EVALUATION SCRIPT
# Run this AFTER training is complete
# ============================================================================

import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from collections import Counter
import gc

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TRAINED_MODEL_DIR = "./mistral_vlm_final"  # Where your trained model is saved
TEST_UNSEEN_BRANDS_PATH = "test_unseen_brands.csv"
TEST_UNSEEN_TIME_PATH = "test_unseen_time.csv"

print("="*70)
print("📊 TWEET GENERATION EVALUATION")
print("="*70)

# --- 1. Load Trained Model ---
print("\n🔄 Loading trained model for inference...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load trained LoRA weights
model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_DIR)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR)

print("✅ Model loaded successfully!")

# --- 2. Tweet Generation Function ---
def generate_tweet(company, username, timestamp, vlm_description, max_length=150):
    """Generate a tweet given metadata"""
    
    # Build visual context
    if vlm_description not in ['no media', 'media could not be processed', 
                                'media could not be downloaded', 'nan', None]:
        visual_context = f" The image shows: {vlm_description}"
    else:
        visual_context = ""
    
    # Parse timestamp
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        day_name = timestamp.day_name()
        hour = timestamp.hour
    except:
        day_name = 'a weekday'
        hour = 12
    
    # Create prompt
    prompt = (
        f"<s>[INST] Generate an engaging marketing tweet for {company} "
        f"(username: @{username}). "
        f"Context: It's {day_name} at {hour}:00."
        f"{visual_context} [/INST]"
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract tweet (after [/INST])
    if "[/INST]" in generated_text:
        tweet = generated_text.split("[/INST]")[-1].strip()
    else:
        tweet = generated_text.strip()
    
    # Clean up
    tweet = tweet.replace("<s>", "").replace("</s>", "").strip()
    
    return tweet

# --- 3. Load Test Data ---
print("\n🔄 Loading test datasets...")

try:
    test_unseen_brands = pd.read_csv(TEST_UNSEEN_BRANDS_PATH)
    test_unseen_time = pd.read_csv(TEST_UNSEEN_TIME_PATH)
    
    # Rename columns if needed
    if 'inferred company' in test_unseen_brands.columns:
        test_unseen_brands.rename(columns={'inferred company': 'company'}, inplace=True)
    if 'date' in test_unseen_brands.columns:
        test_unseen_brands.rename(columns={'date': 'timestamp'}, inplace=True)
        
    if 'inferred company' in test_unseen_time.columns:
        test_unseen_time.rename(columns={'inferred company': 'company'}, inplace=True)
    if 'date' in test_unseen_time.columns:
        test_unseen_time.rename(columns={'date': 'timestamp'}, inplace=True)
    
    print(f"✅ Test data loaded:")
    print(f"   • Unseen brands: {len(test_unseen_brands)} samples")
    print(f"   • Unseen time: {len(test_unseen_time)} samples")
    
except FileNotFoundError as e:
    print(f"❌ Error: Test files not found!")
    print(f"   Please provide: {TEST_UNSEEN_BRANDS_PATH} and {TEST_UNSEEN_TIME_PATH}")
    print(f"\n💡 Creating dummy test data for demonstration...")
    
    # Create dummy data for demo
    test_unseen_brands = pd.DataFrame({
        'company': ['nike', 'adidas', 'puma'],
        'username': ['Nike', 'adidas', 'PUMA'],
        'timestamp': ['2024-01-01 10:00:00'] * 3,
        'media': [''] * 3,
        'content': ['Sample tweet 1', 'Sample tweet 2', 'Sample tweet 3'],
        'vlm_description': ['no media'] * 3
    })
    test_unseen_time = test_unseen_brands.copy()

# --- 4. Generate Predictions ---
def generate_predictions_batch(test_df, task_name):
    """Generate predictions for a test dataset"""
    print(f"\n🔄 Generating predictions for: {task_name}")
    
    predictions = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Generating"):
        # Get VLM description if available
        vlm_desc = row.get('vlm_description', 'no media')
        
        # Generate tweet
        generated_tweet = generate_tweet(
            company=row.get('company', 'unknown'),
            username=row.get('username', 'unknown'),
            timestamp=row.get('timestamp', pd.Timestamp.now()),
            vlm_description=vlm_desc
        )
        
        predictions.append({
            'company': row.get('company', ''),
            'username': row.get('username', ''),
            'timestamp': row.get('timestamp', ''),
            'generated_content': generated_tweet,
            'actual_content': row.get('content', ''),
        })
    
    return pd.DataFrame(predictions)

print("\n" + "="*70)
print("📝 GENERATING PREDICTIONS")
print("="*70)

predictions_unseen_brands = generate_predictions_batch(
    test_unseen_brands, 
    "Unseen Brands (Test Set 1)"
)

predictions_unseen_time = generate_predictions_batch(
    test_unseen_time, 
    "Unseen Time Period (Test Set 2)"
)

# --- 5. Save Predictions ---
predictions_unseen_brands.to_csv("predictions_unseen_brands.csv", index=False)
predictions_unseen_time.to_csv("predictions_unseen_time.csv", index=False)

print("\n✅ Predictions saved:")
print("   • predictions_unseen_brands.csv")
print("   • predictions_unseen_time.csv")

# --- 6. Calculate Metrics ---
print("\n" + "="*70)
print("📊 CALCULATING METRICS")
print("="*70)

def calculate_bleu_1(reference, candidate):
    """Calculate BLEU-1 score"""
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if len(cand_words) == 0:
        return 0.0
    
    ref_counter = Counter(ref_words)
    cand_counter = Counter(cand_words)
    
    matches = sum((cand_counter & ref_counter).values())
    precision = matches / len(cand_words)
    
    return precision

def calculate_metrics(predictions_df):
    """Calculate evaluation metrics"""
    
    bleu_scores = []
    length_ratios = []
    word_counts_gen = []
    word_counts_actual = []
    
    for idx, row in predictions_df.iterrows():
        actual = str(row['actual_content'])
        generated = str(row['generated_content'])
        
        # BLEU-1 score
        bleu = calculate_bleu_1(actual, generated)
        bleu_scores.append(bleu)
        
        # Length metrics
        len_gen = len(generated.split())
        len_actual = len(actual.split())
        word_counts_gen.append(len_gen)
        word_counts_actual.append(len_actual)
        
        if len_actual > 0:
            length_ratios.append(len_gen / len_actual)
    
    metrics = {
        'avg_bleu_1': np.mean(bleu_scores),
        'std_bleu_1': np.std(bleu_scores),
        'avg_length_ratio': np.mean(length_ratios),
        'avg_generated_words': np.mean(word_counts_gen),
        'avg_actual_words': np.mean(word_counts_actual),
        'length_diff': abs(np.mean(word_counts_gen) - np.mean(word_counts_actual)),
    }
    
    return metrics, bleu_scores

metrics_brands, bleu_brands = calculate_metrics(predictions_unseen_brands)
metrics_time, bleu_time = calculate_metrics(predictions_unseen_time)

print("\n📊 RESULTS - Test Set 1: Unseen Brands")
print("="*70)
print(f"   📈 Average BLEU-1 Score:     {metrics_brands['avg_bleu_1']:.4f} ± {metrics_brands['std_bleu_1']:.4f}")
print(f"   📏 Average Length Ratio:     {metrics_brands['avg_length_ratio']:.2f}x")
print(f"   📝 Generated Length (words): {metrics_brands['avg_generated_words']:.1f}")
print(f"   📝 Actual Length (words):    {metrics_brands['avg_actual_words']:.1f}")
print(f"   📊 Length Difference:        {metrics_brands['length_diff']:.1f} words")

print("\n📊 RESULTS - Test Set 2: Unseen Time Period")
print("="*70)
print(f"   📈 Average BLEU-1 Score:     {metrics_time['avg_bleu_1']:.4f} ± {metrics_time['std_bleu_1']:.4f}")
print(f"   📏 Average Length Ratio:     {metrics_time['avg_length_ratio']:.2f}x")
print(f"   📝 Generated Length (words): {metrics_time['avg_generated_words']:.1f}")
print(f"   📝 Actual Length (words):    {metrics_time['avg_actual_words']:.1f}")
print(f"   📊 Length Difference:        {metrics_time['length_diff']:.1f} words")

# --- 7. Create Submission Files (PS Format) ---
print("\n" + "="*70)
print("📦 CREATING SUBMISSION FILES")
print("="*70)

# Create submission format: date, likes, username, media, inferred company, content
submission_brands = pd.DataFrame({
    'date': predictions_unseen_brands['timestamp'],
    'likes': 0,  # Placeholder
    'username': predictions_unseen_brands['username'],
    'media': '',  # Placeholder
    'inferred company': predictions_unseen_brands['company'],
    'content': predictions_unseen_brands['generated_content']
})

submission_time = pd.DataFrame({
    'date': predictions_unseen_time['timestamp'],
    'likes': 0,  # Placeholder
    'username': predictions_unseen_time['username'],
    'media': '',  # Placeholder
    'inferred company': predictions_unseen_time['company'],
    'content': predictions_unseen_time['generated_content']
})

# Save
submission_brands.to_csv("submission_unseen_brands.csv", index=False)
submission_time.to_csv("submission_unseen_time.csv", index=False)

print("✅ Submission files created:")
print("   • submission_unseen_brands.csv")
print("   • submission_unseen_time.csv")

# --- 8. Show Sample Predictions ---
print("\n" + "="*70)
print("📝 SAMPLE PREDICTIONS")
print("="*70)

def show_sample(df, title, num_samples=3):
    print(f"\n{title}")
    print("-" * 70)
    for i in range(min(num_samples, len(df))):
        sample = df.iloc[i]
        print(f"\n🔹 Sample {i+1}:")
        print(f"   Company: {sample['company']}")
        print(f"   Username: @{sample['username']}")
        print(f"   Time: {sample['timestamp']}")
        print(f"   \n   💬 Generated: {sample['generated_content'][:150]}...")
        if 'actual_content' in sample and sample['actual_content']:
            print(f"   ✅ Actual: {sample['actual_content'][:150]}...")

show_sample(predictions_unseen_brands, "🎯 Unseen Brands Predictions")
show_sample(predictions_unseen_time, "🎯 Unseen Time Period Predictions")

# --- 9. Summary ---
print("\n" + "="*70)
print("🎉 EVALUATION COMPLETE!")
print("="*70)

print("\n📁 Generated Files:")
print("   1. predictions_unseen_brands.csv      (Detailed predictions)")
print("   2. predictions_unseen_time.csv        (Detailed predictions)")
print("   3. submission_unseen_brands.csv       (Submission format)")
print("   4. submission_unseen_time.csv         (Submission format)")

print("\n📊 Overall Performance:")
print(f"   • Unseen Brands BLEU-1:    {metrics_brands['avg_bleu_1']:.4f}")
print(f"   • Unseen Time BLEU-1:      {metrics_time['avg_bleu_1']:.4f}")
print(f"   • Combined Average:        {(metrics_brands['avg_bleu_1'] + metrics_time['avg_bleu_1'])/2:.4f}")

print("\n✅ Ready for submission!")

# Cleanup
del model, base_model
gc.collect()
torch.cuda.empty_cache()

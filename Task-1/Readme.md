# Tweet Engagement Predictor: A Deep Learning Approach

This project predicts the number of likes a tweet will receive by using a sophisticated, two-stage machine learning pipeline. The core strategy involves first classifying a tweet into a "popularity bucket" and then using a specialist regressor trained for that specific bucket to predict the final like count.

This approach is designed to handle the highly skewed nature of social media engagement data, where most tweets have low engagement and a few go viral.

## Methodology: Classification-then-Regression

Our pipeline breaks the complex problem of predicting a continuous number into two simpler, more manageable steps.

### 1\. **Class Definition**

The continuous `likes` target is first converted into three discrete classes based on quantiles to ensure a balanced approach:

  - **Class 0 (Common):** The bottom 75% of tweets.
  - **Class 1 (Popular):** Tweets between the 75th and 95th percentile.
  - **Class 2 (Viral):** The top 5% of all tweets.

### 2\. **Feature Extraction via Embeddings**

We use a powerful pre-trained **Sentence Transformer model (`all-MiniLM-L6-v2`)** to act as a feature extractor. This encoder reads the raw text of the tweets (content, company, username) and converts it into dense, 384-dimensional numerical vectors called **embeddings**. These embeddings capture the deep semantic meaning of the text, providing rich features for our models.

### 3\. **The Classifier (The "Triage Nurse")**

An **XGBoost Classifier** is trained on the generated text embeddings. Its sole purpose is to predict the `popularity_class` (0, 1, or 2) of a given tweet. This model is trained with class weights to counteract the data imbalance and improve its ability to identify rare "Popular" and "Viral" tweets.

### 4\. **Specialist Regressors (The "Doctors")**

Three separate **XGBoost Regressor** models are trained, one for each popularity class.

  - **Regressor 0** is trained *only* on tweets from Class 0.
  - **Regressor 1** is trained *only* on tweets from Class 1.
  - **Regressor 2** is trained *only* on tweets from Class 2.

Each model becomes an expert in predicting likes within its specific engagement range, leading to more accurate and nuanced predictions.

-----

## File Structure

The project is organized into a modular pipeline of scripts that should be run in order.

```
.
├── train_data.csv                # Raw input training data
├── requirements.txt              # All necessary Python libraries
|
├── 01_create_classes.py          # Adds 'popularity_class' to the data
├── 02_generate_embeddings.py     # Creates and saves text embeddings using MiniLM
├── 03b_train_classifier_improved.py # Trains the classifier model with class weights
├── 04_train_regressors.py        # Trains the three specialist regressor models
├── 05_predict_likes.py           # Runs the full inference pipeline on new data
|
├── train_data_with_classes.csv   # (Generated) Data with class labels
├── embeddings_content.npy        # (Generated) Embeddings for the classifier
├── embeddings_combined.npy       # (Generated) Embeddings for the regressors
├── classifier_model_improved.joblib # (Generated) The trained classifier model
├── regressor_model_class_0.joblib  # (Generated) Specialist regressor for Class 0
├── regressor_model_class_1.joblib  # (Generated) Specialist regressor for Class 1
├── regressor_model_class_2.joblib  # (Generated) Specialist regressor for Class 2
└── final_predictions.csv         # (Generated) Final output file with predictions
```

-----

## Setup and Installation

### Prerequisites

  - Python 3.8+
  - `pip` and `venv`

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # On Windows
    python -m venv venv
    venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Create a `requirements.txt` file** with the following content:

    ```text
    # Core Libraries
    pandas==2.2.2
    scikit-learn==1.4.2
    xgboost==2.0.3

    # Deep Learning & Embeddings
    torch==2.3.0
    transformers==4.41.2
    sentence-transformers==2.7.0
    accelerate
    ```

    *Note: For GPU support, install the CUDA-enabled version of PyTorch from their official website first.*

4.  **Install all dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## How to Run the Pipeline

Execute the scripts from your terminal in the following order. Each script builds upon the output of the previous one.

1.  **Step 1: Create Popularity Classes**

    ```bash
    python 01_create_classes.py
    ```

2.  **Step 2: Generate Text Embeddings**
    *This step can be slow, especially on a CPU. Using a GPU is highly recommended.*

    ```bash
    python 02_generate_embeddings.py
    ```

3.  **Step 3: Train the Classifier**

    ```bash
    python 03b_train_classifier_improved.py
    ```

4.  **Step 4: Train the Specialist Regressors**

    ```bash
    python 04_train_regressors.py
    ```

5.  **Step 5: Generate Final Predictions**
    *This script runs the complete inference pipeline on a test portion of the data and saves the final output.*

    ```bash
    python 05_predict_likes.py
    ```

After running all the steps, the final output will be in **`final_predictions.csv`**, containing the original tweet, its actual likes, the predicted class, and the final predicted like count.

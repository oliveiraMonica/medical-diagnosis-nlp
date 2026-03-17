import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing import clean_text


# ==============================
# PATH CONFIGURATION
# ==============================

# Project root directory (one level above src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data and models paths
DATA_PATH = os.path.join(BASE_DIR, "data", "medical_transcriptions.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ==============================
# LOAD DATA
# ==============================

def load_data():
    """
    Load the medical transcription dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing clinical transcriptions and medical specialties.
    """

    df = pd.read_csv(DATA_PATH)
    return df


# ==============================
# PREPROCESSING
# ==============================

def preprocess_data(df):
    """
    Clean and prepare text data.

    - Applies text cleaning
    - Creates clean_text column
    - Removes very short texts
    """

    df["clean_text"] = df["transcription"].astype(str).apply(clean_text)

    df["clean_word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    df = df[df["clean_word_count"] >= 5]

    return df


# ==============================
# FEATURE ENGINEERING
# ==============================

def build_features(df):
    """
    Convert text into TF-IDF features.
    """

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df["clean_text"])

    return X, vectorizer


# ==============================
# DATA SPLIT
# ==============================

def prepare_data(X, df):
    """
    Split dataset into training and testing sets.
    """

    y = df["medical_specialty"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# ==============================
# TRAINING
# ==============================

def train_model(X_train, X_test, y_train, y_test, vectorizer):
    """
    Train and evaluate Logistic Regression model.
    """

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Ensure models directory exists at project root
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model and vectorizer
    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))

    print("\nModels saved in:", MODELS_DIR)

    return model


# ==============================
# MAIN PIPELINE
# ==============================

def main():
    """
    Run full ML pipeline.
    """

    df = load_data()

    df = preprocess_data(df)

    X, vectorizer = build_features(df)

    X_train, X_test, y_train, y_test = prepare_data(X, df)

    train_model(X_train, X_test, y_train, y_test, vectorizer)


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()
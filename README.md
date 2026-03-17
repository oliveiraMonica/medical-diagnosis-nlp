# 🧠 Medical Diagnosis NLP API

This project is a complete **Natural Language Processing (NLP) pipeline** for classifying clinical transcriptions into medical specialties.

It includes:

* Text preprocessing (cleaning, normalization, stopword removal)
* Feature extraction using **TF-IDF**
* Machine Learning model (**Logistic Regression**)
* Model evaluation and persistence
* REST API built with **Flask** for real-time predictions

---

##  Project Overview

The goal of this project is to automatically classify medical text into its corresponding specialty, such as:

* Cardiovascular / Pulmonary
* Neurology
* Orthopedic
* General Medicine
* Surgery

The system processes raw clinical notes and predicts the most likely specialty using machine learning.

---

## 📊 Exploratory Data Analysis (EDA)

Before building the model, an exploratory data analysis was conducted to better understand the dataset.

The analysis includes:

- Distribution of medical specialties
- Text length analysis
- Missing values inspection
- Class imbalance identification

All visualizations and analysis can be found in the notebook: 👉 [View EDA Notebook](notebooks/eda.ipynb)

This step was essential to:

- Identify class imbalance issues
- Understand text variability
- Guide preprocessing and modeling decisions
---

## Architecture

```
Input Text
   ↓
Preprocessing (clean_text)
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression Model
   ↓
Prediction + Top 3 Probabilities
   ↓
Flask API Response (JSON)
```

---

## 📂 Project Structure

```
medical-diagnosis-nlp/
│
├── data/                  # Dataset (ignored in Git)
├── models/                # Saved model and vectorizer
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── notebooks/             # EDA and experimentation
│   └── eda.ipynb
│
├── src/
│   ├── preprocessing.py   # Text cleaning functions
│   └── train.py           # Training pipeline
│
├── app.py                 # Flask API
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Create and activate a virtual environment:

```bash
  python -m venv .venv
  source .venv/bin/activate
```

Install dependencies:

```bash
  pip install -r requirements.txt
```

---

## Training the Model

Run the training pipeline:

```bash
  python src/train.py
```

This will:

* Train the model
* Evaluate performance
* Save artifacts in `/models`

---

## Running the API

Start the Flask server:

```bash
  python app.py
```

API will be available at:

```
http://127.0.0.1:5000/
```

---

## Prediction Endpoint

### POST `/predict`

#### Request:

```json
{
  "text": "patient with chest pain and shortness of breath"
}
```

#### Response:

```json
{
  "prediction": "Cardiovascular / Pulmonary",
  "top_3": [
    {
      "label": "Cardiovascular / Pulmonary",
      "score": 0.4089
    },
    {
      "label": "General Medicine",
      "score": 0.0667
    },
    {
      "label": "Consult - History and Phy.",
      "score": 0.0643
    }
  ]
}
```

---

## 📊 Model Performance

* Algorithm: **Logistic Regression**
* Features: **TF-IDF (5000 max features)**
* Accuracy: ~26%

This is a **baseline model**, and performance is affected by:

* Large number of classes (~40)
* Class imbalance in dataset

---

## Key Learnings

* End-to-end NLP pipeline implementation
* Handling imbalanced multi-class classification
* Feature engineering with TF-IDF
* Model evaluation with precision/recall/F1
* Building production-ready ML APIs

---

## Future Improvements

* Use advanced models (e.g., **BERT / Transformers**)
* Improve class imbalance handling
* Add model versioning
* Deploy API to cloud (Render / Railway / AWS)
* Add authentication and monitoring

---

## 👩‍💻 Author

**Mônica Oliveira**

* 💻 Software Developer transitioning into AI/ML
* ☕ Passionate about code and coffee

---

## ⭐ If you like this project

Give it a star on GitHub ⭐ and feel free to contribute!

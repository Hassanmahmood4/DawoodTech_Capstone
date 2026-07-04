# Fake News Prediction System

End-to-end machine learning capstone project that classifies news articles as **Fake** or **Real** using the ISOT Fake News Dataset, TF-IDF feature engineering, scikit-learn models, and a Streamlit deployment.

## Problem Statement

Misinformation spreads quickly online. This project builds a practical screening tool that analyzes article titles and body text, compares multiple ML models, optimizes the best performer, and serves predictions through an interactive web app.

## Dataset

**ISOT Fake News Dataset**

- Source: [Kaggle — Fake News Classification Dataset](https://www.kaggle.com/datasets/atharvaingle/fake-news-classification-dataset)
- Files: `Fake.csv`, `True.csv`
- Columns: `title`, `text`, `subject`, `date`
- Labels: `0 = Fake`, `1 = Real`

### Download Instructions

1. Download the dataset from Kaggle.
2. Place both CSV files in `data/raw/`:

```text
data/raw/Fake.csv
data/raw/True.csv
```

Alternative mirror:

```bash
curl -L "https://huggingface.co/datasets/declan101/NewsFineTuning/resolve/main/Fake.csv" -o data/raw/Fake.csv
curl -L "https://huggingface.co/datasets/declan101/NewsFineTuning/resolve/main/True.csv" -o data/raw/True.csv
```

## Project Structure

```text
DawoodTech_Capstone/
├── app.py
├── train.py
├── data/raw/
├── notebooks/project.ipynb
├── models/best_model.joblib
├── reports/
├── utils/
├── requirements.txt
└── requirements-train.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

## Train Models

```bash
python train.py
```

This script will:

- clean and merge the dataset
- generate EDA plots in `reports/figures/`
- train 4 models: Logistic Regression, Decision Tree, Random Forest, SVM
- compare model performance
- run `GridSearchCV` on the best model
- save `models/best_model.joblib` and metrics JSON files

## Run Streamlit App

```bash
streamlit run app.py
```

### App Features

- Home dashboard
- Dataset overview
- Exploratory data analysis
- Prediction form with confidence scores
- Model performance metrics

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 99.77% |
| F1 Score | 99.77% |
| ROC-AUC | 99.99% |
| Best Model | Random Forest |

## Models Compared

| Model | Role |
|-------|------|
| Logistic Regression | Linear baseline for sparse TF-IDF text |
| Decision Tree | Non-linear interpretable classifier |
| Random Forest | Ensemble comparison |
| Support Vector Machine | Strong margin-based text classifier |

## Technologies Used

- Python, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Joblib, Streamlit, Jupyter

## Deliverables

- [x] Jupyter notebook: `notebooks/project.ipynb`
- [x] Streamlit app: `app.py`
- [x] Trained model: `models/best_model.joblib`
- [x] Project documentation: `reports/documentation.md`
- [ ] PDF report export
- [ ] Presentation slides
- [ ] Demo video (3–5 minutes)

## Author

DawoodTech Capstone — Week 8 Final Project

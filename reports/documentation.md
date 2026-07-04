# Fake News Prediction System — Project Documentation

## 1. Project Introduction

The Fake News Prediction System is an end-to-end machine learning application built for the DawoodTech Week 8 Capstone. It classifies news articles as fake or real using natural language processing techniques and classical machine learning models.

The project demonstrates the complete ML lifecycle: data preparation, exploratory analysis, model development, optimization, deployment, and documentation.

## 2. Problem Statement

Online misinformation can influence public opinion, elections, and public health decisions. Manual fact-checking does not scale to the volume of content published every day.

This project addresses the problem by building an automated binary classifier that analyzes article text and returns:

- a predicted label (`Fake` or `Real`)
- a confidence score
- model performance transparency for stakeholders

## 3. Dataset Information

### Source

ISOT Fake News Dataset

- Kaggle: https://www.kaggle.com/datasets/atharvaingle/fake-news-classification-dataset
- Official ISOT documentation: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

### Files

- `Fake.csv` — articles from unreliable sources
- `True.csv` — articles from Reuters

### Columns

| Column | Description |
|--------|-------------|
| `title` | Article headline |
| `text` | Full article body |
| `subject` | Topic/category |
| `date` | Publication date |

### Target Variable

- `0` = Fake
- `1` = Real

### Preprocessing Decisions

- Combined `title` and `text` into one modeling field
- Removed empty records
- Cleaned URLs, HTML, punctuation, and extra whitespace
- Removed articles above the 99th percentile text length to reduce outlier impact

## 4. Technologies Used

- Python 3.11
- Pandas and NumPy for data handling
- Scikit-learn for modeling and evaluation
- TF-IDF vectorization for text features
- Matplotlib and Seaborn for visualization
- Joblib for model serialization
- Streamlit for deployment
- Jupyter Notebook for experimentation and reporting
- Git and GitHub for version control

## 5. Methodology

### Pipeline

1. Load and merge `Fake.csv` and `True.csv`
2. Clean and combine article text
3. Perform EDA and generate visual insights
4. Split data into train/test sets with stratification
5. Convert text to TF-IDF features
6. Train and compare four ML models
7. Select the best model by F1-score
8. Optimize with `GridSearchCV`
9. Save the final model and deploy with Streamlit

### Feature Engineering

- `TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, stop_words='english')`
- Engineered numeric features for EDA:
  - text length
  - title length
  - word count

## 6. Model Comparison

The capstone requires at least three models. This project trains four:

| Model | Type | Purpose |
|-------|------|---------|
| Logistic Regression | Linear | Baseline for sparse text features |
| Decision Tree | Non-linear | Interpretable single-tree model |
| Random Forest | Ensemble | More robust non-linear model |
| Support Vector Machine | Margin-based | Strong classifier for high-dimensional text |

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- 5-fold cross-validation

See `reports/model_comparison.json` for the full comparison table after training.

## 7. Results

After running `python train.py`, the optimized model metrics are saved to:

- `reports/optimized_metrics.json`
- `reports/figures/optimized_confusion_matrix.png`
- `reports/figures/optimized_roc_curve.png`

Typical ISOT results with TF-IDF and classical ML models reach strong performance, often above 95% accuracy depending on preprocessing and selected model.

## 8. Challenges Faced

### Text Noise

News articles contain URLs, punctuation errors, and inconsistent formatting. Custom text cleaning improved model stability.

### High-Dimensional Features

TF-IDF creates sparse, high-dimensional input. Limiting `max_features` and using linear models helped control memory usage and training time.

### Probability Outputs for SVM

`LinearSVC` does not provide `predict_proba` by default. The project uses `CalibratedClassifierCV` when SVM is part of the comparison and prefers models with native probability support for deployment when appropriate.

### Large Dataset Training Time

Random Forest and grid search are slower on full text data. A reproducible `train.py` script makes regeneration easier outside the notebook.

## 9. Future Improvements

- Fine-tune transformer models such as DistilBERT or RoBERTa
- Add explainability with LIME or SHAP
- Support batch scoring from URLs or RSS feeds
- Add multilingual fake news detection
- Deploy to Streamlit Cloud or containerized production hosting
- Add model monitoring and periodic retraining

## 10. Submission Checklist

- [x] Complete source code
- [x] Jupyter notebook
- [x] Streamlit application
- [x] Trained model artifact
- [x] README and documentation
- [ ] Export this file to PDF (`project_report.pdf`)
- [ ] Create presentation slides
- [ ] Record 3–5 minute demo video

## 11. How to Reproduce

```bash
pip install -r requirements-train.txt
python train.py
streamlit run app.py
```

## 12. References

- ISOT Fake News Dataset documentation
- Scikit-learn model selection and pipeline guides
- Streamlit documentation

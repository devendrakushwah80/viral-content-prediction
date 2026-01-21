# Social Media Viral Content Prediction (Leakage-Free ML Pipeline)

## 📌 Project Overview

This project builds a **machine learning pipeline to predict whether a social media post will go viral** using **only pre-post features** (i.e., information available *before* publishing the post).

A major focus of this project is **data leakage detection and removal**, ensuring that the model performance is realistic, explainable, and interview-ready.

---

## 🎯 Problem Statement

Given metadata about a social media post (platform, topic, language, posting time, sentiment, etc.), predict:

```
is_viral = 1  → Viral post
is_viral = 0  → Non-viral post
```

This is a **binary classification problem**.

---

## 📂 Dataset Description

### Target Column

* **is_viral** (binary)

### Original Features

* platform
* content_type
* topic
* language
* region
* post_datetime
* hashtags
* views
* likes
* comments
* shares
* engagement_rate
* sentiment_score

---

## 🚨 Data Leakage Analysis (CRITICAL)

Initial experiments showed **100% accuracy and ROC-AUC = 1.0** across tree-based models. This indicated **severe data leakage**.

### ❌ Leakage Features (REMOVED)

These features are generated *after* a post becomes viral and were therefore dropped:

* views
* likes
* comments
* shares
* engagement_rate

📌 **Key lesson:** High correlation does NOT mean a feature is valid.

---

## 🛠 Feature Engineering

### Datetime Processing

`post_datetime` was initially an object column and was:

1. Converted to datetime
2. Engineered into meaningful features
3. Original column dropped

Generated features:

* post_day
* post_month
* post_weekday

### Dropped Columns

* post_id (identifier)
* hashtags (raw text, NLP not used)
* post_datetime (after feature extraction)

---

## ✅ Final Feature Set (Leakage-Free)

### Numerical Features

* sentiment_score
* post_day
* post_month
* post_weekday

### Categorical Features

* platform
* content_type
* topic
* language
* region

---

## 🔄 Preprocessing Pipeline

* **Numerical:** StandardScaler
* **Categorical:** OneHotEncoder (`handle_unknown='ignore'`)
* **ColumnTransformer** used for clean preprocessing

Automatic column selection was implemented to avoid column mismatch errors.

---

## 🤖 Models Trained

The following models were trained using **Pipeline + GridSearchCV**:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree

Evaluation was done on a held-out test set.

---

## 📊 Final Model Performance (Leakage-Free)

| Model               | Accuracy | Precision | Recall   | F1-score | ROC-AUC |
| ------------------- | -------- | --------- | -------- | -------- | ------- |
| Logistic Regression | **0.70** | 0.70      | **0.99** | **0.82** | 0.47    |
| Random Forest       | 0.69     | 0.70      | 0.96     | 0.81     | 0.48    |
| Gradient Boosting   | 0.68     | 0.70      | 0.95     | 0.81     | 0.49    |
| KNN                 | 0.64     | 0.69      | 0.86     | 0.77     | 0.45    |
| SVM                 | 0.59     | 0.68      | 0.78     | 0.73     | 0.49    |
| Decision Tree       | 0.54     | 0.67      | 0.67     | 0.67     | 0.45    |

📌 **Best Model:** Logistic Regression

---

## ⚠️ Key Insight: ROC-AUC vs F1

Although F1-score and recall are high, ROC-AUC is close to 0.5 due to **class imbalance** and a bias toward predicting the positive class.

### Planned Improvements

* Class-weighted models
* Threshold tuning
* ROC-AUC–based GridSearch scoring

---

## 🧠 Key Learnings

* Data leakage can silently produce perfect but useless models
* Heatmaps only apply to numerical features
* Categorical features require statistical tests or encoding-based evaluation
* Logistic Regression can outperform complex models with clean features
* Realistic metrics are more valuable than perfect scores

---

## 🧪 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## 📌 How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Open notebook

```bash
jupyter notebook social_media_viral_content_classification.ipynb
```

3. Run cells top to bottom

---

## 📄 Future Work

* NLP on hashtags
* Feature importance visualization
* ROC & Precision–Recall curves
* Deployment as a REST API

---

## 🧠 Interview-Ready Summary

> "This project demonstrates an end-to-end ML pipeline with strong emphasis on data leakage detection, feature engineering, and realistic model evaluation for viral content prediction."

---

⭐ If you found this project useful, consider starring the repository!

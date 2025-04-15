# Credit Card Fraud Detection using Machine Learning

This project aims to identify fraudulent credit card transactions using machine learning techniques. With the rise in online transactions, detecting fraud is crucial for financial institutions. The model used here is Logistic Regression, which is trained on a modified version of the original dataset to handle class imbalance.

---

## 📌 Project Overview

- **Dataset Source**: Kaggle ([Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Records**: 284,807 credit card transactions from European cardholders
- **Fraudulent Cases**: 492 (≈0.172% of the data)
- **Features**: 31 total (Time, Amount, 28 PCA-transformed features, and the Class label)

The dataset is highly imbalanced, with very few fraudulent cases compared to normal transactions. Our goal is to build a reliable classification model that can detect fraud effectively while maintaining high recall and precision.

---

## ⚙️ Workflow

### 1. Importing Libraries
All required Python libraries like `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` are imported.

### 2. Loading the Dataset
The dataset is loaded using `pandas`. Initial exploration includes checking data shape, column types, and verifying class imbalance.

### 3. Data Exploration
- Analyze the distribution of fraud vs. non-fraud
- Use visualizations like bar charts and correlation heatmaps
- Check missing/null values

### 4. Data Preprocessing
- **Feature Scaling**: Apply `StandardScaler` on `Time` and `Amount`
- **Class Balancing**: Under-sampling is used to create a balanced dataset by reducing normal transactions
- **Splitting Data**: Divide into training and testing sets

### 5. Model Building
- Logistic Regression is used for binary classification
- The model is trained on the balanced dataset

### 6. Model Evaluation
- Evaluate using Accuracy, Precision, Recall, and ROC-AUC
- Confusion Matrix and ROC Curve are plotted to visualize performance
- Focus is given to minimizing false negatives to avoid missing fraud

---

## 🚀 Future Work

- **Use SMOTE** (Synthetic Minority Over-sampling Technique) instead of under-sampling to preserve data
- **Experiment with Advanced Models**:
  - Random Forest
  - XGBoost
  - Neural Networks
- **Feature Engineering**:
  - Transform `Time` into meaningful groups (e.g., day/night)
  - Explore interactions between PCA features
- **Model Optimization**:
  - Apply cross-validation
  - Use grid search for hyperparameter tuning
- **Deploy Model**:
  - Integrate into an API or web dashboard for real-time fraud detection

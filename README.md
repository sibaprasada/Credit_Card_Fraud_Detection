# 💳 Credit Card Fraud Detection System

This repository contains a Machine Learning project designed to detect fraudulent credit card transactions. The project uses a **Logistic Regression** model to classify transactions as either "Legit" (Normal) or "Fraudulent".

## 📂 Dataset Details

The project uses the **Credit Card Fraud Detection** dataset.

* **Original Dataset Shape:** 284,807 Rows × 31 Columns
* **Columns:**
    * `Time`, `Amount`: The only features that have not been transformed.
    * `V1` to `V28`: Anonymized features resulting from PCA transformation.
    * `Class`: The target variable (0 = Legit Transaction, 1 = Fraudulent Transaction).
* **Class Imbalance:** The original dataset is highly unbalanced:
    * **Legit Transactions:** 284,315
    * **Fraud Transactions:** 492

## ⚙️ Methodology & Steps Taken

Since the dataset was highly unbalanced (very few fraud cases compared to normal ones), I used the **Undersampling** technique to build a balanced dataset.

### 1. Data Preprocessing
* **Exploration:** Analyzed the dataset structure and checked for missing values (none found).
* **Separation:** Separated the data into legit and fraud transactions to analyze their statistical properties (e.g., mean amount).

### 2. Handling Imbalance (Undersampling)
* Because the fraud cases were so few (492), training a model on the original data would make it biased towards legit transactions.
* **Solution:** I created a new sub-sample dataset containing:
    * All 492 Fraud transactions.
    * A random sample of 492 Legit transactions.
* **New Dataset Size:** 984 rows (evenly distributed between both classes).

### 3. Feature Extraction
* Split the data into Features (`X`) and Target (`Y`).
* Split the data into **Training (80%)** and **Testing (20%)** sets using `train_test_split`.

## 🤖 Model Used: Logistic Regression

I chose **Logistic Regression** for this project.

### Why this model?
1.  **Binary Classification:** The problem requires a simple "Yes/No" (Fraud/Legit) output, which Logistic Regression handles perfectly.
2.  **Efficiency:** It works very well on smaller, balanced datasets (like the one created after undersampling) and provides interpretable results.

## 📊 Results & Accuracy

The model was evaluated using the **Accuracy Score**.

| Metric | Score |
| :--- | :--- |
| **Training Data Accuracy** | **94.66%** |
| **Test Data Accuracy** | **93.91%** |

* The model achieved a high accuracy on both training and unseen test data, indicating that it generalized well without overfitting.

## 🚀 How to Run

1.  Clone the repository.
2.  Install the required libraries:
    ```bash
    pip install numpy pandas scikit-learn
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook credit_card_fraud_detection.ipynb
    ```

## 🔮 Future Improvements
* Try **Oversampling** (SMOTE) instead of Undersampling to retain more data.
* Implement **Random Forest** or **XGBoost** to potentially improve accuracy further.

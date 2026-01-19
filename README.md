# ML Assignment 2

## Problem Statement

Customer churn, the loss of customers to competitors, is a critical challenge for telecommunications companies. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to develop machine learning classification models to predict which customers are likely to churn based on their demographic information, account details, and service usage patterns.

The objective is to:
1. Build and compare six different classification algorithms on customer churn data
2. Evaluate each model using multiple performance metrics
3. Identify the most effective model for predicting customer churn
4. Deploy an interactive web application for real-time churn prediction

By accurately predicting customer churn, businesses can proactively implement retention strategies, personalize customer experiences, and reduce revenue loss.

---

## Dataset Description

### Dataset: Telco Customer Churn

**Source:** Kaggle - Telco Customer Churn Dataset
**Total Instances:** 7,043 customer records
**Total Features:** 20 (after removing customerID)
**Features after Preprocessing:** 30 (after one-hot encoding categorical variables)
**Target Variable:** Churn (Binary: Yes/No)
**Class Distribution:**
- No Churn: 5,174 customers (73.5%)
- Churn: 1,869 customers (26.5%)

### Feature Categories

**1. Demographic Information:**
- `gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Whether customer is a senior citizen (0/1)
- `Partner`: Whether customer has a partner (Yes/No)
- `Dependents`: Whether customer has dependents (Yes/No)

**2. Account Information:**
- `tenure`: Number of months the customer has stayed with the company
- `Contract`: Contract type (Month-to-month, One year, Two year)
- `PaperlessBilling`: Whether customer uses paperless billing (Yes/No)
- `PaymentMethod`: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)
- `MonthlyCharges`: Monthly charges amount
- `TotalCharges`: Total amount charged to the customer

**3. Service Information:**
- `PhoneService`: Whether customer has phone service (Yes/No)
- `MultipleLines`: Whether customer has multiple lines (Yes/No/No phone service)
- `InternetService`: Type of internet service (DSL, Fiber optic, No)
- `OnlineSecurity`: Whether customer has online security (Yes/No/No internet service)
- `OnlineBackup`: Whether customer has online backup (Yes/No/No internet service)
- `DeviceProtection`: Whether customer has device protection (Yes/No/No internet service)
- `TechSupport`: Whether customer has tech support (Yes/No/No internet service)
- `StreamingTV`: Whether customer has streaming TV (Yes/No/No internet service)
- `StreamingMovies`: Whether customer has streaming movies (Yes/No/No internet service)

### Data Preprocessing

The following preprocessing steps were applied:

1. **Removed** `customerID` column (not a predictive feature)
2. **Converted** `TotalCharges` from object to numeric (handled empty strings)
3. **Encoded** binary categorical variables (gender, Partner, Dependents, PhoneService, PaperlessBilling) to 0/1
4. **One-hot encoded** multi-category variables (MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaymentMethod)
5. **Converted** target variable `Churn` to binary (0: No, 1: Yes)
6. **Applied** Standard Scaling to features for models requiring normalized data (Logistic Regression, kNN)

### Dataset Characteristics

- **Class Imbalance:** The dataset exhibits class imbalance with approximately 3:1 ratio of non-churners to churners
- **Mixed Data Types:** Contains both numerical (tenure, charges) and categorical features
- **No Missing Values:** After preprocessing, the dataset has no missing values
- **Feature Correlations:** Monthly charges and total charges show high correlation; service features are correlated with internet service type

---

## Models Used

Six classification models were implemented and evaluated on the Telco Customer Churn dataset. All models were trained using a 70-30 train-test split with stratification to maintain class distribution.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|-----|
| Logistic Regression | 0.8093 | 0.8447 | 0.6667 | 0.5633 | 0.6106 | 0.4885 |
| Decision Tree | 0.7686 | 0.7502 | 0.5700 | 0.5223 | 0.5451 | 0.3910 |
| kNN | 0.7501 | 0.7698 | 0.5299 | 0.5205 | 0.5252 | 0.3557 |
| Naive Bayes | 0.6507 | 0.8101 | 0.4234 | 0.8717 | 0.5699 | 0.3926 |
| Random Forest (Ensemble) | 0.7842 | 0.8245 | 0.6196 | 0.4848 | 0.5440 | 0.4106 |
| XGBoost (Ensemble) | 0.7870 | 0.8150 | 0.6168 | 0.5223 | 0.5656 | 0.4284 |

**Best Model Overall:** **Logistic Regression** (achieved highest scores in 5 out of 6 metrics)

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved the best overall performance with 80.93% accuracy and 0.8447 AUC. High precision (0.6667) makes it excellent for minimizing false alarms in retention campaigns. The model demonstrates strong discriminative ability and is highly interpretable, making it ideal for business deployment. Balances performance and explainability effectively. |
| **Decision Tree** | Showed moderate performance with 76.86% accuracy and balanced precision-recall (around 52-57%). The lowest AUC (0.7502) indicates weaker discriminative ability. While the model offers excellent interpretability through decision rules, it appears to underfit the data. May benefit from hyperparameter tuning (max_depth, min_samples_split) to improve generalization. |
| **kNN** | Exhibited below-average performance with 75.01% accuracy and lowest MCC (0.3557). The model struggles with the high-dimensional feature space (30 features) due to the curse of dimensionality. Distance-based classification is less effective when features vary in scale and importance. Computational cost during prediction is high as it requires distance calculations to all training samples. |
| **Naive Bayes** | Demonstrated unique characteristics with lowest accuracy (65.07%) but highest recall (0.8717). Excels at identifying actual churners, catching 87% of them, but suffers from many false positives (precision: 0.4234). The feature independence assumption is violated in this dataset, affecting precision. Best suited for scenarios where missing a churner is very costly, such as high-value customer retention programs. |
| **Random Forest (Ensemble)** | Delivered solid performance with 78.42% accuracy and strong AUC (0.8245). High precision (0.6196) but lower recall (0.4848) indicates conservative predictions. The ensemble approach reduces overfitting compared to single decision trees. Provides valuable feature importance information for business insights. The 16MB model size is larger than other models, impacting deployment efficiency. |
| **XGBoost (Ensemble)** | Achieved second-best accuracy (78.70%) with well-balanced metrics across the board. Strong MCC (0.4284) and consistent performance indicate excellent generalization capability. Handles class imbalance effectively through gradient boosting. Compact model size (276KB) compared to Random Forest makes it deployment-friendly. Offers the best balance between performance, efficiency, and scalability for production environments. |

---

## Key Findings

### Best Models by Metric

- **Accuracy:** Logistic Regression (0.8093)
- **AUC (Discriminative Power):** Logistic Regression (0.8447)
- **Precision:** Logistic Regression (0.6667)
- **Recall:** Naive Bayes (0.8717)
- **F1 Score:** Logistic Regression (0.6106)
- **MCC:** Logistic Regression (0.4885)

### Insights

1. **Simple beats complex:** Logistic Regression outperformed ensemble methods, suggesting the churn relationship is relatively linear or that ensemble methods need hyperparameter tuning.

2. **Precision-Recall Trade-off:** Models optimize for different objectives:
   - High precision models (Logistic Regression, XGBoost, Random Forest) minimize false alarms
   - High recall model (Naive Bayes) catches more actual churners at the cost of false positives

3. **Class Imbalance Impact:** All models achieved better performance on the majority class (non-churners), with MCC scores reflecting the challenge of imbalanced data.

4. **Feature Engineering Matters:** One-hot encoding created 30 features, which benefited linear models but challenged distance-based models like kNN.

### Business Recommendations

- **Primary Model:** Deploy **Logistic Regression** for its superior overall performance (80.93% accuracy) and interpretability
- **Supplementary Screening:** Use **Naive Bayes** as an initial filter to identify high-risk customers for further analysis
- **Future Improvements:**
  - Apply SMOTE or class weighting to address imbalance
  - Perform hyperparameter tuning for ensemble methods
  - Implement cost-sensitive learning based on business costs of false positives vs false negatives
  - Feature selection to reduce dimensionality and improve kNN performance

---

## Repository Structure

```
ML-Assignment2/
├── data/
│   └── telco_customer_churn.csv          # Dataset file
├── model/
│   ├── train_telco_models.py             # Main training script
│   ├── logistic_regression.pkl            # Trained models (6 total)
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl                         # Feature scaler
│   ├── feature_names.pkl                  # Feature names
│   └── model_comparison_results.csv       # Results table
├── app.py                                 # Streamlit web application
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── model_observations.md                  # Detailed model analysis
└── Instructions.md                        # Assignment instructions
```

---

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd ML-Assignment2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (optional, pre-trained models included):**
   ```bash
   cd model
   python train_telco_models.py
   ```

4. **Run the Streamlit app locally:**
   ```bash
   streamlit run app.py
   ```

5. **Access the app:**
   Open your browser and navigate to `http://localhost:8501`

---

## Streamlit Application Features

The interactive web application provides:

1. **Dataset Upload:** Upload CSV files with customer data for churn prediction
2. **Model Selection:** Choose from 6 trained classification models
3. **Evaluation Metrics:** View all 6 performance metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. **Confusion Matrix:** Visualize prediction performance with heatmap
5. **Classification Report:** Detailed per-class performance statistics
6. **Prediction Interface:** Get real-time churn predictions on new customer data

---

## Technologies Used

- **Python 3.11**
- **Machine Learning:** scikit-learn, XGBoost
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Web Framework:** Streamlit
- **Model Persistence:** pickle

---

## Performance Summary

All models were trained on 4,930 samples and tested on 2,113 samples.

**Training Environment:**
- Split Ratio: 70% train, 30% test
- Stratification: Applied to maintain class distribution
- Feature Scaling: StandardScaler for Logistic Regression and kNN
- Random State: 42 (for reproducibility)

**Computational Details:**
- Training Time: < 2 minutes for all models combined
- Model Sizes: 1KB - 16MB (Random Forest largest)
- Inference Speed: < 100ms per prediction for all models

---

## Author

**Name:** [Your Name]
**Roll Number:** [Your Roll Number]
**Program:** M.Tech (AIML/DSE)
**Course:** Machine Learning
**Assignment:** 2

---

## License

This project is submitted as part of academic coursework for BITS Pilani M.Tech program.

---

## Acknowledgments

- Dataset Source: Kaggle - Telco Customer Churn
- BITS Pilani Work Integrated Learning Programmes Division
- Course Instructor: Machine Learning Faculty

---

## References

1. IBM Watson Analytics - Telco Customer Churn Dataset
2. Scikit-learn Documentation: https://scikit-learn.org/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. Streamlit Documentation: https://docs.streamlit.io/

---

**Note:** This README is part of the required submission and should be included in the final PDF along with GitHub and Streamlit app links.

# Credit Card Fraud Detection Dashboard

A cloud-based machine learning application that detects fraudulent credit card transactions using a **Random Forest Classifier**. The application is built with **Streamlit** and deployed on **AWS Elastic Beanstalk**, enabling users to upload transaction datasets or use a sample dataset stored in **Amazon S3** for fraud analysis.

## Features

- Upload credit card transaction datasets (CSV)
- Predict genuine and fraudulent transactions
- Fraud probability and confidence score for each transaction
- Interactive dashboard with summary metrics and visualizations
- Sample dataset integration from Amazon S3
- Downloadable prediction report
- Cloud deployment on AWS Elastic Beanstalk

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- AWS Elastic Beanstalk
- Amazon EC2 (t3.medium)
- Amazon S3

## Project Structure

```
credit-card-fraud-detection-dashboard/
│
├── .platform/                  # Elastic Beanstalk platform configuration
├── .streamlit/                 # Streamlit configuration
├── documents/                  # Data preprocessing and 4 model training and evaluation collab notebook file and Architectue diagram
├── fraud_dashboard/            # Dashboard components
├── sample_data/                # Sample transaction datasets
│
├── app.py                      # Streamlit application entry point
├── utils.py                    # Helper functions
├── model.pkl                   # Trained Random Forest model
├── scaler.pkl                  # StandardScaler used during training
├── requirements.txt            # Python dependencies
├── Procfile                    # Elastic Beanstalk startup configuration
└── .gitignore
```
## Architecture Diagram

![Architecture Diagram](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Credit%20Card%20Fraud%20Detection%20Dashboard%20Architecture%20Diagram%20with%20AWS.png)

## Data Analysis & Preprocessing Visualizations

### Exploratory Data Analysis (EDA)

#### 1. Missing Value Analysis
![Missing Value Analysis](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Missing_Value_Analysis.png)

#### 2. Class Distribution (Original Data)
![Class Distribution Original](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Class_Distribution_Original.png)

#### 3. Transaction Amount & Time Distributions
![Transaction Amount Distribution](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Transaction_Amount_Distribution.png)
![Transaction Time Distribution](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Transaction_Time_Distribution.png)

#### 4. Correlation Heatmap & Top Features
![Correlation Heatmap](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Correlation_Heatmap.png)
![Top Correlated Features Distribution](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Top_Correlated_Features_Distribution.png)

#### 5. Feature Relationships & Outlier Analysis
![Amount and Time by Class](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Amount_and_Time_by_Class.png)
![Outlier Analysis](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Outlier_Analysis.png)
![Log Transformed Amount Comparison](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Log_Transformed_Amount_Comparison.png)

#### 6. Resampling (SMOTE Class Distribution)
![Class Distribution After SMOTE](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Class_Distribution_After_SMOTE.png)

## Model Evaluation & Performance Visualizations

### Overall Model Performance Comparison
![Model Performance Comparison](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Model_Performance_Comparison.png)

### 1. Random Forest Classifier (Primary Model)
![Random Forest Confusion Matrix](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Random_Forest_Confusion_Matrix.png)
![Random Forest ROC Curve](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Random_Forest_ROC_Curve.png)
![Random Forest Precision Recall Curve](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Random_Forest_Precision_Recall_Curve.png)
![Random Forest Error Analysis](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Random_Forest_Error_Analysis.png)

### 2. Logistic Regression Model
![Logistic Regression Confusion Matrix](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Logistic_Regression_Confusion_Matrix.png)
![Logistic Regression ROC Curve](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Logistic_Regression_ROC_Curve.png)
![Logistic Regression Precision Recall Curve](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Logistic_Regression_Precision_Recall_Curve.png)
![Logistic Regression Error Analysis](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/Logistic_Regression_Error_Analysis.png)

### 3. XGBoost Classifier Model
![XGBoost Confusion Matrix](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/XGBoost_Confusion_Matrix.png)
![XGBoost ROC Curve](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/XGBoost_ROC_Curve.png)
![XGBoost Precision Recall Curve](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/XGBoost_Precision_Recall_Curve.png)
![XGBoost Error Analysis](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/XGBoost_Error_Analysis.png)

### 4. Multi-Layer Perceptron (MLP) Classifier
![MLP Classifier Confusion Matrix](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/MLP_Classifier_Confusion_Matrix.png)
![MLP Classifier ROC Curve](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/MLP_Classifier_ROC_Curve.png)
![MLP Classifier Precision Recall Curve](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/MLP_Classifier_Precision_Recall_Curve.png)
![MLP Classifier Error Analysis](https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML/blob/main/documents/MLP_Classifier_Error_Analysis.png)


## Getting Started

Clone the repository:

```bash
git clone <https://github.com/ajayanithaganesan/Credit-Card-Fraud-Detection-Dashboard_CML>
cd credit-card-fraud-detection-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application locally:

```bash
streamlit run app.py
```

## Deployment

The application is deployed on **AWS Elastic Beanstalk** using an **Amazon EC2 t3.medium** instance. Sample transaction data is stored in **Amazon S3** and accessed through Elastic Beanstalk environment variables.

## Machine Learning Pipeline

- Data preprocessing and feature engineering
- Feature scaling using StandardScaler
- Random Forest model training and evaluation
- Model serialization using Joblib
- Cloud deployment for real-time fraud prediction

## License

Developed for academic purposes as part of the MSc Cloud Computing Machine Learning project.

## Team Members

- Ajay Anitha Ganesan
- Dharanish Punjaipuliampatti Mayilsamy
- Arun Natrajan
- Aravind Singam


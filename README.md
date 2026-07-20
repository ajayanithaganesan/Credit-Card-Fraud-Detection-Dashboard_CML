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


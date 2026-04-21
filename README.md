#  Customer Churn Prediction App

A complete **end-to-end Machine Learning project** to predict telecom customer churn and deploy it as an interactive web application.


##  Live Demo

 **Try the app here:**  
 https://churn-prediction-app-23vn0604.streamlit.app/


##  Project Overview

Customer churn is a major problem in the telecom industry. This project predicts whether a customer will churn based on usage patterns, service interactions, and plan details.

###  Solution Includes:
- Data preprocessing & EDA  
- Machine Learning model building  
- Model evaluation & feature importance  
- Deployment using Streamlit  


##  Objective

To build a predictive system that:
- Identifies customers at risk of churn  
- Provides probability-based predictions  
- Explains key factors influencing churn  


##  Dataset

- **Total Records:** 3333 customers  
- **Target Variable:** `churn` (0 = No, 1 = Yes)

### Features Include:
- Account details (account length, plans)  
- Usage metrics (day, evening, night, international minutes)  
- Call behavior (number of calls)  
- Customer service interactions  


##  Project Workflow

###  Data Cleaning
- Removed redundant features (charges highly correlated with minutes)  
- Dropped unnecessary columns like `voice_mail_messages`  

###  Outlier Handling
- Applied **Winsorization (1%–99%)** to cap extreme values  

###  Exploratory Data Analysis (EDA)
- Customers with **international plans** churn more  
- High **customer service calls → higher churn**  
- Usage behavior impacts churn  

###  Preprocessing
- Missing value imputation (median)  
- Feature scaling using **StandardScaler**  

###  Handling Imbalance
- Applied **SMOTE** to balance churn and non-churn classes  


##  Model Building

Trained and compared multiple models:

- Logistic Regression  
- Gradient Boosting  
- Random Forest  
-  **XGBoost (Final Model)**  


##  Model Performance

| Model                | Accuracy | Recall (Churn) | ROC-AUC |
|---------------------|----------|----------------|--------|
| Logistic Regression | 76%      | 71%            | 0.81   |
| Gradient Boosting   | 91%      | 72%            | 0.89   |
| Random Forest       | 92%      | 73%            | 0.89   |
|  XGBoost          | **94%**  | **78%**        | **0.90** |

 **XGBoost is selected as the final model** due to best overall performance.


##  Key Insights

-  Customers with more **service calls** are likely to churn  
-  Customers with **international plans** have higher churn risk  
-  Usage patterns (especially daytime minutes) influence churn  
-  Feature importance highlights actionable business insights  


##  Web Application Features

- User-friendly interface to input customer details  
- Predicts churn probability instantly  
- Displays **risk level (Low / High)**  
- Shows **top contributing factors** for prediction  


##  Model Explainability

The app provides:
- **Churn probability (%)**  
- **Top 5 features influencing prediction**  
- Visual impact (positive/negative contribution)  


##  Project Structure
- app.py
- xgb_churn_model.joblib
- scaler.joblib
- imputer.joblib
- telecommunications_Dataset.csv
- requirements.txt
- EXCELR PROJECT MODEL BUILDING.ipynb
 

##  Future Improvements

- Hyperparameter tuning  
- Cloud deployment improvements  
- API integration  
- Business dashboard  


##  Author

**Narendra Damera**  


##  Conclusion

This project demonstrates a complete ML pipeline from data preprocessing to deployment.  
The final **XGBoost model achieves 94% accuracy**, making it highly effective for churn prediction.


 If you like this project, consider giving it a star!

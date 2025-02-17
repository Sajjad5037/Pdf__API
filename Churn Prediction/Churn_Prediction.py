import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace this with your actual dataset path
df = pd.read_csv('customer_churn_data.csv')

# Check the first few rows of the data
print(df.head())

# Data Preprocessing
# Handle missing values if needed (e.g., fill or drop)
df = df.fillna(df.mean())  # Example: fill missing values with column means

# Feature selection (assuming 'Churn' is the target column)
X = df.drop(['Churn'], axis=1)  # Features
y = df['Churn']  # Target variable

# Convert categorical columns to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (important for models like Logistic Regression and Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Random Forest Classifier Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation of models
def evaluate_model(y_test, y_pred):
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.show()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

# Evaluate Logistic Regression
print('Logistic Regression Evaluation:')
evaluate_model(y_test, y_pred_log_reg)

# Evaluate Random Forest
print('Random Forest Evaluation:')
evaluate_model(y_test, y_pred_rf)

# Evaluate XGBoost
print('XGBoost Evaluation:')
evaluate_model(y_t

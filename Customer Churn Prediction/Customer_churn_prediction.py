import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Step 1: Load the dataset (replace with your file path)
csv_file_name = 'customer_churn.csv'
df = pd.read_csv(csv_file_name)

# Step 2: Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values (depending on your data)
df.fillna(df.mean(), inplace=True)  # Simple imputation for missing numerical values

# Encode categorical variables (for example, 'gender', 'subscription_type', etc.)
df = pd.get_dummies(df, drop_first=True)

# Step 3: Split data into features (X) and target (y)
X = df.drop('churn', axis=1)  # 'churn' is the target column
y = df['churn']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling (for better model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the model (Logistic Regression as an example)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"ROC-AUC Score: {roc_auc:.2f}")

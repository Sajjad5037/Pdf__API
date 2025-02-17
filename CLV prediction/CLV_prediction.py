import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset (Replace 'customer_data.csv' with actual file)
df = pd.read_csv('customer_data.csv')

# Feature Engineering
df['total_orders'] = df.groupby('customer_id')['order_amount'].transform('count')
df['total_spent'] = df.groupby('customer_id')['order_amount'].transform('sum')
df['avg_order_value'] = df['total_spent'] / df['total_orders']
df['customer_age_days'] = (pd.to_datetime(df['last_purchase_date']) - pd.to_datetime(df['first_purchase_date'])).dt.days

df = df.drop_duplicates(subset=['customer_id'])

drop_columns = ['customer_id', 'first_purchase_date', 'last_purchase_date']
df = df.drop(columns=drop_columns)

# Prepare data for training
X = df.drop(columns=['total_spent'])  # Features
y = df['total_spent']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

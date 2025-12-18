
import pandas as pd

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("Loading and preparing data...")
#load data
training_data = pd.read_csv("../Data/sales_with_weather_tx_train.csv")

# cut down to smaller size for testing
training_data = training_data.sample(n=5000, random_state=42).reset_index(drop=True)    
#prepare data
y_train = training_data["product_detail"]
X_train = training_data.drop(columns=["product_detail"])

print(f"Data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print("Columns:", X_train.columns.tolist())
print("Applying one-hot encoding to categorical features...")
onehot_cols = X_train.select_dtypes(include=['object']).columns
X_train = pd.get_dummies(X_train, columns=onehot_cols, drop_first=True)

print(f"Data shape after one-hot encoding: {X_train.shape}")
#split data using group shuffle split
print("Performing group-wise data split based on product categories...")
groups = training_data['product_category']
gss = GroupShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3, random_state=0)
tr_idx, val_idx = next(gss.split(X_train, y_train, groups=groups))
X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
groups_tr = groups.iloc[tr_idx]
print(f"Train rows: {len(X_tr)} | Val rows: {len(X_val)}")

print("Setting up and training Logistic Regression model...")
#create pipeline
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])  

#train model

logreg_pipeline.fit(X_tr, y_tr)
#validate model
train_accuracy = logreg_pipeline.score(X_tr, y_tr)
print(f"Training Accuracy: {train_accuracy:.4f}")

y_val_pred = logreg_pipeline.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")   

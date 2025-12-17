import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# 1. Load data
print("Loading and preparing data...")
training_data = pd.read_csv("../Data/sales_with_weather_tx_train.csv")

# Drop ID columns that aren't features
if "product_id" in training_data.columns:
    training_data = training_data.drop(columns=["product_id"])

# 2. Prepare X and y
y_train = training_data["product_detail"]
X_train = training_data.drop(columns=["product_detail"])

print(f"Data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print("Columns:", X_train.columns.tolist())

# 3. Sparse One-Hot Encoding
print("Applying sparse one-hot encoding to categorical features...")
onehot_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(exclude=['object']).columns

# Keep numeric cols as-is
X_numeric = X_train[numeric_cols].values

# One-hot encode categorical cols as sparse matrix
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_categorical_sparse = encoder.fit_transform(X_train[onehot_cols])

# Combine numeric + sparse categorical
X_train_sparse = hstack([X_numeric, X_categorical_sparse])

# Convert to CSR format for efficient indexing
X_train_sparse = X_train_sparse.tocsr()

print(f"Sparse data shape: {X_train_sparse.shape}, sparsity: {1 - X_train_sparse.nnz / (X_train_sparse.shape[0] * X_train_sparse.shape[1]):.2%}")

# 4. Split Data
print("Performing random train/test split...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train_sparse, y_train, test_size=0.3, random_state=0)

print(f"Train rows: {X_tr.shape[0]} | Val rows: {X_val.shape[0]}")

# 5. Setup and Train KNN Model
print("Setting up and training KNN model (this may take a moment)...")

# Note: n_neighbors=5 is standard. 
# with_mean=False is REQUIRED for sparse matrices to avoid exploding memory.
knn_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
])

knn_pipeline.fit(X_tr, y_tr)

# 6. Validate Model
print("Evaluating model...")
train_accuracy = knn_pipeline.score(X_tr, y_tr)
print(f"Training Accuracy: {train_accuracy:.4f}")

val_accuracy = knn_pipeline.score(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 7. Random Sample Inspection
print("\n--- Random Sample Predictions from Validation Set ---")
sample_size = 10
n_val = X_val.shape[0]
if n_val < sample_size:
    sample_size = n_val

random_indices = np.random.choice(n_val, size=sample_size, replace=False)

# Get the samples
X_sample = X_val[random_indices]
y_sample_actual = y_val.iloc[random_indices].values

# Predict
y_sample_pred = knn_pipeline.predict(X_sample)
results_df = pd.DataFrame({
'Actual': y_sample_actual,
'Predicted': y_sample_pred,
'Match': y_sample_actual == y_sample_pred
})

print(results_df)
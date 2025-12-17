
import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GroupShuffleSplit, train_test_split, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



#read data from file
print("Loading data...")
training_data = pd.read_csv("../Data/sales_with_weather_tx_test.csv")
testing_data = pd.read_csv("../Data/sales_with_weather_tx_test.csv")


# cut down to smaller size for testing
training_data = training_data.sample(n=5000, random_state=42).reset_index(drop=True)    
# Use training data only for splits (don't concatenate train+test yet)
y = training_data["product_detail"]
X = training_data.drop(columns=["product_detail"])

print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
print("Columns:", X.columns.tolist())

# one-hot encode
print("Applying one-hot encoding to categorical features...")
one_hot_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=one_hot_cols, drop_first=True)

print(f"Data shape after one-hot encoding: {X.shape}")

# group-wise data split
print("Performing group-wise data split based on product categories...")

subj_counts_df = training_data.groupby('product_category').size().reset_index(name='n_samples')

groups = training_data['product_category']

print("Groups distribution:")
gss = GroupShuffleSplit(n_splits=1, train_size=0.9, test_size=0.1, random_state=0)

tr_idx, val_idx = next(gss.split(X, y, groups=groups))

X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

print("Overlap categories (should be empty):",
      np.intersect1d(groups.iloc[tr_idx], groups.iloc[val_idx]))

print(f"Train rows: {len(X_train)} | Val rows: {len(X_val)}")

#train SVM to predict specific product for each product category for the given time and weather 

# pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ("linearsvc", LinearSVC())
])

# groups
groups_train = groups.iloc[tr_idx]

cv = GroupKFold(n_splits=5)

# gridsearch

gs = RandomizedSearchCV(
    estimator=svm_pipeline,
    param_distributions={
        'linearsvc__C': np.logspace(-4, 4, 20),
        'linearsvc__max_iter': [1000, 5000, 10000],
        'linearsvc__tol': [1e-4, 1e-3, 1e-2]
    },
    n_iter=5,
    cv=cv,
    scoring='accuracy',
    refit=True,
    n_jobs=-1,
    verbose=3,
    random_state=0
)

gs.fit(X_train, y_train, groups=groups_train)

predictions = gs.predict(y_train)
accuracy = accuracy_score(y_train, predictions)
print(f"Training Accuracy: {accuracy}")

predictions_val = gs.predict(X_val)
accuracy_val = accuracy_score(y_val, predictions_val)
print(f"Validation Accuracy: {accuracy_val}")

#save model?

#serve predictions?
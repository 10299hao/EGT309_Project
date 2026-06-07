# =====================================================
# SUPPORT VECTOR CLASSIFIER FOR GAS MONITORING DATASET
# =====================================================

import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# STEP 1 - CONNECT TO DATABASE
# =====================================================

DATABASE_FILE = "gas_monitoring.db"

conn = sqlite3.connect(DATABASE_FILE)

# =====================================================
# STEP 2 - FIND AVAILABLE TABLES
# =====================================================

tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)

print("\nAvailable Tables:")
print(tables)

# =====================================================
# STEP 3 - LOAD DATA
# =====================================================

TABLE_NAME = input(
    "\nEnter table name exactly as shown above: "
)

df = pd.read_sql_query(
    f"SELECT * FROM {TABLE_NAME}",
    conn
)

conn.close()

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# =====================================================
# STEP 4 - IDENTIFY TARGET COLUMN
# =====================================================

print("\nColumns:")
for col in df.columns:
    print(col)

target_column = input(
    "\nEnter the target column name: "
)

# =====================================================
# STEP 5 - SEPARATE FEATURES & TARGET
# =====================================================

X = df.drop(columns=[target_column])
y = df[target_column]

# =====================================================
# STEP 6 - IDENTIFY FEATURE TYPES
# =====================================================
#I AM GOING TO DIE
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

print("\nNumeric Features:")
print(list(numeric_features))

print("\nCategorical Features:")
print(list(categorical_features))

# =====================================================
# STEP 7 - PREPROCESSING
# =====================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# =====================================================
# STEP 8 - TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =====================================================
# STEP 9 - BUILD SVC MODEL
# =====================================================

svc_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale"
            )
        )
    ]
)

# =====================================================
# STEP 10 - TRAIN MODEL
# =====================================================

print("\nTraining SVC Model...")

svc_pipeline.fit(
    X_train,
    y_train
)

print("Training Complete!")

# =====================================================
# STEP 11 - PREDICTIONS
# =====================================================

y_pred = svc_pipeline.predict(
    X_test
)

# =====================================================
# STEP 12 - EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==========================")
print("MODEL PERFORMANCE")
print("==========================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# =====================================================
# STEP 13 - SAVE MODEL
# =====================================================

joblib.dump(
    svc_pipeline,
    "svc_gas_monitoring_model.pkl"
)

print(
    "\nModel saved as:"
)
print(
    "svc_gas_monitoring_model.pkl"
)

print(
    "\nProcess Completed Successfully!"
)
# catboost_gas.py
import sqlite3
import pandas as pd
import os
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATABASE_FILE = "gas_monitoring.db"

if not os.path.exists(DATABASE_FILE):
    raise FileNotFoundError("Database file not found. Please ensure 'gas_monitoring.db' exists.")

conn = sqlite3.connect(DATABASE_FILE)
query = "SELECT * FROM gas_data"  # Replace with actual table name
df = pd.read_sql_query(query, conn)
conn.close()

X = df.drop("target", axis=1)   # Replace 'target' with actual target column
y = df["target"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=False
)

train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

model.fit(train_pool, eval_set=test_pool)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "catboost_gas_model.pkl")
print("Model saved as 'catboost_gas_model.pkl'")
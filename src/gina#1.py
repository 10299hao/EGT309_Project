
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

#python class for preprocessing
class TemperatureUnitNormalizer(BaseEstimator, TransformerMixin):
    """
    A custom transformer class to scan for mixed temperature units (Celsius/Kelvin)
    and automatically standardize them to Celsius.
    """
    def __init__(self, temperature_col='Temperature', threshold=100.0):
        self.temperature_col = temperature_col
        self.threshold = threshold
        
    def fit(self, X, y=None):
        # Transformers that don't compute statistics from the data just return self
        return self
        
    def transform(self, X):
        # Create a copy to prevent modifying the original dataframe slice unexpectedly
        X_built = X.copy()
        
        if self.temperature_col in X_built.columns:
            # Locate rows where the scale jumps over the threshold and step it down
            is_kelvin = X_built[self.temperature_col] > self.threshold
            X_built.loc[is_kelvin, self.temperature_col] = X_built.loc[is_kelvin, self.temperature_col] - 273.15
            
        return X_built
    
#data extraction & label standardisation
db_path = 'data/gas_monitoring.db' 
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
conn.close()

df = df.drop_duplicates()

# Basic mapping functions for high-level structure cleanup
df['HVAC Operation Mode'] = df['HVAC Operation Mode'].astype(str).str.strip().str.lower()
target_label_mapping = {
    'High Activity': 'High Activity', 'Low Activity': 'Low Activity',
    'LowActivity': 'Low Activity', 'Low_Activity': 'Low Activity',
    'Moderate Activity': 'Moderate Activity', 'ModerateActivity': 'Moderate Activity'
}
df['Activity Level'] = df['Activity Level'].map(target_label_mapping)
target_numeric_map = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
df['Target'] = df['Activity Level'].map(target_numeric_map)
df = df.dropna(subset=['Target'])

X = df.drop(columns=['Session ID', 'Activity Level', 'Target'])
y = df['Target']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#data pipeine using class
sub_numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

sub_categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Structural transformation block
core_feature_processor = ColumnTransformer(
    transformers=[
        ('num', sub_numerical_pipeline, numerical_cols),
        ('cat', sub_categorical_pipeline, categorical_cols)
    ]
)

# Global Master Pipeline combining our custom class execution block with downstream steps
full_pipeline = Pipeline(steps=[
    ('temp_normalizer', TemperatureUnitNormalizer(temperature_col='Temperature', threshold=100.0)),
    ('feature_processing', core_feature_processor)
])

# Process the structured sets safely using the integrated pipelines
X_train_processed = full_pipeline.fit_transform(X_train)
X_test_processed = full_pipeline.transform(X_test)

#MODEL 1: XGBOOST
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

print("Training XGBoost Classifier via Custom Class Pipeline Integration...")

xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=5,
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train_processed, y_train, sample_weight=sample_weights)
y_pred_xgb = xgb_model.predict(X_test_processed)

print(f"XGBoost Overall Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=list(target_numeric_map.keys())))

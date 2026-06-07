import os
import sqlite3
import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

##python class for preprocessing
class TemperatureUnitNormalizer(BaseEstimator, TransformerMixin):
    """
    A custom transformer class to scan for mixed temperature units (Celsius/Kelvin)
    and automatically standardize them down to Celsius.
    """
    def __init__(self, temperature_col='Temperature', threshold=100.0):
        self.temperature_col = temperature_col
        self.threshold = threshold
        
    def fit(self, X, y=None):
        # Statless transformers do not need to learn internal variables, so we simply return self
        return self
        
    def transform(self, X):
        # Create a deep copy to preserve the original dataframe structures safely
        X_built = X.copy()
        
        if self.temperature_col in X_built.columns:
            # Mask data points where values jump past 100 and transform them
            is_kelvin = X_built[self.temperature_col] > self.threshold
            X_built.loc[is_kelvin, self.temperature_col] = X_built.loc[is_kelvin, self.temperature_col] - 273.15
            
        return X_built
    
##data extraction & label standardisation
db_path = 'data/gas_monitoring.db' 
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
conn.close()

# Drop exact duplicate rows
df = df.drop_duplicates()

# Standardize text inconsistencies 
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

# Isolate features X and targets y
X = df.drop(columns=['Session ID', 'Activity Level', 'Target'])
y = df['Target']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Chart 1: Target Distribution 
plt.figure(figsize=(6, 4))
sns.countplot(x='Activity Level', data=df, order=['Low Activity', 'Moderate Activity', 'High Activity'], hue='Activity Level', palette='viridis', legend=False)
plt.title('k-NN File - Distribution of Resident Activity Levels')
plt.xlabel('Activity Status')
plt.ylabel('Total Count')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'knn_target_distribution.png'), dpi=300)
plt.close()

# Chart 2: Temperature Boxplot 
plt.figure(figsize=(8, 5))
sns.boxplot(x='Activity Level', y='Temperature', data=df, hue='Activity Level', palette='Set2', legend=False)
plt.title('k-NN File - Raw Temperature Variations by Activity Level')
plt.ylabel('Temperature')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'knn_temperature_boxplot.png'), dpi=300)
plt.close()

# 80/20 Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

##data pipeline w class
# Sub-components for column types
sub_numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Essential for distance calculation models like k-NN
])

sub_categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Structural transformation mapping 
core_feature_processor = ColumnTransformer(
    transformers=[
        ('num', sub_numerical_pipeline, numerical_cols),
        ('cat', sub_categorical_pipeline, categorical_cols)
    ]
)

# Global Master Pipeline combining the custom transformer, column preprocessing, and k-NN classifier
knn_master_pipeline = Pipeline(steps=[
    ('temp_normalizer', TemperatureUnitNormalizer(temperature_col='Temperature', threshold=100.0)),
    ('feature_processing', core_feature_processor),
    ('classifier', KNeighborsClassifier(n_neighbors=5, weights='distance')) 
])

#KNN MODEL
print("Training k-NN Classifier via Custom Class Pipeline Integration...")

#With the integrated master pipeline, calling fit applies the custom 
# class conversion, scales the numerical components, and fits the neighbor matrix.
knn_master_pipeline.fit(X_train, y_train)

# Call predict on the raw test set; it passes securely through all preprocessing stages automatically
y_pred_knn = knn_master_pipeline.predict(X_test)

print(f"k-NN Overall Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=list(target_numeric_map.keys())))

# Chart 3: Confusion Matrix Heatmap (Configured with Greens for k-NN to distinguish it from XGBoost)
cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=list(target_numeric_map.keys()))
disp.plot(cmap='Greens', values_format='d')
plt.title('k-NN Confusion Matrix Heatmap')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'knn_confusion_matrix.png'), dpi=300)
plt.close()

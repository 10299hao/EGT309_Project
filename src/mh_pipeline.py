"""
mh_pipeline.py
------------
End-to-end ML pipeline for classifying room activity level from gas
and environmental sensor readings.
"""
import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

def suppress_all_warnings(*args, **kwargs):
    pass
warnings.warn = suppress_all_warnings
warnings.showwarning = suppress_all_warnings
# -----------------------------------
import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import custom model functions
from mh_RF import train_random_forest
from mh_mlp import train_mlp


TARGET = "Activity Level"
CLASS_ORDER = ["Low Activity", "Moderate Activity", "High Activity"]

ACTIVITY_MAP: dict[str, str] = {
    "Low Activity": "Low Activity",
    "Low_Activity": "Low Activity",
    "LowActivity": "Low Activity",
    "Moderate Activity": "Moderate Activity",
    "ModerateActivity": "Moderate Activity",
    "High Activity": "High Activity",
}

# Restored ALL raw sensors - Tree models and NNs benefit from maximum feature availability
NUMERIC_COLS: list[str] = [
    "Temperature", "Humidity", "CO2_InfraredSensor", "CO2_ElectroChemicalSensor",
    "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2", "MetalOxideSensor_Unit3",
    "MetalOxideSensor_Unit4", "CO_GasSensor", "CO2_Average", "TotalMOS",
    "CO2_CO_Ratio", "TimeOfDay_Ordinal",
]

CATEGORICAL_COLS: list[str] = [
    "Time of Day", "HVAC Operation Mode", "Ambient Light Level",
]

LOGGER = logging.getLogger(__name__)

class GasActivityPipeline:
    def __init__(self, db_path: Path, table_name: str = "gas_monitoring", test_size: float = 0.2, random_state: int = 42, cv_folds: int = 5):
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models: dict[str, Any] = {}
        self.results: dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {self.table_name}", conn)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[TARGET] = df[TARGET].map(ACTIVITY_MAP)
        df.loc[df["Temperature"] > 40, "Temperature"] = np.nan
        df.loc[(df["Humidity"] < 0) | (df["Humidity"] > 100), "Humidity"] = np.nan
        for col in CATEGORICAL_COLS:
            df[col] = self._normalize_text(df[col])
        return df

    @staticmethod
    def _normalize_text(series: pd.Series) -> pd.Series:
        cleaned = series.copy()
        mask = cleaned.notna()
        cleaned.loc[mask] = cleaned.loc[mask].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)
        return cleaned.astype(object)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["CO2_Average"] = (df["CO2_InfraredSensor"] + df["CO2_ElectroChemicalSensor"]) / 2
        df["TotalMOS"] = (df["MetalOxideSensor_Unit1"] + df["MetalOxideSensor_Unit2"] + df["MetalOxideSensor_Unit3"] + df["MetalOxideSensor_Unit4"])
        df["CO2_CO_Ratio"] = df["CO2_Average"] / (df["CO_GasSensor"] + 1)
        time_map = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}
        df["TimeOfDay_Ordinal"] = df["Time of Day"].map(time_map)
        return df

    def split(self, df: pd.DataFrame):
        y = df[TARGET]
        X = df.drop(columns=[TARGET, "Session ID"], errors="ignore")
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

    def build_preprocessor(self) -> ColumnTransformer:
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        return ColumnTransformer([("num", numeric, NUMERIC_COLS), ("cat", categorical, CATEGORICAL_COLS)])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
        preprocessor = self.build_preprocessor()
        self.models["random_forest"] = train_random_forest(preprocessor, X_train, y_train, self.cv_folds, self.random_state)
        self.models["mlp_neural_net"] = train_mlp(preprocessor, X_train, y_train, self.cv_folds, self.random_state)
        return self.models

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        pred = model.predict(X_test)
        return {
            "best_params": model.best_params_,
            "best_cv_score": model.best_score_,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "weighted_f1": f1_score(y_test, pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=CLASS_ORDER).tolist(),
        }

    def run(self) -> dict[str, Any]:
        print("Step 1/5: Loading data...")
        df = self.clean_data(self.load_data())
        print("Step 2/5 & 3/5: Cleaning and Engineering...")
        df = self.add_features(df)
        print("Step 4/5: Splitting and training models...")
        X_train, X_test, y_train, y_test = self.split(df)
        self.train(X_train, y_train)
        print("Step 5/5: Evaluating models...")
        self.results = {name: self.evaluate(model, X_test, y_test) for name, model in self.models.items()}
        return self.results

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=Path("data/gas_monitoring.db"))
    parser.add_argument("--table-name", default="gas_monitoring")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    pipeline = GasActivityPipeline(db_path=args.db_path, table_name=args.table_name, test_size=args.test_size, random_state=args.random_state, cv_folds=args.cv_folds)
    print(json.dumps(pipeline.run(), indent=2))

if __name__ == "__main__":
    main()
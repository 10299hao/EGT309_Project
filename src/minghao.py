import argparse
import json
import logging
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


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

NUMERIC_COLS: list[str] = [
    "Temperature",
    "Humidity",
    "CO2_InfraredSensor",
    "CO2_ElectroChemicalSensor",
    "MetalOxideSensor_Unit1",
    "MetalOxideSensor_Unit2",
    "MetalOxideSensor_Unit3",
    "MetalOxideSensor_Unit4",
    "CO_GasSensor",
    "CO2_Average",
    "TotalMOS",
    "CO2_CO_Ratio",
    "TimeOfDay_Ordinal",
]

CATEGORICAL_COLS: list[str] = [
    "Time of Day",
    "HVAC Operation Mode",
    "Ambient Light Level",
]

LOGGER = logging.getLogger(__name__)


#class for pipeline
class GasActivityPipeline:
    """
    End-to-end ML pipeline for classifying room activity level from gas
    and environmental sensor readings.

    SMOTE (Synthetic Minority Oversampling Technique) is applied after
    preprocessing inside the training pipeline to address the class imbalance
    between Low Activity (majority) and High Activity (minority). SMOTE is
    only applied to the training fold during cross-validation — never to the
    test set — which prevents data leakage.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    table_name : str
        Name of the table containing sensor readings.
    test_size : float
        Fraction of data held out for evaluation (0–1).
    random_state : int
        Seed for all random operations, ensuring reproducibility.
    cv_folds : int
        Number of stratified folds used in cross-validation.
    """

    def __init__(
        self,
        db_path: Path,
        table_name: str = "gas_monitoring",
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
    ) -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds

        self.models: dict[str, GridSearchCV] = {}
        self.results: dict[str, Any] = {}

    # =========================
    # LOAD DATA
    # =========================
    def load_data(self) -> pd.DataFrame:
        """Read all rows from the configured SQLite table."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {self.table_name}", conn)

        LOGGER.info("Loaded %d rows from '%s'", len(df), self.table_name)
        return df

    # =========================
    # CLEAN DATA
    # =========================
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise labels, remove physically impossible sensor readings,
        and normalise text columns.

        Raises
        ------
        ValueError
            If any target label cannot be mapped to a known activity class.
        """
        df = df.copy()

        # Normalise label variants (e.g. "Low_Activity" → "Low Activity")
        df[TARGET] = df[TARGET].map(ACTIVITY_MAP)
        if df[TARGET].isna().any():
            raise ValueError(
                "Unknown labels found in target column. "
                "Update ACTIVITY_MAP to include all label variants."
            )
        
        # Nullify physically impossible sensor values
        df.loc[df["Temperature"] > 40, "Temperature"] = np.nan
        df.loc[(df["Humidity"] < 0) | (df["Humidity"] > 100), "Humidity"] = np.nan

        # Normalise text columns to lowercase with spaces (no underscores)
        for col in CATEGORICAL_COLS:
            df[col] = self._normalize_text(df[col])

        return df

    @staticmethod
    def _normalize_text(series: pd.Series) -> pd.Series:
        """
        Strip whitespace, lowercase, and replace underscores with spaces.
        Preserves NaN values.
        """
        cleaned = series.copy()
        mask = cleaned.notna()

        cleaned.loc[mask] = (
            cleaned.loc[mask]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
        )

        cleaned.loc[~mask] = np.nan
        return cleaned.astype(object)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive composite features from raw sensor columns.

        New columns
        -----------
        CO2_Average      : Mean of infrared and electrochemical CO2 sensors,
                           reducing noise from a single sensor.
        TotalMOS         : Sum of all metal-oxide sensor units, representing
                           overall volatile organic compound (VOC) exposure.
        CO2_CO_Ratio     : CO2 relative to CO (+ 1 to avoid division by zero),
                           useful for distinguishing combustion vs occupancy.
        TimeOfDay_Ordinal: Ordinal encoding of time-of-day (night=0 … evening=3)
                           to preserve temporal ordering.
        """
        df = df.copy()

        df["CO2_Average"] = (
            df["CO2_InfraredSensor"] + df["CO2_ElectroChemicalSensor"]
        ) / 2

        df["TotalMOS"] = (
            df["MetalOxideSensor_Unit1"]
            + df["MetalOxideSensor_Unit2"]
            + df["MetalOxideSensor_Unit3"]
            + df["MetalOxideSensor_Unit4"]
        )

        df["CO2_CO_Ratio"] = df["CO2_Average"] / (df["CO_GasSensor"] + 1)

        time_map = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}
        df["TimeOfDay_Ordinal"] = df["Time of Day"].map(time_map)

        return df

    # =========================
    # SPLIT
    # =========================
    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Stratified train/test split so each activity class is proportionally
        represented in both sets.
        """
        y = df[TARGET]
        X = df.drop(columns=[TARGET, "Session ID"], errors="ignore")

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    # =========================
    # PREPROCESSOR
    # =========================
    def build_preprocessor(self) -> ColumnTransformer:
        """
        Build a ColumnTransformer that:
        - Imputes + standard-scales numeric features
        - Imputes + one-hot encodes categorical features
        """
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer([
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CATEGORICAL_COLS),
        ])

    # =========================
    # MODELS
    # =========================
    def build_models(self) -> dict[str, tuple[ImbPipeline, dict]]:
        """
        Define two classifier pipelines with SMOTE and their hyperparameter grids.

        SMOTE is placed after the preprocessor in each pipeline so it only
        sees scaled, imputed data — matching the distribution the model will
        encounter at inference time.

        Models
        ------
        logistic_regression : Fast, interpretable baseline. Uses l1_ratio
                              (ElasticNet mixing) instead of the deprecated
                              penalty parameter: l1_ratio=1 gives pure L1
                              (feature selection), l1_ratio=0 gives pure L2
                              (weight shrinkage).
        random_forest       : Ensemble of decision trees. Handles non-linear
                              interactions between sensor readings and is
                              robust to outliers. Generally stronger than
                              logistic regression on tabular sensor data.

        Both models use class_weight='balanced' as a second layer of imbalance
        correction on top of SMOTE.
        """
        smote = SMOTE(random_state=self.random_state)

        logistic = ImbPipeline([
            ("prep", self.build_preprocessor()),
            ("smote", smote),
            ("model", LogisticRegression(
                solver="saga",
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            )),
        ])

        forest = ImbPipeline([
            ("prep", self.build_preprocessor()),
            ("smote", SMOTE(random_state=self.random_state)),
            ("model", RandomForestClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state,
            )),
        ])

        return {
            "logistic_regression": (
                logistic,
                {
                    # l1_ratio=1 → pure L1 (Lasso), l1_ratio=0 → pure L2 (Ridge)
                    # Replaces deprecated 'penalty' parameter (removed in sklearn 1.10)
                    "model__C": [0.1, 1.0, 10.0],
                    "model__l1_ratio": [0.0, 1.0],
                },
            ),
            "random_forest": (
                forest,
                {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 12, 20],
                },
            ),
        }

    # =========================
    # TRAIN
    # =========================
    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> dict[str, GridSearchCV]:
        """
        Train all models using GridSearchCV with stratified k-fold CV.
        Optimises for weighted F1 to account for class imbalance.
        SMOTE oversampling is applied inside each CV fold to prevent leakage.
        """
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for name, (model, params) in self.build_models().items():
            LOGGER.info("Training %s ...", name)

            search = GridSearchCV(
                model,
                params,
                scoring="f1_weighted",
                cv=cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            self.models[name] = search

            LOGGER.info(
                "Best %s — params: %s | CV weighted-F1: %.4f",
                name,
                search.best_params_,
                search.best_score_,
            )

        return self.models

    # =========================
    # EVALUATE
    # =========================
    def evaluate(
        self,
        model: GridSearchCV,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """
        Compute evaluation metrics for a fitted model on held-out test data.

        Metrics
        -------
        accuracy          : Overall fraction of correct predictions.
        balanced_accuracy : Mean per-class recall; preferred when classes are
                            imbalanced as it is not skewed by the majority class.
        weighted_f1       : F1 averaged by class support; good overall summary.
        macro_f1          : Unweighted mean F1 across classes; penalises poor
                            performance on minority classes (High Activity).
        best_params       : Hyperparameters selected by GridSearchCV.
        best_cv_score     : Best weighted-F1 score seen during cross-validation.
        """
        pred = model.predict(X_test)

        return {
            "best_params": model.best_params_,
            "best_cv_score": model.best_score_,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "weighted_f1": f1_score(y_test, pred, average="weighted"),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "classification_report": classification_report(
                y_test,
                pred,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                pred,
                labels=CLASS_ORDER,
            ).tolist(),
        }

    # =========================
    # RUN FULL PIPELINE
    # =========================
    def run(self) -> dict[str, Any]:
        """Execute the full pipeline: load → clean → feature engineer → split → train → evaluate."""
        df = self.load_data()
        df = self.clean_data(df)
        df = self.add_features(df)

        X_train, X_test, y_train, y_test = self.split(df)

        self.train(X_train, y_train)

        self.results = {
            name: self.evaluate(model, X_test, y_test)
            for name, model in self.models.items()
        }

        return self.results





def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline configuration."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate gas activity classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path("data/gas_monitoring.db"),
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--table-name", default="gas_monitoring",
        help="Name of the table inside the database.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to use as the test set (0–1).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: configure logging, run pipeline, print JSON results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    pipeline = GasActivityPipeline(
        db_path=args.db_path,
        table_name=args.table_name,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )

    results = pipeline.run()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
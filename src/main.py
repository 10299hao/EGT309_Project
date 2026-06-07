"""
main.py
------------
End-to-end ML pipeline for classifying room activity level from gas
and environmental sensor readings.

Pipeline steps:
    1. Load data from SQLite database
    2. Clean data (fix labels, remove impossible values)
    3. Engineer new features
    4. Save EDA visualisations (target distribution, temperature boxplot)
    5. Split into train/test sets
    6. Train models (Random Forest, MLP Neural Network)
    7. Evaluate models and save visualisations (confusion matrix, F1 chart)
    8. Save model comparison chart and CSV
    9. Save trained models to saved_model/ folder

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
import joblib
import logging
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — saves plots without needing a display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mh_RF import train_random_forest
from mh_mlp import train_mlp



TARGET      = "Activity Level"
CLASS_ORDER = ["Low Activity", "Moderate Activity", "High Activity"]

PALETTE = {
    "Low Activity":      "#4C9BE8",
    "Moderate Activity": "#F5A623",
    "High Activity":     "#E85C5C",
}

# Maps all dirty label variants to 3 clean canonical classes
ACTIVITY_MAP: dict[str, str] = {
    "Low Activity":      "Low Activity",
    "Low_Activity":      "Low Activity",
    "LowActivity":       "Low Activity",
    "Moderate Activity": "Moderate Activity",
    "ModerateActivity":  "Moderate Activity",
    "High Activity":     "High Activity",
}

# All sensor + engineered features used as model inputs
NUMERIC_COLS: list[str] = [
    "Temperature", "Humidity",
    "CO2_InfraredSensor", "CO2_ElectroChemicalSensor",
    "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2",
    "MetalOxideSensor_Unit3", "MetalOxideSensor_Unit4",
    "CO_GasSensor",
    "CO2_Average",        # engineered: average of two CO2 sensors
    "TotalMOS",           # engineered: sum of all 4 MOS sensors
    "CO2_CO_Ratio",       # engineered: CO2 vs CO signal ratio
    "TimeOfDay_Ordinal",  # engineered: ordered time encoding
]

CATEGORICAL_COLS: list[str] = [
    "Time of Day", "HVAC Operation Mode", "Ambient Light Level",
]

SAVED_MODEL_DIR = Path("saved_model")
RESULTS_DIR     = Path("results")

LOGGER = logging.getLogger(__name__)


#visualisation

def save_target_distribution(df: pd.DataFrame):
    """
    Save a bar chart showing the count of each activity class.
    Used to visually confirm class imbalance identified in EDA.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
    pcts   = counts / counts.sum() * 100

    bars = ax.bar(
        counts.index, counts.values,
        color=[PALETTE[c] for c in counts.index],
        edgecolor="white", width=0.5,
    )
    # Label each bar with its percentage
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{pct:.1f}%",
            ha="center", fontsize=11, fontweight="bold",
        )

    ax.set_title("Distribution of Activity Levels", fontsize=13, fontweight="bold")
    ax.set_xlabel("Activity Level")
    ax.set_ylabel("Count")
    plt.tight_layout()

    save_path = RESULTS_DIR / "target_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def save_temperature_boxplot(df: pd.DataFrame):
    """
    Save a boxplot of Temperature by Activity Level after cleaning.
    Verifies that impossible values (>40 degrees C) have been removed.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x=TARGET, y="Temperature",
        order=CLASS_ORDER,
        palette=PALETTE, ax=ax,
    )
    ax.set_title("Temperature by Activity Level (Post-Cleaning)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Activity Level")
    ax.set_ylabel("Temperature (°C)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()

    save_path = RESULTS_DIR / "temperature_boxplot.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def save_confusion_matrix(y_test, y_pred, model_name: str):
    """
    Save a heatmap of the confusion matrix for a model.
    Shows how many predictions were correct vs which classes were confused.
    """
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(
        f"Confusion Matrix — {model_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(False)
    plt.tight_layout()

    save_path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def save_f1_bar_chart(y_test, y_pred, model_name: str):
    """
    Save a bar chart showing F1 score per activity class.
    Makes it easy to see if the model is ignoring the minority class (High Activity).
    """
    report = classification_report(
        y_test, y_pred, target_names=CLASS_ORDER, output_dict=True
    )
    f1_scores = {cls: report[cls]["f1-score"] for cls in CLASS_ORDER}

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        f1_scores.keys(), f1_scores.values(),
        color=[PALETTE[c] for c in f1_scores.keys()],
        edgecolor="white", width=0.5,
    )
    for bar, val in zip(bars, f1_scores.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="0.8 threshold")
    ax.set_title(
        f"F1 Score per Class — {model_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Activity Level")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=9)
    plt.tight_layout()

    save_path = RESULTS_DIR / f"{model_name}_f1_per_class.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def save_model_comparison(all_results: dict):
    """
    Save a grouped bar chart and CSV comparing all models side by side.
    Metrics shown: Weighted F1, Macro F1, Accuracy, Balanced Accuracy.
    """
    metrics       = ["weighted_f1", "macro_f1", "accuracy", "balanced_accuracy"]
    metric_labels = ["Weighted F1", "Macro F1", "Accuracy", "Balanced Accuracy"]
    model_names   = list(all_results.keys())
    colors        = ["#4C9BE8", "#E85C5C", "#5DBE8A"]

    x     = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(model_names):
        values = [all_results[model_name][m] for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=model_name.replace("_", " ").title(),
            color=colors[i % len(colors)],
            edgecolor="white",
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    chart_path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"  Saved: {chart_path}")

    # Also save as CSV for reference
    rows = [
        {
            "model":             name.replace("_", " ").title(),
            "weighted_f1":       round(all_results[name]["weighted_f1"], 4),
            "macro_f1":          round(all_results[name]["macro_f1"], 4),
            "accuracy":          round(all_results[name]["accuracy"], 4),
            "balanced_accuracy": round(all_results[name]["balanced_accuracy"], 4),
        }
        for name in model_names
    ]
    csv_path = RESULTS_DIR / "model_comparison.csv"
    pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def save_classification_report_txt(y_test, y_pred, model_name: str) -> None:
    """Save the full classification report as a readable .txt file."""
    report = classification_report(y_test, y_pred, target_names=CLASS_ORDER)
    save_path = RESULTS_DIR / f"{model_name}_classification_report.txt"
    with open(save_path, "w") as f:
        f.write(f"Classification Report — {model_name.replace('_', ' ').title()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"  Saved: {save_path}")



#for loading, cleaning, training, evaluation, visualisation and save models
class GasActivityPipeline:

    def __init__(
        self,
        db_path: Path,
        table_name: str   = "gas_monitoring",
        test_size: float  = 0.2,
        random_state: int = 42,
        cv_folds: int     = 5,
    ):
        self.db_path      = Path(db_path)
        self.table_name   = table_name
        self.test_size    = test_size
        self.random_state = random_state
        self.cv_folds     = cv_folds
        self.models: dict[str, Any] = {}
        self.results: dict[str, Any] = {}

    #load
    def load_data(self):
        """Load raw data from SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {self.table_name}", conn)

    #clean
    def clean_data(self, df: pd.DataFrame):
        """
        Fix dirty labels, remove physically impossible sensor readings,
        and normalise categorical text values.
        """
        df = df.copy()

        #standardis to 3 class
        df[TARGET] = df[TARGET].map(ACTIVITY_MAP)

        df.loc[df["Temperature"] > 40, "Temperature"] = np.nan
        df.loc[(df["Humidity"] < 0) | (df["Humidity"] > 100), "Humidity"] = np.nan

        for col in CATEGORICAL_COLS:
            df[col] = self._normalize_text(df[col])

        return df

    @staticmethod
    def _normalize_text(series: pd.Series):
        cleaned = series.copy()
        mask    = cleaned.notna()
        cleaned.loc[mask] = (
            cleaned.loc[mask]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
        )
        return cleaned.astype(object)

    
    #feature engineering
    def add_features(self, df: pd.DataFrame):
        """
        Create new features identified as useful during EDA.

        Features added:
            CO2_Average      : Averages two correlated CO2 sensors to reduce noise
            TotalMOS         : Sums all 4 MOS sensors into one combined VOC signal
            CO2_CO_Ratio     : Ratio of CO2 to CO — respiration vs combustion signal
            TimeOfDay_Ordinal: Ordered encoding (night=0, morning=1, afternoon=2, evening=3)
        """
        df = df.copy()

        df["CO2_Average"] = (
            df["CO2_InfraredSensor"] + df["CO2_ElectroChemicalSensor"]
        ) / 2

        df["TotalMOS"] = (
            df["MetalOxideSensor_Unit1"] + df["MetalOxideSensor_Unit2"] +
            df["MetalOxideSensor_Unit3"] + df["MetalOxideSensor_Unit4"]
        )

        # +1 prevents division by zero when CO_GasSensor = 0
        df["CO2_CO_Ratio"] = df["CO2_Average"] / (df["CO_GasSensor"] + 1)

        time_map = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}
        df["TimeOfDay_Ordinal"] = df["Time of Day"].map(time_map)

        return df

    
    #split into train and test sets: using stratified split preserve the class proportion
    def split(self, df: pd.DataFrame):
        y = df[TARGET]
        X = df.drop(columns=[TARGET, "Session ID"], errors="ignore")
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    #preprocessing pipeline applied to features b4 training
    def build_preprocessor(self):
        """
        Numeric : Median imputation → StandardScaler
        Categorical : Mode imputation → OneHotEncoder
        """
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer([
            ("num", numeric_pipeline,     NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ])

    #train
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all models using the shared preprocessor."""
        preprocessor = self.build_preprocessor()
        self.models["random_forest"]  = train_random_forest(preprocessor, X_train, y_train, self.cv_folds, self.random_state)
        self.models["mlp_neural_net"] = train_mlp(preprocessor, X_train, y_train, self.cv_folds, self.random_state)
        return self.models

    
    #evalutation and save models/visualisation
    def evaluate_all(self, X_test: pd.DataFrame, y_test: pd.Series):
 
        all_results = {}

        for model_name, model in self.models.items():
            print(f"\n  Evaluating: {model_name.replace('_', ' ').title()}")
            y_pred = model.predict(X_test)

            # Save visualisations for this model
            save_confusion_matrix(y_test, y_pred, model_name)
            save_f1_bar_chart(y_test, y_pred, model_name)
            save_classification_report_txt(y_test, y_pred, model_name)

            # Collect metrics
            all_results[model_name] = {
                "best_params":           model.best_params_,
                "best_cv_score":         round(model.best_score_, 4),
                "accuracy":              round(accuracy_score(y_test, y_pred), 4),
                "balanced_accuracy":     round(balanced_accuracy_score(y_test, y_pred), 4),
                "weighted_f1":           round(f1_score(y_test, y_pred, average="weighted"), 4),
                "macro_f1":              round(f1_score(y_test, y_pred, average="macro"), 4),
                "confusion_matrix":      confusion_matrix(y_test, y_pred, labels=CLASS_ORDER).tolist(),
                "classification_report": classification_report(y_test, y_pred, target_names=CLASS_ORDER),
            }

        # Save comparison chart and CSV across all models
        save_model_comparison(all_results)

        return all_results

    
    #save models
    def save_models(self):
        """
        Save all trained models to saved_model/ folder using joblib.
        Creates the folder automatically if it does not exist.
        """
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for model_name, model in self.models.items():
            save_path = SAVED_MODEL_DIR / f"{model_name}.pkl"
            joblib.dump(model, save_path)
            print(f"  Saved: {save_path}")


    def run(self) -> dict[str, Any]:

        # Create output folders
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print("Step 1/7  Loading data from database...")
        df_raw = self.load_data()
        print(f"Loading {len(df_raw):,} rows x {df_raw.shape[1]} columns")

        print("Step 2/7  Cleaning data...")
        df = self.clean_data(df_raw)

        print("Step 3/7  Feature engineering")
        df = self.add_features(df)
        print(f"       Dataset now has {df.shape[1]} columns after feature engineering")

        print("Step 4/7  Saving data visualisations")
        save_target_distribution(df)
        save_temperature_boxplot(df)

        print("Step 5/7  Splitting into train/test set")
        X_train, X_test, y_train, y_test = self.split(df)
        print(f"          Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

        print("Step 6/7  Training models (pls be patient)")
        self.train(X_train, y_train)

        print("Step 7/7  Evaluating models and saving results")
        self.results = self.evaluate_all(X_test, y_test)

        print("\n         Saving trained models")
        self.save_models()

        print(" FINALLY!")
        print(f" Models  → {SAVED_MODEL_DIR}/")
        print(f" Results → {RESULTS_DIR}/")

        return self.results



def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Command line arguments — allows hyperparameters to be changed without editing code
    parser = argparse.ArgumentParser(description="EGT309 Gas Activity ML Pipeline")
    parser.add_argument("--db-path",      type=Path,  default=Path("data/gas_monitoring.db"), help="Path to SQLite database")
    parser.add_argument("--table-name",   type=str,   default="gas_monitoring",               help="Table name in database")
    parser.add_argument("--test-size",    type=float, default=0.2,                            help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--random-state", type=int,   default=42,                             help="Random seed for reproducibility")
    parser.add_argument("--cv-folds",     type=int,   default=5,                              help="Number of cross-validation folds")
    args = parser.parse_args()

    pipeline = GasActivityPipeline(
        db_path=args.db_path,
        table_name=args.table_name,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )

    results = pipeline.run()

    # Print final results summary to terminal
    print("\n========================================")
    print(" Results Summary")
    print("========================================")
    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').upper()}")
        print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
        print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
        print(f"  Accuracy       : {metrics['accuracy']:.4f}")
        print(f"  Balanced Acc   : {metrics['balanced_accuracy']:.4f}")
        print(f"\n  Classification Report:\n{metrics['classification_report']}")


if __name__ == "__main__":
    main()
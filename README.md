# EGT309 — Gas Monitoring Activity Level Classification

## Group Information

**Group Name:** Team 3  
**Members:**
| Name | Files Written |
|---|---|
| MH | `main.py`, `mh_mlp.py`, `mh_RF.py`|
| Gina | `gina_1.py`, `gina_2.py`  |
| Eishmeet | `eishmeet_1.py`,`eishmeet_2.py` |


---

## Project Structure

```
AI_SOL_P/
├── data/
│   └── gas_monitoring.db        # SQLite database
├── src/
│   ├── main.py           # Main pipeline
│   ├── mh_RF.py                 # Random Forest model 
│   ├── mh_mlp.py                # MLP Neural Network model 
│   ├── gina_1.py                # Gina's pipeline files
│   └── gina_2.py                
|   ├── eishmeet_1.py            # eishmeet pipeline files
|   ├── eishmeet_2.py            # Eishmeet's pipeline files
├── saved_model/                 # Trained models saved here after running
├── eda.ipynb                    # Exploratory Data Analysis notebook
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run.sh                       # Script to run the full pipeline
└── Dockerfile                   # Docker container configuration
```

---

## How to Run the Pipeline

### Option 1 — Run with Docker (Recommended)

Make sure Docker Desktop is installed and running, then:

```bash
# Step 1 — Build the Docker image
docker build -t team3-pipeline .

# Step 2 — Run the pipeline
docker run team3-pipeline

```

### Option 2 — Run directly with Python

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt


# Or run individual scripts
python src/mh_pipeline.py
```

---

## How to Start the Docker Development Environment

```bash
# Build the image
docker build -t team3-pipeline .

# Inside the container you can run any script
python src/main.py
```

### Option 3 — Docker Compose (Shortest)
```bash
docker compose up
```

---
---

## Summary of Key EDA Findings

The EDA was conducted on 10,000 rows and 14 columns from the `gas_monitoring.db` dataset. The goal was to predict `Activity Level` (Low, Moderate, or High) from indoor gas and environment sensor readings.

### Data Quality Issues Found and Resolved

| Issue | What Was Found | How It Was Handled |
|---|---|---|
| Dirty Activity Level labels | 6 raw variants for 3 real classes | Mapped all variants to 3 canonical labels |
| Dirty HVAC Operation Mode labels | 23 raw variants for 6 real modes | Lowercased, stripped spaces, replaced underscores |
| Mixed temperature units | 795 rows recorded in Kelvin (289–307 K range) | Converted to Celsius by subtracting 273.15; any remaining out-of-range values set to NaN then median imputed |
| Impossible humidity readings | 414 rows outside 0–100% | Set to NaN, then median imputed |
| Missing numerical values | 4 columns, 8.3%–19.3% missing | Median imputation (fit on training data only) |
| Missing categorical values | Ambient Light Level 10.5% missing | Mode imputation (fit on training data only) |
| Class imbalance | High Activity only 10.9% of data | Stratified split, weighted F1 metric, class weights |
| Session ID column | Identifier, not a sensor feature | Dropped before training |

### Strongest Predictors Found

1. `MetalOxideSensor_Unit2` and `MetalOxideSensor_Unit4` — clearest median shift across activity classes
2. `CO2_ElectroChemicalSensor` — rises with activity due to increased breathing
3. `CO_GasSensor` — inverse signal, decreases as activity increases
4. `TotalMOS` (engineered) — combined VOC signal, strongest engineered feature

### Weak Predictors

- `Temperature` and `Humidity` — medians barely shift across activity classes, low predictive value

---

## Feature Engineering

Four new features were created and validated against the target before inclusion:

| Feature | Formula | Justification |
|---|---|---|
| `TotalMOS` | Sum of MetalOxideSensor Unit1–4 | MOS sensors are top predictors; combining them captures total VOC/gas load |
| `CO2_Average` | Mean of CO2_InfraredSensor and CO2_ElectroChemicalSensor | The two sensors are strongly correlated (r=−0.32 cross-sensor); averaging reduces noise |
| `CO2_CO_Ratio` | CO2_Average / (CO_GasSensor + 1) | Captures ratio of respiration signal (CO2) to combustion signal (CO); +1 avoids division by zero |
| `TimeOfDay_Ordinal` | night=0, morning=1, afternoon=2, evening=3 | Ordered encoding for models that benefit from feature magnitude; used alongside one-hot encoding |

All engineered features were validated using box plots and effect size (eta-squared) against `Activity Level` before being included in the pipeline.

---

## Model Choices and Justification

The EDA showed significant overlap between Low and Moderate Activity classes, indicating a non-linear decision boundary. Three models were trained and compared:

### 1. k-Nearest Neighbors (k-NN)
- **Why:** k-NN is an instance-based, non-parametric algorithm that computes geometric distances between data points. Because it makes no structural assumptions about data distributions, it is especially useful for capturing irregular, non-linear decision boundaries in regions where classes greatly overlap.
- **Limitation:** k-NN is sensitive to irrelevant features, extreme noise, and scaling variations. It struggles with severe class imbalance, favoring majority classes unless significantly adjusted.
- **Tuning:** `n_neighbours`: Adjusting the number of neighbors (k=5, 7, 11) to balance the bias-variance trade-off. `weights`: Switching from 'uniform' to 'distance' ensures that closer, highly relevant neighbors have a greater statistical pull on the target vote

### 2. Random Forest
- **Why:** Handles non-linear patterns naturally; provides feature importance scores to validate EDA findings; robust to multicollinearity between MOS sensors
- **Tuning:** `n_estimators`, `max_depth`, `min_samples_split` tuned via GridSearchCV; `class_weight='balanced'` to handle imbalance

### 3. 
- **Why:** 
- **Tuning:**

---

## Evaluation Metrics and Justification

Standard accuracy is misleading here because predicting `Low Activity` for every row gives 57.7% accuracy without learning anything useful.

| Metric | Why It Was Chosen |
|---|---|
| **Weighted F1-score** | Primary metric — weights each class by its support, accounting for imbalance |
| **Macro F1-score** | Secondary metric — treats all classes equally, ensures High Activity is not ignored |
| **High Activity Recall** | Ensures the rare but important minority class is actually being detected |
| **Confusion Matrix** | Shows exactly which classes the model confuses, not just overall performance |

---

## Requirements

```
pandas>=2.2
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
imbalanced-learn>=0.12
xgboost>=2.0
joblib>=1.3
jupyter>=1.0
ipykernel>=6.29
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Version Control

All code changes are tracked on GitHub with descriptive commit messages.  
Each member worked on a separate branch and changes were merged into `main`.



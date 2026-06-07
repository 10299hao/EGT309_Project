#random forest
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import logging

LOGGER = logging.getLogger(__name__)

def build_forest_pipeline(preprocessor, random_state: int = 42):
    return Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            class_weight="balanced_subsample", 
            n_jobs=-1, 
            random_state=random_state
        )),
    ])

def get_forest_param_dist():
    """Hyperparameter distribution for RandomizedSearchCV."""
    return {
        "model__n_estimators": [300, 500],
        "model__max_depth": [20, 30, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", "log2"]
    }

def train_random_forest(
    preprocessor,
    X_train,
    y_train,
    cv_folds: int = 5,
    random_state: int = 42,
):
    
    pipeline = build_forest_pipeline(preprocessor, random_state)
    param_dist = get_forest_param_dist()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10, 
        scoring="f1_weighted", 
        cv=cv,
        n_jobs=-1, 
        random_state=random_state
    )

    LOGGER.info("Training Random Forest...")
    search.fit(X_train, y_train)
    LOGGER.info("Best RF params: %s | Score: %.4f", search.best_params_, search.best_score_)
    
    return search
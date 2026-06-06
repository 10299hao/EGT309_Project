"""
model_mlp.py
-------------
Defines the Multi-Layer Perceptron (MLP) neural network pipeline.
Uses RandomOverSampler to prevent interpolation noise from SMOTE.
"""
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import logging

LOGGER = logging.getLogger(__name__)

def build_mlp_pipeline(preprocessor, random_state: int = 42) -> ImbPipeline:
    return ImbPipeline([
        ("prep", preprocessor),
        ("ros", RandomOverSampler(random_state=random_state)),
        ("model", MLPClassifier(
            max_iter=800, 
            early_stopping=False,
            random_state=random_state
        )),
    ])

def get_mlp_param_dist() -> dict:
    return {
        "model__hidden_layer_sizes": [(128, 64), (256, 128, 64)],
        "model__activation": ["relu"],
        "model__alpha": [0.001, 0.01],
        "model__learning_rate_init": [0.001, 0.005],
    }

def train_mlp(
    preprocessor,
    X_train,
    y_train,
    cv_folds: int = 5,
    random_state: int = 42,
) -> RandomizedSearchCV:
    
    pipeline = build_mlp_pipeline(preprocessor, random_state)
    param_dist = get_mlp_param_dist()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=6,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        random_state=random_state
    )

    LOGGER.info("Training Multi-Layer Perceptron...")
    search.fit(X_train, y_train)
    LOGGER.info("Best MLP params: %s | Score: %.4f", search.best_params_, search.best_score_)
    
    return search
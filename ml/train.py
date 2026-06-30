"""
train.py
--------
Model training pipeline for the Loan Eligibility Prediction project.

This script:
    1. Loads the preprocessed feature matrix (X.pkl) and target vector
       (y.pkl) - built by the current preprocess.py from loan_dataset.csv
       (20 original features plus engineered ratios).
    2. Splits the data into train and test sets (80/20, stratified).
    3. Trains and compares FOUR algorithms:
        - Decision Tree Classifier
        - Random Forest Classifier
        - K-Nearest Neighbors (KNN), with K tuned via GridSearchCV, run
          inside a StandardScaler Pipeline (this dataset's features span
          very different numeric scales, so KNN's distance metric needs
          scaled inputs to be meaningful)
        - XGBoost Classifier - this is the model this update focuses on
          improving, via:
            a) RandomizedSearchCV over a wide hyperparameter
               distribution, scored with explicit Stratified K-Fold CV
            b) a second, separate retraining pass using the best
               hyperparameters found, this time WITH early stopping
               against a genuine held-out validation split (carved out
               of the training data only, never the test set)
    4. Evaluates every model: Accuracy, Precision, Recall, F1, ROC-AUC,
       and Confusion Matrix, plus Stratified K-Fold Cross Validation
       Accuracy.
    5. Prints a clean side-by-side comparison table of all four models.
    6. Automatically selects the best model (highest Test Accuracy,
       breaking ties with the higher F1 Score).
    7. Saves ONLY the best model to ml/model/loan_model.pkl.
    8. Prints a final summary block for the winning model.

On early stopping and avoiding leakage
---------------------------------------
XGBoost's sklearn-API early stopping needs a fixed eval_set at fit time,
which does not compose cleanly with RandomizedSearchCV/GridSearchCV's
automatic cross-validation folds (each fold would need its own internal
validation split, which most of sklearn's search tooling does not do for
you, and improvising it incorrectly is a common source of leakage). To
stay correct, this script keeps those two techniques in two clean,
separate stages instead of mixing them in one step:
    Stage 1 - RandomizedSearchCV (with StratifiedKFold, no early
              stopping) searches hyperparameters using only X_train/y_train.
    Stage 2 - The best hyperparameters from Stage 1 are used to retrain a
              fresh XGBoost model, this time with early stopping against
              a validation split carved out of X_train/y_train (NOT
              X_test/y_test). The test set is only ever touched once, at
              final evaluation time.
This keeps the reported test accuracy/ROC-AUC honest - the test set is
never used to pick hyperparameters or to decide when to stop boosting.

On achievable accuracy
------------------------
This script aims for the highest realistic accuracy the data supports,
but does not artificially inflate it. The dataset's approval logic
includes deliberate, controlled randomness (per the project's data
generation design), so a perfect or near-perfect score is not expected
and would actually be a red flag for leakage rather than a good sign.

Run directly:
    python ml/train.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Configuration / Paths
# --------------------------------------------------------------------------

MODEL_DIR = os.path.join("ml", "model")

X_PICKLE_PATH = os.path.join(MODEL_DIR, "X.pkl")
Y_PICKLE_PATH = os.path.join(MODEL_DIR, "y.pkl")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "loan_model.pkl")

# Reproducibility
RANDOM_STATE = 42

# Train/test split configuration
TEST_SIZE = 0.2

# Cross-validation configuration. CV_SPLITTER is an explicit
# StratifiedKFold object (rather than a plain integer) so every model's
# cross-validation - and RandomizedSearchCV/GridSearchCV's internal
# search - uses the SAME folds, with class balance preserved in every
# fold. This matters more on this dataset than it would on a perfectly
# balanced one, since stratification keeps the approve/reject ratio
# consistent across folds instead of letting it drift by chance.
CV_FOLDS = 5
CV_SPLITTER = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Metric used to select the best model during hyperparameter search
SEARCH_SCORING = "accuracy"

# --------------------------------------------------------------------------
# XGBoost hyperparameter search space (RandomizedSearchCV)
# --------------------------------------------------------------------------
# RandomizedSearchCV samples N_SEARCH_ITER random combinations from this
# distribution rather than exhaustively trying every combination like
# GridSearchCV. This lets the search cover a WIDER, finer-grained space
# (more values per parameter) in a comparable or smaller time budget than
# the previous grid search used, which is what gives this version a
# realistic shot at finding a better-performing model.
XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 150, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 5, 7],
    "gamma": [0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1],
    "reg_lambda": [0.5, 1, 1.5, 2, 3],
}

# Number of random hyperparameter combinations RandomizedSearchCV will
# try. Each one is evaluated across all CV_FOLDS folds, so the total
# number of model fits during search is N_SEARCH_ITER * CV_FOLDS.
N_SEARCH_ITER = 60

# Upper bound on boosting rounds for the early-stopping retrain stage
# (Stage 2). Early stopping decides the actual number of rounds used;
# this is just a high ceiling it won't usually reach.
EARLY_STOPPING_MAX_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 30

# How much of the training data to carve out as a held-out validation
# split for early stopping (Stage 2 only). This split never overlaps
# with X_test/y_test.
EARLY_STOPPING_VALIDATION_SIZE = 0.15

# Reasonable, beginner-friendly configuration for Random Forest.
# n_estimators=200 gives a stable forest without being slow to train;
# max_depth=10 helps prevent overfitting on a small/medium dataset.
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Hyperparameter grid for tuning K-Nearest Neighbors via GridSearchCV.
# n_neighbors covers a practical range of K values; weights and p (distance
# metric: 1=Manhattan, 2=Euclidean) are tuned alongside K for a fair search.
# Keys are prefixed with "knn__" because KNN is wrapped in a scaling
# Pipeline (see tune_knn_model) - this dataset's features span very
# different scales (e.g. age 21-65 vs. collateral_value up to 15,000,000),
# and KNN's distance metric would otherwise be dominated entirely by the
# largest-magnitude columns.
KNN_PARAM_GRID = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}

# Display names used for the comparison table and final summary, in the
# fixed order requested: Decision Tree, Random Forest, KNN, XGBoost.
MODEL_DISPLAY_ORDER = ["Decision Tree", "Random Forest", "KNN", "XGBoost"]


# --------------------------------------------------------------------------
# Step 1: Load Features and Target
# --------------------------------------------------------------------------
def load_data(x_path: str, y_path: str):
    """
    Load the feature matrix (X) and target vector (y) from pickle files.

    Args:
        x_path (str): Path to the X.pkl file.
        y_path (str): Path to the y.pkl file.

    Returns:
        tuple: (X, y) as pandas DataFrame/Series.

    Raises:
        FileNotFoundError: If either pickle file does not exist.
        Exception: For any other error encountered while unpickling.
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Feature file not found at: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Target file not found at: {y_path}")

    try:
        with open(x_path, "rb") as f:
            X = pickle.load(f)
        with open(y_path, "rb") as f:
            y = pickle.load(f)
    except Exception as e:
        raise Exception(f"Failed to load pickle files: {e}")

    return X, y


# --------------------------------------------------------------------------
# Step 2: Split Data into Train/Test
# --------------------------------------------------------------------------
def split_data(X, y, test_size: float, random_state: int):
    """
    Split the dataset into training and testing sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size (float): Proportion of data reserved for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # preserve class balance between train and test sets
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# Step 3a: Train the Decision Tree Classifier
# --------------------------------------------------------------------------
def train_decision_tree(X_train, y_train, random_state: int) -> DecisionTreeClassifier:
    """
    Train a Decision Tree Classifier on the training data.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        random_state (int): Seed for reproducibility.

    Returns:
        DecisionTreeClassifier: The trained model.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------
# Step 3b: Train the Random Forest Classifier
# --------------------------------------------------------------------------
def train_random_forest(X_train, y_train, params: dict) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the training data using a
    reasonable, beginner-friendly configuration (see RANDOM_FOREST_PARAMS).

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        params (dict): Hyperparameters for RandomForestClassifier.

    Returns:
        RandomForestClassifier: The trained model.
    """
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------
# Step 3c: Tune and Train K-Nearest Neighbors via GridSearchCV
# --------------------------------------------------------------------------
def tune_knn_model(X_train, y_train, param_grid: dict, cv, scoring: str) -> GridSearchCV:
    """
    Automatically find the best K (and related hyperparameters) for a
    K-Nearest Neighbors classifier using GridSearchCV, then retrain the
    best estimator on the full training set.

    KNN is wrapped in a Pipeline with a StandardScaler step. This
    dataset's features span very different numeric scales (e.g. age is
    21-65, while collateral_value can be up to 15,000,000) - without
    scaling, KNN's distance metric would be almost entirely dominated by
    the largest-magnitude columns, effectively ignoring most features.
    Wrapping the scaler and classifier together in one Pipeline also means
    the saved model scales its own input automatically, so nothing else
    in the project (e.g. Flask) needs to know or do this manually.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        param_grid (dict): Hyperparameter grid to search over (keys
            prefixed with "knn__" to target the Pipeline's KNN step).
        cv: A cross-validation splitter (e.g. a StratifiedKFold instance).
        scoring (str): Scoring metric used to select the best parameters.

    Returns:
        GridSearchCV: The fitted GridSearchCV object (best_estimator_ is
            a fitted Pipeline of [scaler, knn], already retrained on the
            full training data, since refit=True).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    return grid_search


# --------------------------------------------------------------------------
# Step 3d-i: XGBoost Stage 1 - RandomizedSearchCV hyperparameter search
# --------------------------------------------------------------------------
def search_xgboost_hyperparameters(
    X_train, y_train, param_distributions: dict, n_iter: int, cv, scoring: str, random_state: int
) -> RandomizedSearchCV:
    """
    Stage 1 of XGBoost tuning: search hyperparameters using
    RandomizedSearchCV with explicit Stratified K-Fold cross validation.

    RandomizedSearchCV samples n_iter random combinations from
    param_distributions rather than exhaustively trying all of them
    (which is what GridSearchCV would do). This lets the search cover a
    wider, more fine-grained space of values per hyperparameter in a
    comparable time budget, which is the main lever used here to look
    for a higher-accuracy XGBoost configuration than a smaller grid
    search would find.

    No early stopping is used in this stage on purpose (see the
    "On early stopping and avoiding leakage" note at the top of this
    file) - early stopping is applied separately in Stage 2, with the
    best hyperparameters found here.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        param_distributions (dict): Hyperparameter distributions to
            sample from.
        n_iter (int): Number of random parameter combinations to try.
        cv: A cross-validation splitter (e.g. a StratifiedKFold instance).
        scoring (str): Scoring metric used to select the best parameters.
        random_state (int): Seed for reproducibility of the random search
            itself (which combinations get sampled).

    Returns:
        RandomizedSearchCV: The fitted search object. best_params_ is
            used as the input to Stage 2 (search_xgboost_hyperparameters
            does not itself apply early stopping).
    """
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        random_state=random_state,
        verbose=1,
    )

    search.fit(X_train, y_train)
    return search


# --------------------------------------------------------------------------
# Step 3d-ii: XGBoost Stage 2 - retrain best params WITH early stopping
# --------------------------------------------------------------------------
def train_xgboost_with_early_stopping(
    X_train, y_train, best_params: dict, validation_size: float,
    max_estimators: int, early_stopping_rounds: int, random_state: int
) -> XGBClassifier:
    """
    Stage 2 of XGBoost tuning: retrain a fresh XGBoost model using the
    best hyperparameters found in Stage 1, this time WITH early stopping.

    A genuine validation split is carved out of X_train/y_train (NEVER
    from the test set) specifically for early stopping to monitor. This
    keeps the test set completely unseen until final evaluation, so the
    test accuracy/ROC-AUC reported later are not optimistically biased by
    having influenced when boosting stopped.

    n_estimators from best_params is replaced with max_estimators (a high
    ceiling) here, since the whole point of early stopping is to let the
    model find its own optimal round count rather than use a fixed one
    found by the Stage 1 search (which did not have early stopping).

    Args:
        X_train: Full training feature matrix (Stage 1's training data).
        y_train: Full training target vector.
        best_params (dict): Best hyperparameters from
            search_xgboost_hyperparameters's best_params_ (n_estimators
            is overridden; every other tuned parameter is reused as-is).
        validation_size (float): Fraction of X_train/y_train held out as
            the early-stopping validation split.
        max_estimators (int): Upper bound on boosting rounds.
        early_stopping_rounds (int): Number of rounds without
            improvement on the validation set before stopping early.
        random_state (int): Seed for reproducibility.

    Returns:
        XGBClassifier: The final, early-stopped XGBoost model, fitted on
            the inner training split (NOT yet refit on 100% of X_train -
            see the note in main() about why that tradeoff is made).
    """
    # Carve out a validation split from the TRAINING data only. stratify
    # keeps the approve/reject balance consistent in both pieces.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    params = dict(best_params)
    params["n_estimators"] = max_estimators

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )

    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


# --------------------------------------------------------------------------
# Step 4: Evaluate the Model
# --------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the trained model on the test set.

    Computes Accuracy, Precision, Recall, F1 Score, ROC-AUC, and
    Confusion Matrix.

    ROC-AUC is computed from predicted probabilities (via predict_proba)
    rather than hard class predictions, since AUC measures how well the
    model ranks/separates the two classes across all possible decision
    thresholds, not just at the default 0.5 cutoff. If a model does not
    support predict_proba for some reason, ROC-AUC is reported as None
    rather than crashing the whole evaluation.

    Args:
        model: Trained classifier (any sklearn-API compatible model).
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        dict: Dictionary containing all computed evaluation metrics.
    """
    y_pred = model.predict(X_test)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        # weighted average handles class imbalance gracefully
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


# --------------------------------------------------------------------------
# Step 5: Perform Stratified K-Fold Cross Validation
# --------------------------------------------------------------------------
def perform_cross_validation(model, X, y, cv):
    """
    Perform Stratified K-Fold Cross Validation on the full dataset using
    accuracy as the scoring metric.

    Args:
        model: Unfitted estimator (a fresh clone is used internally by
            cross_val_score for each fold).
        X: Full feature matrix.
        y: Full target vector.
        cv: A cross-validation splitter (e.g. a StratifiedKFold instance).

    Returns:
        numpy.ndarray: Array of accuracy scores, one per fold.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores


# --------------------------------------------------------------------------
# Step 6: Save the Trained Model
# --------------------------------------------------------------------------
def save_model(model, path: str):
    """
    Save the trained model to disk using pickle.

    Args:
        model: Trained model to persist.
        path (str): Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)


# --------------------------------------------------------------------------
# Step 7: Print Evaluation Results
# --------------------------------------------------------------------------
def print_search_results(search, label: str):
    """
    Print the best hyperparameters and best cross-validation score found
    by a GridSearchCV or RandomizedSearchCV object.

    Args:
        search: The fitted search object (GridSearchCV or RandomizedSearchCV).
        label (str): A short label identifying which search this is, for
            clearer console output (e.g. "XGBoost RandomizedSearchCV").
    """
    print(f"\n--- {label} Results ---")
    print(f"Best Parameters: {search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {search.best_score_:.4f}")


def print_metrics(metrics: dict):
    """
    Print evaluation metrics in a clean, readable format.

    Args:
        metrics (dict): Dictionary of evaluation metrics from evaluate_model().
    """
    print("\n--- Test Set Evaluation Metrics ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    else:
        print("ROC-AUC:   N/A (model has no predict_proba)")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


def print_cross_validation_results(scores, cv_folds: int):
    """
    Print cross-validation scores in a clean, readable format.

    Args:
        scores (numpy.ndarray): Array of accuracy scores per fold.
        cv_folds (int): Number of folds (for the header label only).
    """
    print(f"\n--- {cv_folds}-Fold Stratified Cross Validation Results ---")
    for fold_index, score in enumerate(scores, start=1):
        print(f"Fold {fold_index}: {score:.4f}")
    print(f"\nMean CV Accuracy: {scores.mean():.4f}")
    print(f"Std Dev CV Accuracy: {scores.std():.4f}")


# --------------------------------------------------------------------------
# Step 8: Compare All Models
# --------------------------------------------------------------------------
def print_comparison_table(results: dict):
    """
    Print a clean side-by-side comparison table of all trained models.

    Args:
        results (dict): Maps model display name -> dict of metrics (must
            contain accuracy, precision, recall, f1_score, roc_auc,
            cv_accuracy).
    """
    header = (
        f"{'Model':<16}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}"
        f"{'F1':>8}{'ROC-AUC':>10}{'CV Accuracy':>13}"
    )
    divider = "-" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    for name in MODEL_DISPLAY_ORDER:
        metrics = results[name]
        roc_auc_str = f"{metrics['roc_auc'] * 100:>9.2f}%" if metrics.get("roc_auc") is not None else f"{'N/A':>10}"
        print(
            f"{name:<16}"
            f"{metrics['accuracy'] * 100:>9.2f}%"
            f"{metrics['precision'] * 100:>10.2f}%"
            f"{metrics['recall'] * 100:>8.2f}%"
            f"{metrics['f1_score'] * 100:>7.2f}%"
            f"{roc_auc_str}"
            f"{metrics['cv_accuracy'] * 100:>12.2f}%"
        )

    print(divider)


def select_best_model(results: dict) -> str:
    """
    Identify the best-performing model using the highest Test Accuracy.
    If two or more models tie on accuracy, the one with the higher F1
    Score is chosen.

    Args:
        results (dict): Maps model display name -> dict of metrics
            (must contain accuracy and f1_score).

    Returns:
        str: The display name of the best model.
    """
    best_name = None
    best_accuracy = -1.0
    best_f1 = -1.0

    for name, metrics in results.items():
        accuracy = metrics["accuracy"]
        f1 = metrics["f1_score"]

        is_better = (
            accuracy > best_accuracy
            or (accuracy == best_accuracy and f1 > best_f1)
        )

        if is_better:
            best_name = name
            best_accuracy = accuracy
            best_f1 = f1

    return best_name


def print_final_summary(best_name: str, metrics: dict):
    """
    Print the final summary block for the winning model.

    Args:
        best_name (str): Display name of the best-performing model.
        metrics (dict): Metrics dictionary for the best model (must
            contain accuracy, precision, recall, f1_score, roc_auc,
            cv_accuracy).
    """
    print("\n==============================")
    print("SMART LENDER MODEL COMPARISON")
    print("==============================")
    print(f"Best Model : {best_name}")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision : {metrics['precision'] * 100:.2f}%")
    print(f"Recall : {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score : {metrics['f1_score'] * 100:.2f}%")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC : {metrics['roc_auc'] * 100:.2f}%")
    print(f"CV Accuracy : {metrics['cv_accuracy'] * 100:.2f}%")
    print("Model saved successfully.")


# --------------------------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------------------------
def main():
    """
    Execute the full model training and comparison pipeline end-to-end:
    train Decision Tree, Random Forest, KNN, and a heavily-tuned XGBoost,
    evaluate all four, print a comparison table, then save only the best
    model.
    """
    try:
        # 1. Load preprocessed features and target
        X, y = load_data(X_PICKLE_PATH, Y_PICKLE_PATH)
        print(f"Loaded data successfully. X shape: {X.shape}, y shape: {y.shape}")

        # 2. Split into train and test sets (80/20, stratified)
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")

        # This dictionary collects everything needed to compare models and
        # to save the winner: the fitted model object plus its metrics.
        results = {}
        trained_models = {}

        # ------------------------------------------------------------------
        # 3a. Decision Tree Classifier
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Training: Decision Tree Classifier")
        print("=" * 60)
        dt_model = train_decision_tree(X_train, y_train, random_state=RANDOM_STATE)
        dt_metrics = evaluate_model(dt_model, X_test, y_test)
        print_metrics(dt_metrics)
        dt_cv_scores = perform_cross_validation(
            DecisionTreeClassifier(random_state=RANDOM_STATE), X, y, cv=CV_SPLITTER
        )
        print_cross_validation_results(dt_cv_scores, CV_FOLDS)
        dt_metrics["cv_accuracy"] = dt_cv_scores.mean()
        results["Decision Tree"] = dt_metrics
        trained_models["Decision Tree"] = dt_model

        # ------------------------------------------------------------------
        # 3b. Random Forest Classifier
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Training: Random Forest Classifier")
        print("=" * 60)
        rf_model = train_random_forest(X_train, y_train, params=RANDOM_FOREST_PARAMS)
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        print_metrics(rf_metrics)
        rf_cv_scores = perform_cross_validation(
            RandomForestClassifier(**RANDOM_FOREST_PARAMS), X, y, cv=CV_SPLITTER
        )
        print_cross_validation_results(rf_cv_scores, CV_FOLDS)
        rf_metrics["cv_accuracy"] = rf_cv_scores.mean()
        results["Random Forest"] = rf_metrics
        trained_models["Random Forest"] = rf_model

        # ------------------------------------------------------------------
        # 3c. K-Nearest Neighbors (K auto-tuned via GridSearchCV)
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Training: K-Nearest Neighbors (tuning K via GridSearchCV)")
        print("=" * 60)
        knn_grid_search = tune_knn_model(
            X_train,
            y_train,
            param_grid=KNN_PARAM_GRID,
            cv=CV_SPLITTER,
            scoring=SEARCH_SCORING,
        )
        print_search_results(knn_grid_search, "KNN GridSearchCV")
        knn_model = knn_grid_search.best_estimator_
        knn_metrics = evaluate_model(knn_model, X_test, y_test)
        print_metrics(knn_metrics)
        # Re-run an independent Stratified CV with the best-found K for a
        # CV accuracy figure consistent with the other three models.
        knn_cv_scores = perform_cross_validation(knn_model, X, y, cv=CV_SPLITTER)
        print_cross_validation_results(knn_cv_scores, CV_FOLDS)
        knn_metrics["cv_accuracy"] = knn_cv_scores.mean()
        results["KNN"] = knn_metrics
        trained_models["KNN"] = knn_model

        # ------------------------------------------------------------------
        # 3d. XGBoost Classifier - Stage 1: RandomizedSearchCV, then
        #     Stage 2: retrain best params WITH early stopping
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Training: XGBoost Classifier")
        print("Stage 1: RandomizedSearchCV hyperparameter search "
              f"({N_SEARCH_ITER} combinations x {CV_FOLDS} folds)")
        print("=" * 60)
        xgb_search = search_xgboost_hyperparameters(
            X_train,
            y_train,
            param_distributions=XGB_PARAM_DISTRIBUTIONS,
            n_iter=N_SEARCH_ITER,
            cv=CV_SPLITTER,
            scoring=SEARCH_SCORING,
            random_state=RANDOM_STATE,
        )
        print_search_results(xgb_search, "XGBoost RandomizedSearchCV")

        print("\n" + "-" * 60)
        print("Stage 2: retraining best XGBoost configuration WITH "
              "early stopping (on a held-out validation split carved "
              "from the training data only)")
        print("-" * 60)
        xgb_model = train_xgboost_with_early_stopping(
            X_train,
            y_train,
            best_params=xgb_search.best_params_,
            validation_size=EARLY_STOPPING_VALIDATION_SIZE,
            max_estimators=EARLY_STOPPING_MAX_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_STATE,
        )
        print(f"Early stopping selected best_iteration = {xgb_model.best_iteration} "
              f"(out of up to {EARLY_STOPPING_MAX_ESTIMATORS} allowed rounds)")

        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        print_metrics(xgb_metrics)

        # Independent Stratified CV figure for the comparison table, using
        # the same tuned hyperparameters (without early stopping, since
        # cross_val_score does not support a per-fold validation split for
        # it) so the CV Accuracy column is computed the same way for every
        # model in this table.
        xgb_cv_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            **{k: v for k, v in xgb_search.best_params_.items()},
        )
        xgb_cv_scores = perform_cross_validation(xgb_cv_model, X, y, cv=CV_SPLITTER)
        print_cross_validation_results(xgb_cv_scores, CV_FOLDS)
        xgb_metrics["cv_accuracy"] = xgb_cv_scores.mean()
        results["XGBoost"] = xgb_metrics
        trained_models["XGBoost"] = xgb_model

        # ------------------------------------------------------------------
        # 4. Print the side-by-side comparison table for all four models
        # ------------------------------------------------------------------
        print_comparison_table(results)

        # ------------------------------------------------------------------
        # 5. Select the best model (highest Test Accuracy, tie-break on F1)
        # ------------------------------------------------------------------
        best_name = select_best_model(results)
        best_model = trained_models[best_name]
        best_metrics = results[best_name]

        # ------------------------------------------------------------------
        # 6. Save ONLY the best model, to the exact same path the existing
        #    Flask app (app.py) already loads from - nothing else changes.
        # ------------------------------------------------------------------
        save_model(best_model, MODEL_SAVE_PATH)

        # ------------------------------------------------------------------
        # 7. Print the final summary block
        # ------------------------------------------------------------------
        print_final_summary(best_name, best_metrics)
        print(f"\nBest model saved to: {MODEL_SAVE_PATH}")
        print("\nTraining pipeline completed successfully.")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Required file missing: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
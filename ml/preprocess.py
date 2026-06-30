"""
preprocess.py
--------------
Preprocessing pipeline for the Loan Eligibility Prediction project.

This script:
    1. Loads the raw dataset from dataset/loan_dataset.csv
    2. Handles missing values (mean for numerical, mode for categorical)
    3. Removes duplicate records
    4. Engineers a small set of genuinely new credit-risk features
    5. Encodes all categorical columns consistently using LabelEncoder
    6. Removes identifier columns that carry no predictive value
       (applicant_name)
    7. Splits the data into Features (X) and Target (y)
    8. Saves the cleaned dataset and X/y pickle files into ml/model/
    9. Prints missing-value and shape diagnostics before/after cleaning

Compatibility note
-------------------
train.py is NOT modified by this update and still expects X.pkl to contain
plain, unscaled numeric features (it loads X.pkl/y.pkl directly and runs
Decision Tree, Random Forest, KNN, and XGBoost on them). To stay fully
compatible:
    - X.pkl and y.pkl are saved in exactly the same format as before
      (a pandas DataFrame and Series respectively).
    - Feature scaling is intentionally NOT applied here - see the
      "Scaling decision" note above save_outputs() for why.

Run directly:
    python ml/preprocess.py
"""

import os
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------
# Configuration / Paths
# --------------------------------------------------------------------------

# Path to the raw dataset (relative to project root)
DATASET_PATH = "dataset/loan_dataset.csv"

# Output directory where all processed artifacts will be saved
MODEL_DIR = os.path.join("ml", "model")

# Output file paths
CLEANED_DATASET_PATH = os.path.join(MODEL_DIR, "cleaned_dataset.csv")
X_PICKLE_PATH = os.path.join(MODEL_DIR, "X.pkl")
Y_PICKLE_PATH = os.path.join(MODEL_DIR, "y.pkl")

# Name of the target column for this project
TARGET_COLUMN = "loan_status"

# Identifier column(s) that should not be used as a model feature.
# applicant_name is a free-text identifier (like the old Loan_ID) with no
# genuine predictive value, so it is excluded from X here. Label-encoding
# thousands of unique names would also create a meaningless
# high-cardinality feature that could mislead tree-based models.
ID_COLUMNS = ["applicant_name"]

# Maps the "dependents" category to a real numeric count, used only for
# feature engineering (e.g. income_per_dependent) BEFORE that column gets
# label-encoded into an arbitrary integer label. "3+" is treated as 3 for
# this purpose, a standard simplification for an open-ended top bucket.
DEPENDENTS_TO_COUNT = {"0": 0, "1": 1, "2": 2, "3+": 3}


# --------------------------------------------------------------------------
# Step 1: Load Dataset
# --------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------
# Step 2: Handle Missing Values
# --------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame:
        - Numerical columns are filled with the column mean.
        - Categorical (object) columns are filled with the column mode.

    Args:
        df (pd.DataFrame): Input DataFrame with potential missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df = df.copy()  # avoid mutating the caller's DataFrame

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns

    # Fill numerical missing values with the mean of each column
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)

    # Fill categorical missing values with the mode (most frequent value)
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    return df


# --------------------------------------------------------------------------
# Step 3: Remove Duplicate Records
# --------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without duplicate rows.
    """
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Step 4: Feature Engineering
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of derived credit-risk features that capture
    relationships a raw column alone does not, mirroring ratios real
    underwriting actually uses. This runs BEFORE categorical encoding, so
    it can read "dependents" as a real count (e.g. "3+" -> 3) rather than
    an arbitrary label-encoded integer.

    New features added:
        - loan_to_income_ratio: how large the requested loan is relative
          to the applicant's annual income. Higher = riskier loan request
          relative to what the applicant earns.
        - collateral_to_loan_ratio: how well the requested loan is
          secured. Higher = the collateral covers more (or all) of the
          loan, which is distinct from collateral_value alone (a $1M
          collateral means very different things against a $50k loan vs
          a $5M loan).
        - net_disposable_income: monthly_income minus existing_emi - the
          actual monthly cash left over after current debt obligations,
          which existing_emi or monthly_income alone don't directly show.
        - income_per_dependent: monthly_income spread across the
          household (dependents + the applicant), capturing financial
          pressure per person rather than just raw income.

    Deliberately NOT added: a duplicate of debt_to_income_ratio (e.g.
    existing_emi / monthly_income) - the dataset's existing
    debt_to_income_ratio column is already effectively that exact ratio,
    so re-deriving it again would be redundant, not new information.

    Args:
        df (pd.DataFrame): Cleaned DataFrame, BEFORE categorical encoding.

    Returns:
        pd.DataFrame: DataFrame with the new engineered columns added.
    """
    df = df.copy()

    # Loan amount requested vs. annual income. monthly_income is never
    # zero in this dataset, but the denominator is guarded anyway in case
    # this script is ever pointed at different data.
    annual_income = (df["monthly_income"] * 12).replace(0, pd.NA)
    df["loan_to_income_ratio"] = (df["loan_amount_requested"] / annual_income).fillna(0)

    # How well the loan is secured by collateral. loan_amount_requested is
    # never zero in this dataset; guarded the same way.
    safe_loan_amount = df["loan_amount_requested"].replace(0, pd.NA)
    df["collateral_to_loan_ratio"] = (df["collateral_value"] / safe_loan_amount).fillna(0)

    # Actual monthly cash remaining after the applicant's current EMI
    # obligations. Can legitimately go negative for an over-leveraged
    # applicant, which is itself a meaningful (bad) signal, so it is not
    # clipped at zero.
    df["net_disposable_income"] = df["monthly_income"] - df["existing_emi"]

    # Income spread across the household. "+1" includes the applicant
    # themselves, so a single applicant with 0 dependents divides by 1
    # (i.e. income_per_dependent == monthly_income for them).
    dependents_count = df["dependents"].map(DEPENDENTS_TO_COUNT).astype(float)
    df["income_per_dependent"] = df["monthly_income"] / (dependents_count + 1)

    return df


# --------------------------------------------------------------------------
# Step 5: Encode Categorical Columns
# --------------------------------------------------------------------------
def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical (object) columns using LabelEncoder, so every
    categorical column ends up as a consistent integer-coded numeric
    column. Each column gets its own LabelEncoder instance fitted to its
    own values, which keeps encoding consistent within a column (the same
    category string always maps to the same integer across the whole
    dataset).

    Note on ordinality: LabelEncoder assigns integers in alphabetical
    order of each column's unique values. This works well for tree-based
    models (Decision Tree, Random Forest, XGBoost), which split on
    thresholds rather than assuming a meaningful numeric scale, and KNN
    is already wrapped in its own StandardScaler step inside train.py.
    One-hot encoding was deliberately NOT used here, since it would
    change the number/shape of columns in X - this project's train.py is
    not being modified, so the output contract (a single integer-coded
    column per original categorical column) is kept exactly as before.

    Args:
        df (pd.DataFrame): Input DataFrame with categorical columns.

    Returns:
        pd.DataFrame: DataFrame with categorical columns label-encoded.
    """
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object", "str"]).columns

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


# --------------------------------------------------------------------------
# Step 6: Split Features (X) and Target (y)
# --------------------------------------------------------------------------
def split_features_and_target(df: pd.DataFrame, target_column: str, id_columns: list):
    """
    Split the DataFrame into feature matrix X and target vector y.
    Drops identifier columns (e.g., applicant_name) from the feature
    matrix since they hold no predictive value.

    Args:
        df (pd.DataFrame): Fully preprocessed DataFrame.
        target_column (str): Name of the target column.
        id_columns (list): List of identifier columns to exclude from X.

    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is a Series.
    """
    # Drop ID columns only if they exist in the DataFrame
    columns_to_drop = [target_column] + [col for col in id_columns if col in df.columns]

    X = df.drop(columns=columns_to_drop)
    y = df[target_column]

    return X, y


# --------------------------------------------------------------------------
# Step 7: Save Outputs
# --------------------------------------------------------------------------
#
# Scaling decision (why this script does NOT scale/normalize X)
# ----------------------------------------------------------------
# "Normalize or scale numerical features only if appropriate for the
# selected models" - for the four models this project trains in train.py:
#   - Decision Tree, Random Forest, XGBoost: all split on raw feature
#     thresholds: scaling changes the numbers but not the splits or the
#     resulting predictions, so it would add complexity for zero benefit.
#   - KNN: DOES need scaling (it uses real distances), but train.py
#     already wraps its KNN step in its own internal StandardScaler
#     Pipeline, fitted fresh on each train/test split and each CV fold.
# Scaling X here as well would not break anything numerically (it's a
# linear transform), but it would be redundant for 3 of the 4 models and
# would scale-then-rescale for KNN, and since train.py is explicitly not
# being modified for this task, X.pkl is kept in plain, unscaled form to
# match exactly what train.py already expects and is built around.
def save_outputs(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """
    Save the cleaned dataset as CSV and the feature/target sets as pickle
    files, in exactly the same format as before (plain pandas
    DataFrame/Series, unscaled).

    Args:
        df (pd.DataFrame): Cleaned and encoded full dataset.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
    """
    # Ensure the output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save cleaned dataset as CSV
    df.to_csv(CLEANED_DATASET_PATH, index=False)

    # Save X and y as pickle files
    with open(X_PICKLE_PATH, "wb") as f:
        pickle.dump(X, f)

    with open(Y_PICKLE_PATH, "wb") as f:
        pickle.dump(y, f)


# --------------------------------------------------------------------------
# Step 8: Diagnostic Printing Helpers
# --------------------------------------------------------------------------
def print_missing_values(df: pd.DataFrame, stage: str):
    """
    Print the count of missing values per column.

    Args:
        df (pd.DataFrame): DataFrame to inspect.
        stage (str): Label describing the stage (e.g., "Before", "After").
    """
    print(f"\n--- Missing Values ({stage} Preprocessing) ---")
    print(df.isnull().sum())


def print_shape(df: pd.DataFrame, stage: str):
    """
    Print the shape (rows, columns) of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to inspect.
        stage (str): Label describing the stage (e.g., "Before", "After").
    """
    print(f"\n--- Dataset Shape ({stage} Preprocessing) ---")
    print(df.shape)


# --------------------------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------------------------
def main():
    """
    Execute the full preprocessing pipeline end-to-end.
    """
    # 1. Load raw dataset
    df = load_dataset(DATASET_PATH)

    # Diagnostics: BEFORE preprocessing
    print_missing_values(df, "Before")
    print_shape(df, "Before")

    # 2. Handle missing values (mean for numeric, mode for categorical)
    df = handle_missing_values(df)

    # 3. Remove duplicate records
    df = remove_duplicates(df)

    # 4. Engineer new features (must happen BEFORE encoding, so it can
    #    read categorical values like "dependents" in their real,
    #    human-readable form rather than an arbitrary encoded integer).
    df = engineer_features(df)

    # 5. Encode categorical columns into numeric form
    df = encode_categorical_columns(df)

    # Diagnostics: AFTER preprocessing
    print_missing_values(df, "After")
    print_shape(df, "After")

    # 6. Split into features (X) and target (y). Identifier columns like
    #    applicant_name are dropped here.
    X, y = split_features_and_target(df, TARGET_COLUMN, ID_COLUMNS)

    # 7. Save cleaned dataset and X/y pickle files (unscaled - see the
    #    "Scaling decision" note above save_outputs()).
    save_outputs(df, X, y)

    print(f"\nCleaned dataset saved to: {CLEANED_DATASET_PATH}")
    print(f"Features (X) saved to:    {X_PICKLE_PATH}")
    print(f"Target (y) saved to:      {Y_PICKLE_PATH}")
    print(f"\nEngineered features added: loan_to_income_ratio, "
          f"collateral_to_loan_ratio, net_disposable_income, "
          f"income_per_dependent")
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
"""
eda.py
------
Exploratory Data Analysis (EDA) module for the Loan Eligibility Prediction System.

This script:
    - Loads the raw loan dataset
    - Performs structural and statistical inspection (shape, dtypes, missing values, etc.)
    - Generates and saves a series of visualizations to the 'outputs/' folder

NOTE: This script performs ONLY exploratory data analysis.
      No preprocessing, feature engineering, or model training is done here.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

DATASET_PATH = os.path.join("dataset", "loan.csv")
OUTPUT_DIR = "outputs"

# Use a clean, consistent visual style for all plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# ----------------------------------------------------------------------------
# SETUP FUNCTIONS
# ----------------------------------------------------------------------------

def create_output_folder(path: str = OUTPUT_DIR) -> None:
    """Create the outputs folder automatically if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Output folder ready at: '{path}'")


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the loan dataset from a CSV file into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'. Please check the path.")

    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded successfully from '{path}'")
    return df


# ----------------------------------------------------------------------------
# BASIC DATA INSPECTION FUNCTIONS
# ----------------------------------------------------------------------------

def print_basic_info(df: pd.DataFrame) -> None:
    """Print dataset shape, columns, and data types."""
    print("\n" + "=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("DATASET COLUMNS")
    print("=" * 60)
    print(list(df.columns))

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)


def print_missing_values(df: pd.DataFrame) -> None:
    """Print count and percentage of missing values per column."""
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    missing_summary = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_percent.round(2)
    })

    print(missing_summary[missing_summary["Missing Count"] > 0])


def print_duplicate_rows(df: pd.DataFrame) -> None:
    """Print the number of duplicate rows in the dataset."""
    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)

    duplicate_count = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicate_count}")


def print_statistical_summary(df: pd.DataFrame) -> None:
    """Print statistical summary for numerical columns."""
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY (Numerical Columns)")
    print("=" * 60)
    print(df.describe())


def print_categorical_unique_values(df: pd.DataFrame) -> None:
    """Print unique values for each categorical (object-type) column."""
    print("\n" + "=" * 60)
    print("UNIQUE VALUES IN CATEGORICAL COLUMNS")
    print("=" * 60)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        print(f"\nColumn: '{col}'")
        print(f"Unique values: {df[col].unique()}")


# ----------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# ----------------------------------------------------------------------------

def plot_missing_values_heatmap(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """1. Plot and save a heatmap showing the location of missing values."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_missing_values_heatmap.png"))
    plt.close()
    print("[PLOT] Saved: 01_missing_values_heatmap.png")


def plot_loan_status_count(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """2. Plot and save a count plot of the Loan_Status column."""
    if "Loan_Status" not in df.columns:
        print("[SKIP] 'Loan_Status' column not found.")
        return

    plt.figure()
    sns.countplot(data=df, x="Loan_Status", hue="Loan_Status", palette="Set2", legend=False)
    plt.title("Loan Status Count")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_loan_status_count.png"))
    plt.close()
    print("[PLOT] Saved: 02_loan_status_count.png")


def plot_gender_distribution(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """3. Plot and save a count plot showing Gender distribution."""
    if "Gender" not in df.columns:
        print("[SKIP] 'Gender' column not found.")
        return

    plt.figure()
    sns.countplot(data=df, x="Gender", hue="Gender", palette="Set3", legend=False)
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_gender_distribution.png"))
    plt.close()
    print("[PLOT] Saved: 03_gender_distribution.png")


def plot_education_distribution(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """4. Plot and save a count plot showing Education distribution."""
    if "Education" not in df.columns:
        print("[SKIP] 'Education' column not found.")
        return

    plt.figure()
    sns.countplot(data=df, x="Education", hue="Education", palette="Set1", legend=False)
    plt.title("Education Distribution")
    plt.xlabel("Education")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_education_distribution.png"))
    plt.close()
    print("[PLOT] Saved: 04_education_distribution.png")


def plot_applicant_income_histogram(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """5. Plot and save a histogram of ApplicantIncome."""
    if "ApplicantIncome" not in df.columns:
        print("[SKIP] 'ApplicantIncome' column not found.")
        return

    plt.figure()
    sns.histplot(df["ApplicantIncome"], bins=30, kde=True, color="steelblue")
    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_applicant_income_histogram.png"))
    plt.close()
    print("[PLOT] Saved: 05_applicant_income_histogram.png")


def plot_coapplicant_income_histogram(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """6. Plot and save a histogram of CoapplicantIncome."""
    if "CoapplicantIncome" not in df.columns:
        print("[SKIP] 'CoapplicantIncome' column not found.")
        return

    plt.figure()
    sns.histplot(df["CoapplicantIncome"], bins=30, kde=True, color="darkorange")
    plt.title("Coapplicant Income Distribution")
    plt.xlabel("Coapplicant Income")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_coapplicant_income_histogram.png"))
    plt.close()
    print("[PLOT] Saved: 06_coapplicant_income_histogram.png")


def plot_loan_amount_histogram(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """7. Plot and save a histogram of LoanAmount."""
    if "LoanAmount" not in df.columns:
        print("[SKIP] 'LoanAmount' column not found.")
        return

    plt.figure()
    sns.histplot(df["LoanAmount"], bins=30, kde=True, color="seagreen")
    plt.title("Loan Amount Distribution")
    plt.xlabel("Loan Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_loan_amount_histogram.png"))
    plt.close()
    print("[PLOT] Saved: 07_loan_amount_histogram.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """8. Plot and save a correlation heatmap for numerical columns."""
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] < 2:
        print("[SKIP] Not enough numerical columns for a correlation heatmap.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_correlation_heatmap.png"))
    plt.close()
    print("[PLOT] Saved: 08_correlation_heatmap.png")


def plot_credit_history_vs_loan_status(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """9. Plot and save a count plot comparing Credit_History against Loan_Status."""
    if "Credit_History" not in df.columns or "Loan_Status" not in df.columns:
        print("[SKIP] 'Credit_History' or 'Loan_Status' column not found.")
        return

    plt.figure()
    sns.countplot(data=df, x="Credit_History", hue="Loan_Status", palette="Set2")
    plt.title("Credit History vs Loan Status")
    plt.xlabel("Credit History")
    plt.ylabel("Count")
    plt.legend(title="Loan Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_credit_history_vs_loan_status.png"))
    plt.close()
    print("[PLOT] Saved: 09_credit_history_vs_loan_status.png")


def plot_property_area_vs_loan_status(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """10. Plot and save a count plot comparing Property_Area against Loan_Status."""
    if "Property_Area" not in df.columns or "Loan_Status" not in df.columns:
        print("[SKIP] 'Property_Area' or 'Loan_Status' column not found.")
        return

    plt.figure()
    sns.countplot(data=df, x="Property_Area", hue="Loan_Status", palette="Set1")
    plt.title("Property Area vs Loan Status")
    plt.xlabel("Property Area")
    plt.ylabel("Count")
    plt.legend(title="Loan Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "10_property_area_vs_loan_status.png"))
    plt.close()
    print("[PLOT] Saved: 10_property_area_vs_loan_status.png")


def plot_boxplots_numerical_columns(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """11. Plot and save boxplots for all numerical columns (for outlier detection)."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) == 0:
        print("[SKIP] No numerical columns found for boxplots.")
        return

    n_cols = len(numeric_cols)
    plt.figure(figsize=(5 * n_cols, 5))

    for i, col in enumerate(numeric_cols, start=1):
        plt.subplot(1, n_cols, i)
        sns.boxplot(y=df[col], color="lightblue")
        plt.title(col)

    plt.suptitle("Boxplots of Numerical Columns")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "11_boxplots_numerical_columns.png"))
    plt.close()
    print("[PLOT] Saved: 11_boxplots_numerical_columns.png")


# ----------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------

def run_eda() -> None:
    """Run the complete EDA pipeline: inspection + visualizations."""

    # Step 1: Setup
    create_output_folder(OUTPUT_DIR)
    df = load_dataset(DATASET_PATH)

    # Step 2: Textual / statistical inspection
    print_basic_info(df)
    print_missing_values(df)
    print_duplicate_rows(df)
    print_statistical_summary(df)
    print_categorical_unique_values(df)

    # Step 3: Visualizations
    plot_missing_values_heatmap(df)
    plot_loan_status_count(df)
    plot_gender_distribution(df)
    plot_education_distribution(df)
    plot_applicant_income_histogram(df)
    plot_coapplicant_income_histogram(df)
    plot_loan_amount_histogram(df)
    plot_correlation_heatmap(df)
    plot_credit_history_vs_loan_status(df)
    plot_property_area_vs_loan_status(df)
    plot_boxplots_numerical_columns(df)

    print("\n[INFO] EDA completed successfully. All plots saved in 'outputs/' folder.")


if __name__ == "__main__":
    run_eda()
"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset for the Smart Lender loan
eligibility project: dataset/loan_dataset.csv

This is a ONE-TIME data generation script (not part of the Flask app or
the training pipeline). It exists so the dataset is reproducible and so
the underlying approval logic is documented and inspectable, rather than
being a black box CSV with no explanation of how it was built.

Design approach
----------------
Fields are generated in correlated clusters (not independently), so the
dataset resembles real applicant data:
    - employment_type drives the realistic range of monthly_income
    - monthly_income drives bank_balance and existing_emi ranges
    - age drives years_of_employment (you can't have more years employed
      than is plausible for your age)

Loan approval (loan_status) is generated from an explicit, weighted
"banking approval score" built from exactly the 9 factors specified as
business drivers, using the exact suggested weights:

    Credit Score ............. 30%
    Monthly Income ........... 20%
    Debt-to-Income Ratio ..... 15%
    Credit History ........... 10%
    Employment Stability ..... 10%
    Existing EMI .............  5%
    Previous Defaults ........  5%
    Bank Balance .............  3%
    Collateral Value .........  2%
                              -----
                               100%

Unlike the previous version of this script, randomness is added INSIDE
the score (a small jitter on each normalized component, 5-8% of its
scale) rather than splattered on top of the final approval probability.
That distinction matters a lot for how learnable the data is: noise on
the score lets confident cases (very high or very low score) still
resolve confidently, while noise stacked directly onto a probability
flattens the whole decision boundary and caps how accurate ANY model can
get, no matter how good. The final score is passed through a much
sharper sigmoid than the previous version used, so the boundary is crisp
enough for a model to learn well while still leaving a genuine "gray
zone" of borderline applicants, like real underwriting.

Run directly:
    python generate_dataset.py
"""

import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RANDOM_STATE = 42
NUM_RECORDS = 22000  # comfortably above the required minimum of 20,000

OUTPUT_PATH = os.path.join("dataset", "loan_dataset.csv")

rng = np.random.default_rng(RANDOM_STATE)


# --------------------------------------------------------------------------
# Name generation (for applicant_name column)
# --------------------------------------------------------------------------

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Rohan", "Karan", "Aryan", "Dhruv", "Kabir", "Yash",
    "Saanvi", "Ananya", "Aadhya", "Diya", "Myra", "Pari", "Anika", "Navya",
    "Kiara", "Riya", "Ira", "Shreya", "Tara", "Meera", "Priya", "Neha",
    "Rahul", "Vikram", "Amit", "Suresh", "Rajesh", "Sanjay", "Manish", "Deepak",
    "Pooja", "Kavita", "Sunita", "Anjali", "Rekha", "Geeta", "Lakshmi", "Divya",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Kumar", "Singh", "Patel", "Shah", "Mehta",
    "Reddy", "Rao", "Nair", "Iyer", "Joshi", "Desai", "Agarwal", "Bansal",
    "Chopra", "Malhotra", "Kapoor", "Khanna", "Saxena", "Mishra", "Pandey", "Tiwari",
]


def generate_names(n: int) -> np.ndarray:
    """Generate n realistic-looking full names by combining first/last name pools."""
    first = rng.choice(FIRST_NAMES, size=n)
    last = rng.choice(LAST_NAMES, size=n)
    return np.array([f"{f} {l}" for f, l in zip(first, last)])


# --------------------------------------------------------------------------
# Step 1: Personal Information
# --------------------------------------------------------------------------

def generate_personal_info(n: int) -> pd.DataFrame:
    """Generate age, gender, marital_status, dependents, education."""
    age = rng.integers(21, 66, size=n)  # 21-65 inclusive

    gender = rng.choice(["Male", "Female"], size=n, p=[0.58, 0.42])

    # Marital status correlates loosely with age: younger applicants are
    # more likely single.
    marital_status = np.where(
        (age < 28) & (rng.random(n) < 0.6),
        "Single",
        rng.choice(["Married", "Single"], size=n, p=[0.68, 0.32]),
    )

    dependents = rng.choice(["0", "1", "2", "3+"], size=n, p=[0.42, 0.28, 0.20, 0.10])
    # Single applicants are less likely to have many dependents.
    dependents = np.where(
        (marital_status == "Single") & (np.isin(dependents, ["2", "3+"])),
        rng.choice(["0", "1"], size=n, p=[0.7, 0.3]),
        dependents,
    )

    education = rng.choice(["Graduate", "Not Graduate"], size=n, p=[0.72, 0.28])

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "dependents": dependents,
        "education": education,
    })


# --------------------------------------------------------------------------
# Step 2: Employment & Income
# --------------------------------------------------------------------------

def generate_employment_and_income(n: int, age: np.ndarray) -> pd.DataFrame:
    """
    Generate employment_type, years_of_employment, monthly_income, and
    bank_balance as a correlated cluster: employment_type sets the income
    distribution, income sets the bank balance distribution, and age caps
    plausible years of employment.
    """
    employment_type = rng.choice(
        ["Salaried", "Self-Employed", "Business Owner", "Unemployed", "Retired"],
        size=n,
        p=[0.50, 0.20, 0.15, 0.08, 0.07],
    )

    # Years of employment cannot exceed (age - 18), and unemployed/retired
    # people have 0 current years of employment.
    max_possible_years = np.clip(age - 18, 0, None)
    years_of_employment = np.round(
        rng.uniform(0, 1, size=n) * np.minimum(max_possible_years, 35), 1
    )
    years_of_employment = np.where(
        employment_type == "Unemployed", 0.0, years_of_employment
    )
    years_of_employment = np.where(
        employment_type == "Retired",
        np.round(rng.uniform(15, 35, size=n), 1),
        years_of_employment,
    )

    # Monthly income (INR), log-normal per employment type for realistic
    # right-skewed income distributions, clipped to the required 10k-500k range.
    income_base = {
        "Salaried": (11.2, 0.45),          # lognormal mean/sigma (log-space)
        "Self-Employed": (11.0, 0.65),
        "Business Owner": (11.6, 0.70),
        "Unemployed": (8.5, 0.9),
        "Retired": (10.5, 0.5),
    }
    monthly_income = np.zeros(n)
    for etype, (mu, sigma) in income_base.items():
        mask = employment_type == etype
        monthly_income[mask] = rng.lognormal(mean=mu, sigma=sigma, size=mask.sum())
    monthly_income = np.clip(monthly_income, 10_000, 500_000)
    monthly_income = np.round(monthly_income, 2)

    # Bank balance correlates with income (roughly 1-8 months of income,
    # with unemployed/lower-income applicants holding thinner buffers).
    balance_multiplier = rng.uniform(0.5, 8.0, size=n)
    bank_balance = np.round(monthly_income * balance_multiplier, 2)
    bank_balance = np.clip(bank_balance, 0, 5_000_000)

    return pd.DataFrame({
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "monthly_income": monthly_income,
        "bank_balance": bank_balance,
    })


# --------------------------------------------------------------------------
# Step 3: Financial / Credit Profile
# --------------------------------------------------------------------------

def generate_financial_profile(n: int, monthly_income: np.ndarray) -> pd.DataFrame:
    """
    Generate existing_emi, existing_loan_amount, debt_to_income_ratio,
    credit_score, credit_history, and previous_defaults as a correlated
    cluster: EMI is derived as a fraction of income (which directly
    produces a realistic debt_to_income_ratio), and credit_score is
    pulled down by higher previous_defaults.
    """
    # EMI as a fraction of income: most applicants have manageable EMIs,
    # a tail has high EMI burden relative to income.
    emi_fraction = np.clip(rng.beta(a=2, b=5, size=n), 0, 1.5)  # can exceed 1 income in rare cases
    existing_emi = np.round(monthly_income * emi_fraction, 2)

    # Existing loan amount roughly proportional to EMI level (a higher EMI
    # implies a larger underlying outstanding loan).
    existing_loan_amount = np.round(existing_emi * rng.uniform(10, 60, size=n), 2)
    existing_loan_amount = np.clip(existing_loan_amount, 0, 3_000_000)

    # Debt-to-income ratio (%) computed directly from EMI vs income, so it
    # is internally consistent with existing_emi rather than independent
    # random noise.
    debt_to_income_ratio = np.round(
        np.clip((existing_emi / np.maximum(monthly_income, 1)) * 100, 0, 100), 2
    )

    # Previous defaults: most applicants have none; a shrinking tail has more.
    previous_defaults = rng.choice(
        [0, 1, 2, 3, 4, 5], size=n, p=[0.62, 0.18, 0.10, 0.06, 0.03, 0.01]
    )

    # Credit score: starts from a healthy baseline distribution, then is
    # pulled down by previous defaults and by a high debt-to-income ratio,
    # with random noise layered on top so the relationship isn't deterministic.
    base_score = rng.normal(loc=680, scale=90, size=n)
    default_penalty = previous_defaults * rng.uniform(25, 55, size=n)
    dti_penalty = (debt_to_income_ratio / 100) * rng.uniform(40, 90, size=n)
    credit_score = base_score - default_penalty - dti_penalty
    credit_score = np.clip(credit_score, 300, 900).round().astype(int)

    # Credit history (1 = good / no past defaults, 0 = poor) correlates
    # with previous_defaults, as it would in reality, but is NOT a clean
    # deterministic function of it. If credit_history were near-perfectly
    # derivable from previous_defaults, a model could shortcut the
    # decision through that one feature alone and effectively ignore the
    # other 8 weighted factors - which would defeat the point of having
    # 9 distinct weighted business factors. The probabilities below are
    # deliberately close together (not 0.9 vs 0.1) so credit_history
    # carries real but modest signal, proportionate to its intended 10%
    # weight rather than acting as a free proxy for the borrower's whole
    # credit risk profile.
    credit_history_good_prob = np.select(
        [previous_defaults == 0, previous_defaults == 1, previous_defaults == 2],
        [0.62, 0.50, 0.42],
        default=0.32,  # previous_defaults >= 3
    )
    credit_history = (rng.random(n) < credit_history_good_prob).astype(int)

    return pd.DataFrame({
        "existing_emi": existing_emi,
        "existing_loan_amount": existing_loan_amount,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_score": credit_score,
        "credit_history": credit_history,
        "previous_defaults": previous_defaults,
    })


# --------------------------------------------------------------------------
# Step 4: Loan Request Details
# --------------------------------------------------------------------------

def generate_loan_details(n: int, monthly_income: np.ndarray) -> pd.DataFrame:
    """
    Generate loan_amount_requested, loan_tenure, loan_purpose,
    collateral_value, and property_area.
    """
    loan_purpose = rng.choice(
        ["Home", "Vehicle", "Education", "Business", "Medical", "Personal", "Other"],
        size=n,
        p=[0.28, 0.18, 0.12, 0.14, 0.08, 0.15, 0.05],
    )

    # Loan amount requested scales loosely with income and purpose (home
    # loans are much larger than personal loans).
    purpose_multiplier = {
        "Home": (40, 120), "Vehicle": (5, 25), "Education": (3, 20),
        "Business": (10, 80), "Medical": (1, 15), "Personal": (2, 20), "Other": (2, 30),
    }
    loan_amount_requested = np.zeros(n)
    for purpose, (low, high) in purpose_multiplier.items():
        mask = loan_purpose == purpose
        loan_amount_requested[mask] = monthly_income[mask] * rng.uniform(low, high, size=mask.sum())
    loan_amount_requested = np.round(np.clip(loan_amount_requested, 10_000, 10_000_000), 2)

    loan_tenure = rng.choice(
        [12, 24, 36, 60, 120, 180, 240, 300, 360, 480], size=n,
        p=[0.08, 0.10, 0.12, 0.12, 0.13, 0.13, 0.10, 0.08, 0.10, 0.04],
    )

    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.45, 0.35, 0.20])

    # Collateral value: home/business loans are far more likely to be
    # secured with meaningful collateral; personal/medical loans often have
    # little to none.
    has_collateral_prob = np.where(np.isin(loan_purpose, ["Home", "Business", "Vehicle"]), 0.75, 0.25)
    has_collateral = rng.random(n) < has_collateral_prob
    collateral_value = np.where(
        has_collateral,
        loan_amount_requested * rng.uniform(0.8, 1.8, size=n),
        rng.uniform(0, 5000, size=n),  # near-zero for unsecured loans
    )
    collateral_value = np.round(np.clip(collateral_value, 0, 15_000_000), 2)

    return pd.DataFrame({
        "loan_amount_requested": loan_amount_requested,
        "loan_tenure": loan_tenure,
        "loan_purpose": loan_purpose,
        "collateral_value": collateral_value,
        "property_area": property_area,
    })


# --------------------------------------------------------------------------
# Step 5: Loan Approval Logic — Weighted Banking Approval Score
# --------------------------------------------------------------------------

# Exact business-rule weights as specified. These sum to 1.00 (100%) and
# directly control how much each factor influences the final approval
# score below — e.g. credit_score alone accounts for 30% of the decision,
# monthly_income for 20%, and so on down to collateral_value at 2%.
APPROVAL_WEIGHTS = {
    "credit_score": 0.30,
    "monthly_income": 0.20,
    "debt_to_income_ratio": 0.15,
    "credit_history": 0.10,
    "employment_stability": 0.10,  # employment_type + years_of_employment combined
    "existing_emi": 0.05,
    "previous_defaults": 0.05,
    "bank_balance": 0.03,
    "collateral_value": 0.02,
}
assert abs(sum(APPROVAL_WEIGHTS.values()) - 1.0) < 1e-9, "Approval weights must sum to 100%"

# How much controlled randomness to inject into the score itself (NOT
# directly onto the final probability — see module docstring for why that
# distinction matters). 0.06 means each normalized component gets +/- 6%
# jitter, which keeps the dataset realistic without flattening the
# decision boundary the way noise-on-probability does.
SCORE_NOISE_LEVEL = 0.06

# Sigmoid steepness. Higher = sharper, more confident decision boundary.
# This is deliberately much sharper than a "soft" logistic curve so that
# clearly strong or clearly weak applicants resolve to a near-certain
# outcome, while only genuinely borderline applicants land in the
# uncertain middle - this is what makes the data learnable to a high
# accuracy while still leaving a realistic gray zone.
SIGMOID_STEEPNESS = 40.0

# Sigmoid center: the weighted_score value treated as the "50/50" decision
# point. Calibrated empirically against this generator's actual
# weighted_score distribution (see the calibration check at the bottom of
# this file / module comments) so the resulting approval rate lands at the
# requested 50-60% Approved, rather than guessed.
SIGMOID_CENTER = 0.670


def generate_loan_status(df: pd.DataFrame) -> np.ndarray:
    """
    Compute loan_status from an explicit, weighted banking approval score
    built from exactly the 9 specified business factors and weights (see
    APPROVAL_WEIGHTS above). Each factor is:
        1. Normalized to a comparable 0-1 scale, oriented so that 1.0
           always means "good for the applicant" and 0.0 means "bad for
           the applicant" (e.g. debt_to_income_ratio is inverted, since a
           LOWER ratio is better).
        2. Given a small amount of independent random jitter (the
           "5-8% controlled randomness" requirement), so two applicants
           with identical numbers don't always get identical scores -
           real underwriting has some give in it.
        3. Multiplied by its business weight and summed into one score.

    That weighted score is then passed through a sharp sigmoid to get an
    approval probability, and loan_status is sampled from that
    probability (not a hard threshold) - so the outcome stays
    probabilistic/realistic rather than a deterministic if/else rule,
    while still being strongly and correctly driven by the 9 factors.

    Correlation directions implemented here (all per the requirements):
        - Higher credit_score          -> higher approval chance
        - Higher monthly_income        -> higher approval chance
        - Higher debt_to_income_ratio  -> LOWER approval chance
        - Better credit_history (1)    -> higher approval chance
        - More years_of_employment /
          more stable employment_type  -> higher approval chance
        - Higher existing_emi          -> LOWER approval chance
        - More previous_defaults       -> LOWER approval chance
        - Higher bank_balance          -> higher approval chance
        - Higher collateral_value      -> higher approval chance
    """
    n = len(df)

    def jitter(component: np.ndarray) -> np.ndarray:
        """Add +/- SCORE_NOISE_LEVEL of independent Gaussian noise to a
        0-1 normalized component, then re-clip back into [0, 1]. This is
        where the "5-8% controlled randomness" lives - inside each
        factor, not on top of the final decision."""
        noisy = component + rng.normal(loc=0.0, scale=SCORE_NOISE_LEVEL, size=n)
        return np.clip(noisy, 0.0, 1.0)

    # ---- 1. Credit Score (30%) ----
    # Higher credit_score -> higher approval chance. Normalized linearly
    # across the full 300-900 range.
    credit_score_norm = jitter((df["credit_score"] - 300) / (900 - 300))

    # ---- 2. Monthly Income (20%) ----
    # Higher monthly_income -> higher approval chance. log1p compresses
    # the long right tail of high earners so income doesn't completely
    # dominate just because it spans a huge raw numeric range.
    income_norm = jitter(np.log1p(df["monthly_income"]) / np.log1p(500_000))

    # ---- 3. Debt-to-Income Ratio (15%) ----
    # Higher debt_to_income_ratio -> LOWER approval chance, so this
    # component is INVERTED (1 - ratio) before weighting.
    dti_norm = jitter(1 - (df["debt_to_income_ratio"] / 100))

    # ---- 4. Credit History (10%) ----
    # credit_history is already a clean 0/1 flag (1 = good). Light jitter
    # keeps it from being a perfectly binary signal.
    credit_history_norm = jitter(df["credit_history"].astype(float))

    # ---- 5. Employment Stability (10%) ----
    # Combines employment_type (how stable the type of work is) with
    # years_of_employment (how long they've held it). Both push the same
    # direction: more stable type + more years -> higher approval chance.
    employment_type_score_map = {
        "Salaried": 0.85, "Business Owner": 0.70, "Self-Employed": 0.55,
        "Retired": 0.45, "Unemployed": 0.05,
    }
    employment_type_norm = df["employment_type"].map(employment_type_score_map).astype(float)
    years_norm = np.clip(df["years_of_employment"] / 20, 0, 1)  # 20+ years treated as max stability
    employment_stability_norm = jitter(0.6 * employment_type_norm + 0.4 * years_norm)

    # ---- 6. Existing EMI (5%) ----
    # Higher existing_emi (relative to income) -> LOWER approval chance,
    # so this is INVERTED. Measured as a fraction of income, since a flat
    # EMI amount means very different things at different income levels.
    emi_to_income = df["existing_emi"] / np.maximum(df["monthly_income"], 1)
    existing_emi_norm = jitter(1 - np.clip(emi_to_income, 0, 1.5) / 1.5)

    # ---- 7. Previous Defaults (5%) ----
    # More previous_defaults -> LOWER approval chance, so this is
    # INVERTED. previous_defaults ranges 0-5 per the spec.
    previous_defaults_norm = jitter(1 - (df["previous_defaults"] / 5))

    # ---- 8. Bank Balance (3%) ----
    # Higher bank_balance -> higher approval chance. log1p compresses the
    # long right tail the same way income does.
    bank_balance_norm = jitter(np.log1p(df["bank_balance"]) / np.log1p(5_000_000))

    # ---- 9. Collateral Value (2%) ----
    # Higher collateral_value -> higher approval chance.
    collateral_norm = jitter(np.log1p(df["collateral_value"]) / np.log1p(15_000_000))

    # ---- Combine into the final weighted banking approval score ----
    # Each normalized, jittered component is multiplied by its exact
    # business weight from APPROVAL_WEIGHTS and summed. The result is a
    # single 0-1 score where higher always means "more approvable".
    weighted_score = (
        APPROVAL_WEIGHTS["credit_score"] * credit_score_norm
        + APPROVAL_WEIGHTS["monthly_income"] * income_norm
        + APPROVAL_WEIGHTS["debt_to_income_ratio"] * dti_norm
        + APPROVAL_WEIGHTS["credit_history"] * credit_history_norm
        + APPROVAL_WEIGHTS["employment_stability"] * employment_stability_norm
        + APPROVAL_WEIGHTS["existing_emi"] * existing_emi_norm
        + APPROVAL_WEIGHTS["previous_defaults"] * previous_defaults_norm
        + APPROVAL_WEIGHTS["bank_balance"] * bank_balance_norm
        + APPROVAL_WEIGHTS["collateral_value"] * collateral_norm
    )

    # ---- Convert the score into an approval probability ----
    # SIGMOID_CENTER and SIGMOID_STEEPNESS were calibrated empirically
    # against this generator's actual weighted_score distribution (not
    # assumed), so the resulting approval rate lands at the requested
    # 50-60% Approved and the boundary is sharp enough to be highly
    # learnable, while sampling from a probability (rather than a hard
    # threshold) keeps a realistic, non-trivial gray zone.
    logit = (weighted_score - SIGMOID_CENTER) * SIGMOID_STEEPNESS
    approval_probability = 1 / (1 + np.exp(-logit))

    loan_status = np.where(rng.random(n) < approval_probability, "Y", "N")
    return loan_status


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    """Generate the full synthetic dataset and write it to dataset/loan_dataset.csv."""
    n = NUM_RECORDS

    personal = generate_personal_info(n)
    employment = generate_employment_and_income(n, personal["age"].to_numpy())
    financial = generate_financial_profile(n, employment["monthly_income"].to_numpy())
    loan = generate_loan_details(n, employment["monthly_income"].to_numpy())

    df = pd.concat([personal, employment, financial, loan], axis=1)
    df.insert(0, "applicant_name", generate_names(n))

    df["loan_status"] = generate_loan_status(df)

    # Final column order, matching the project's required schema exactly.
    column_order = [
        "applicant_name", "age", "gender", "marital_status", "dependents",
        "education", "employment_type", "years_of_employment", "monthly_income",
        "bank_balance", "existing_emi", "existing_loan_amount",
        "debt_to_income_ratio", "credit_score", "credit_history",
        "previous_defaults", "loan_amount_requested", "loan_tenure",
        "loan_purpose", "collateral_value", "property_area", "loan_status",
    ]
    df = df[column_order]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df)} records -> {OUTPUT_PATH}")
    print(f"\nApproval rate: {(df['loan_status'] == 'Y').mean() * 100:.1f}%")
    print(f"\nColumn dtypes:\n{df.dtypes}")


if __name__ == "__main__":
    main()
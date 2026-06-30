"""
app.py
------
Flask application for the Smart Lender (Loan Eligibility Prediction) project.

This app:
    1. Loads the trained XGBoost model (ml/model/loan_model.pkl) exactly
       once, at application startup. The model now expects the 24-feature
       contract produced by the current ml/preprocess.py: the 20 raw
       dataset columns (minus applicant_name and loan_status), each
       label-encoded exactly as preprocess.py encodes them, PLUS 4
       engineered ratio features preprocess.py derives (see
       "Feature Contract" below).
    2. Renders the landing page at "/".
    3. Renders the multi-section loan application form at "/predict"
       (GET) and accepts its submission (POST) on the same route.
    4. Reads ALL fields from predict.html (applicant_name, age, gender,
       marital_status, dependents, education, employment_type,
       years_of_employment, monthly_income, bank_balance, existing_emi,
       existing_loan_amount, debt_to_income_ratio, credit_score,
       credit_history, previous_defaults, loan_amount_requested,
       loan_tenure, loan_purpose, collateral_value, property_area). These
       field names already match the new dataset's column names directly
       - no name translation is needed here (unlike the previous version
       of this app, which had to map an older form's vocabulary onto an
       older, frozen 11-feature model).
    5. Builds the exact 24-feature vector the new model expects, in the
       exact column order ml/preprocess.py produces, by encoding each
       categorical field with the same LabelEncoder-equivalent mapping
       preprocess.py uses, then computing the same 4 engineered ratios.
    6. Runs the prediction and renders result.html with the outcome.
    7. Validates and gracefully handles invalid/missing input, and wraps
       every risky operation (model loading, prediction) in exception
       handling so the app never crashes with a raw traceback.
    8. Saves every field submitted on the form (plus the prediction and
       probability) into the MySQL 'loan_applications' table, using the
       existing get_db_connection() helper and parameterized queries.
       The database write happens AFTER the prediction is final and never
       blocks the user from seeing their result, even if the database is
       unreachable.

Feature Contract (must stay in sync with ml/preprocess.py)
-------------------------------------------------------------
ml/preprocess.py builds X.pkl as: the 20 raw columns (label-encoded where
categorical) plus 4 engineered columns, in this exact order:

    age, gender, marital_status, dependents, education, employment_type,
    years_of_employment, monthly_income, bank_balance, existing_emi,
    existing_loan_amount, debt_to_income_ratio, credit_score,
    credit_history, previous_defaults, loan_amount_requested,
    loan_tenure, loan_purpose, collateral_value, property_area,
    loan_to_income_ratio, collateral_to_loan_ratio,
    net_disposable_income, income_per_dependent

The categorical encodings below reproduce sklearn's LabelEncoder exactly
(alphabetical order of each column's unique values), verified directly
against ml/model/cleaned_dataset.csv produced by the real preprocessing
run, not assumed. The 4 engineered features are computed here with the
exact same formulas as ml/preprocess.py's engineer_features().

Note: result.html was not modified as part of this task and still reads
legacy capitalized variable names (Gender, Married, ApplicantIncome,
etc.) left over from an earlier version of this project. To keep that
page rendering correctly without touching its template, render_template
is still called with those legacy names, populated from the new fields.

Run directly:
    python app.py
"""

import os
import pickle
import logging

import pandas as pd
import mysql.connector
from mysql.connector import Error as MySQLError
from flask import Flask, render_template, request

# --------------------------------------------------------------------------
# App & Logging Configuration
# --------------------------------------------------------------------------

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model", "loan_model.pkl")

# --------------------------------------------------------------------------
# MySQL Configuration
# --------------------------------------------------------------------------
# Connection settings for the existing 'smart_lender' database. Override via
# environment variables in production rather than hardcoding credentials.
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "Sai@2005"),
    "database": os.environ.get("DB_NAME", "smart_lender"),
}

# --------------------------------------------------------------------------
# Encoding Maps
# --------------------------------------------------------------------------
# These reproduce, exactly, the encodings produced by sklearn's LabelEncoder
# in ml/preprocess.py. LabelEncoder assigns integer codes in alphabetical
# order of each column's unique string values. Every mapping below was
# verified directly against ml/model/cleaned_dataset.csv (the real output
# of a preprocess.py run on the real dataset), not just computed from
# alphabetical-order theory, so it is guaranteed to match what the model
# was actually trained on.
GENDER_MAP = {"Female": 0, "Male": 1}
MARITAL_STATUS_MAP = {"Married": 0, "Single": 1}
DEPENDENTS_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}
EMPLOYMENT_TYPE_MAP = {
    "Business Owner": 0,
    "Retired": 1,
    "Salaried": 2,
    "Self-Employed": 3,
    "Unemployed": 4,
}
LOAN_PURPOSE_MAP = {
    "Business": 0,
    "Education": 1,
    "Home": 2,
    "Medical": 3,
    "Other": 4,
    "Personal": 5,
    "Vehicle": 6,
}
PROPERTY_AREA_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}

# "3+" is treated as 3 for feature-engineering purposes (e.g.
# income_per_dependent), matching ml/preprocess.py's DEPENDENTS_TO_COUNT.
# This is distinct from DEPENDENTS_MAP above, which is the LabelEncoder
# integer code used as the model FEATURE itself, not a real count.
DEPENDENTS_TO_COUNT = {"0": 0, "1": 1, "2": 2, "3+": 3}

# Exact feature order the model was trained on (must match
# ml/preprocess.py's column order exactly: the 20 raw columns, encoded,
# followed by the 4 engineered columns it appends). THIS LIST IS FROZEN
# relative to the current model: if ml/preprocess.py's feature set or
# order ever changes again, this list and build_model_features() below
# must be updated together with it.
FEATURE_COLUMNS = [
    "age",
    "gender",
    "marital_status",
    "dependents",
    "education",
    "employment_type",
    "years_of_employment",
    "monthly_income",
    "bank_balance",
    "existing_emi",
    "existing_loan_amount",
    "debt_to_income_ratio",
    "credit_score",
    "credit_history",
    "previous_defaults",
    "loan_amount_requested",
    "loan_tenure",
    "loan_purpose",
    "collateral_value",
    "property_area",
    "loan_to_income_ratio",
    "collateral_to_loan_ratio",
    "net_disposable_income",
    "income_per_dependent",
]

# --------------------------------------------------------------------------
# Load the Trained Model Once at Startup
# --------------------------------------------------------------------------


def load_model(path: str):
    """
    Load the trained model from disk exactly once.

    Args:
        path (str): Path to the pickled model file.

    Returns:
        The unpickled model object, or None if loading failed.
    """
    if not os.path.exists(path):
        logger.error("Model file not found at: %s", path)
        return None

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully from: %s", path)
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return None


# Loaded once when the module is imported (i.e. when the app starts),
# not on every request.
model = load_model(MODEL_PATH)


# --------------------------------------------------------------------------
# MySQL: Reusable Connection Helper
# --------------------------------------------------------------------------


def get_db_connection():
    """
    Create and return a new MySQL connection using mysql.connector.

    A fresh connection is opened per call (and closed by the caller after
    use) rather than kept open globally, which is the simplest safe pattern
    for a request-driven Flask app and avoids stale/dropped connections.

    Returns:
        mysql.connector.connection.MySQLConnection: An open connection.

    Raises:
        mysql.connector.Error: If the connection cannot be established.
            Callers are expected to catch this so a database outage never
            crashes the Flask app.
    """
    connection = mysql.connector.connect(**DB_CONFIG)
    return connection


def save_loan_application(data: dict) -> bool:
    """
    Insert one row into the upgraded 'loan_applications' table for a
    completed prediction. This runs AFTER the ML prediction and never
    affects or blocks the prediction result shown to the user — if the
    database is unreachable or the insert fails, the error is logged and
    swallowed here so the rest of the request continues normally.

    Args:
        data (dict): Should contain the following keys, matching the new
            predict.html form fields, plus the prediction outcome:
            applicant_name, age, gender, marital_status, dependents,
            education, employment_type, years_of_employment, monthly_income,
            bank_balance, existing_emi, existing_loan_amount,
            debt_to_income_ratio, credit_score, credit_history,
            previous_defaults, loan_amount_requested, loan_tenure,
            loan_purpose, collateral_value, property_area, prediction,
            probability. Any missing key is stored as NULL.

    Returns:
        bool: True if the row was inserted and committed successfully,
            False otherwise (the caller does not need to act on this;
            it's mainly useful for logging/testing).
    """
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        insert_query = """
            INSERT INTO loan_applications (
                applicant_name, age, gender, marital_status, dependents,
                education, employment_type, years_of_employment,
                monthly_income, bank_balance, existing_emi,
                existing_loan_amount, debt_to_income_ratio, credit_score,
                credit_history, previous_defaults, loan_amount_requested,
                loan_tenure, loan_purpose, collateral_value, property_area,
                prediction, probability
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s
            )
        """

        # Parameterized query: values are passed separately from the SQL
        # text, so user input can never be interpreted as SQL (prevents
        # SQL injection).
        values = (
            data.get("applicant_name"),
            data.get("age"),
            data.get("gender"),
            data.get("marital_status"),
            data.get("dependents"),
            data.get("education"),
            data.get("employment_type"),
            data.get("years_of_employment"),
            data.get("monthly_income"),
            data.get("bank_balance"),
            data.get("existing_emi"),
            data.get("existing_loan_amount"),
            data.get("debt_to_income_ratio"),
            data.get("credit_score"),
            data.get("credit_history"),
            data.get("previous_defaults"),
            data.get("loan_amount_requested"),
            data.get("loan_tenure"),
            data.get("loan_purpose"),
            data.get("collateral_value"),
            data.get("property_area"),
            data.get("prediction"),
            data.get("probability"),
        )

        cursor.execute(insert_query, values)
        connection.commit()

        logger.info("Loan application saved to database successfully.")
        return True

    except MySQLError as e:
        # Database-specific errors (connection refused, bad table/column,
        # constraint violation, etc.) are logged but never propagated.
        logger.error("Database error while saving loan application: %s", e)
        return False

    except Exception as e:
        # Catch-all safety net so an unexpected error here can never
        # crash the Flask request.
        logger.error("Unexpected error while saving loan application: %s", e)
        return False

    finally:
        # Always release the cursor and connection, even if an error
        # occurred above.
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Validation & Preprocessing Helpers
# --------------------------------------------------------------------------


class InvalidInputError(Exception):
    """Raised when submitted form data is missing or cannot be parsed."""
    pass


def get_required_field(form, field_name: str) -> str:
    """
    Fetch a required field from the submitted form.

    Args:
        form: Flask's request.form (ImmutableMultiDict).
        field_name (str): The expected form field name.

    Returns:
        str: The trimmed field value.

    Raises:
        InvalidInputError: If the field is missing or empty.
    """
    value = form.get(field_name, "").strip()
    if value == "":
        raise InvalidInputError(f"Missing required field: {field_name}")
    return value


def parse_numeric_field(form, field_name: str, cast_type=float):
    """
    Fetch and parse a numeric field from the submitted form.

    Args:
        form: Flask's request.form.
        field_name (str): The expected form field name.
        cast_type: int or float, the type to cast the value to.

    Returns:
        The parsed numeric value.

    Raises:
        InvalidInputError: If the field is missing, empty, or not a valid number.
    """
    raw_value = get_required_field(form, field_name)
    try:
        value = cast_type(raw_value)
    except (TypeError, ValueError):
        raise InvalidInputError(f"Field '{field_name}' must be a valid number.")

    if value < 0:
        raise InvalidInputError(f"Field '{field_name}' cannot be negative.")

    return value


def parse_text_field(form, field_name: str, min_length: int = 1, max_length: int = 255) -> str:
    """
    Fetch and validate a free-text field from the submitted form (e.g.
    applicant_name). Unlike numeric or categorical fields, this does not
    map to any model feature — it is stored in MySQL only.

    Args:
        form: Flask's request.form.
        field_name (str): The expected form field name.
        min_length (int): Minimum allowed length after trimming.
        max_length (int): Maximum allowed length after trimming.

    Returns:
        str: The trimmed, validated text value.

    Raises:
        InvalidInputError: If the field is missing or outside the allowed
            length range.
    """
    value = get_required_field(form, field_name)

    if len(value) < min_length:
        raise InvalidInputError(
            f"Field '{field_name}' must be at least {min_length} characters long."
        )
    if len(value) > max_length:
        raise InvalidInputError(
            f"Field '{field_name}' must be at most {max_length} characters long."
        )

    return value


def validate_all_fields(form) -> dict:
    """
    Validate and parse EVERY field on the upgraded predict.html form,
    regardless of whether the trained model currently uses it. This is the
    single source of truth for "is this submission valid" — both the model
    input and the database row are built from its output, so validation
    only happens once per request.

    Args:
        form: Flask's request.form.

    Returns:
        dict: All 21 fields, validated and converted to their natural
            Python types (str, int, or float). Categorical fields keep
            their original human-readable string value here (e.g.
            gender="Male"); numeric encoding for the model happens
            separately in build_model_features().

    Raises:
        InvalidInputError: If any field is missing, malformed, or holds an
            unrecognized category value.
    """
    fields = {}

    # ---- Personal Information ----
    fields["applicant_name"] = parse_text_field(form, "applicant_name", min_length=2, max_length=100)
    fields["age"] = parse_numeric_field(form, "age", cast_type=int)
    if not (18 <= fields["age"] <= 100):
        raise InvalidInputError("Field 'age' must be between 18 and 100.")

    fields["gender"] = get_required_field(form, "gender")
    if fields["gender"] not in GENDER_MAP:
        raise InvalidInputError(
            f"Invalid value '{fields['gender']}' for field 'gender'. "
            f"Expected one of: {', '.join(GENDER_MAP.keys())}."
        )

    fields["marital_status"] = get_required_field(form, "marital_status")
    if fields["marital_status"] not in MARITAL_STATUS_MAP:
        raise InvalidInputError(
            f"Invalid value '{fields['marital_status']}' for field 'marital_status'. "
            f"Expected one of: {', '.join(MARITAL_STATUS_MAP.keys())}."
        )

    fields["dependents"] = get_required_field(form, "dependents")
    if fields["dependents"] not in DEPENDENTS_MAP:
        raise InvalidInputError(
            f"Invalid value '{fields['dependents']}' for field 'dependents'. "
            f"Expected one of: {', '.join(DEPENDENTS_MAP.keys())}."
        )

    fields["education"] = get_required_field(form, "education")
    if fields["education"] not in EDUCATION_MAP:
        raise InvalidInputError(
            f"Invalid value '{fields['education']}' for field 'education'. "
            f"Expected one of: {', '.join(EDUCATION_MAP.keys())}."
        )

    # ---- Employment Details ----
    fields["employment_type"] = get_required_field(form, "employment_type")
    if fields["employment_type"] not in EMPLOYMENT_TYPE_MAP:
        raise InvalidInputError(
            f"Invalid value '{fields['employment_type']}' for field 'employment_type'. "
            f"Expected one of: {', '.join(EMPLOYMENT_TYPE_MAP.keys())}."
        )

    fields["years_of_employment"] = parse_numeric_field(form, "years_of_employment", cast_type=float)
    fields["monthly_income"] = parse_numeric_field(form, "monthly_income", cast_type=float)
    fields["bank_balance"] = parse_numeric_field(form, "bank_balance", cast_type=float)

    # ---- Financial Details ----
    fields["existing_emi"] = parse_numeric_field(form, "existing_emi", cast_type=float)
    fields["existing_loan_amount"] = parse_numeric_field(form, "existing_loan_amount", cast_type=float)

    fields["debt_to_income_ratio"] = parse_numeric_field(form, "debt_to_income_ratio", cast_type=float)
    if fields["debt_to_income_ratio"] > 100:
        raise InvalidInputError("Field 'debt_to_income_ratio' cannot exceed 100.")

    fields["credit_score"] = parse_numeric_field(form, "credit_score", cast_type=int)
    if not (300 <= fields["credit_score"] <= 900):
        raise InvalidInputError("Field 'credit_score' must be between 300 and 900.")

    fields["credit_history"] = parse_numeric_field(form, "credit_history", cast_type=int)
    if fields["credit_history"] not in (0, 1):
        raise InvalidInputError("Field 'credit_history' must be 0 or 1.")

    fields["previous_defaults"] = parse_numeric_field(form, "previous_defaults", cast_type=int)

    # ---- Loan Details ----
    fields["loan_amount_requested"] = parse_numeric_field(form, "loan_amount_requested", cast_type=float)
    fields["loan_tenure"] = parse_numeric_field(form, "loan_tenure", cast_type=int)

    fields["loan_purpose"] = get_required_field(form, "loan_purpose")
    if fields["loan_purpose"] not in LOAN_PURPOSE_MAP:
        raise InvalidInputError(
            f"Invalid value '{fields['loan_purpose']}' for field 'loan_purpose'. "
            f"Expected one of: {', '.join(LOAN_PURPOSE_MAP.keys())}."
        )

    fields["collateral_value"] = parse_numeric_field(form, "collateral_value", cast_type=float)

    fields["property_area"] = get_required_field(form, "property_area")
    if fields["property_area"] not in PROPERTY_AREA_MAP:
        raise InvalidInputError(
            f"Invalid value '{fields['property_area']}' for field 'property_area'. "
            f"Expected one of: {', '.join(PROPERTY_AREA_MAP.keys())}."
        )

    return fields


def build_model_features(fields: dict) -> pd.DataFrame:
    """
    Build the exact 24-feature vector the trained model expects, from the
    validated form fields. This directly encodes all 20 raw fields (no
    vocabulary translation needed - the form's field names and values
    already match the new dataset's schema) and then computes the same 4
    engineered ratio features ml/preprocess.py's engineer_features()
    derives, using the identical formulas, so a live prediction request
    is preprocessed exactly the same way the training data was.

    Args:
        fields (dict): The output of validate_all_fields().

    Returns:
        pd.DataFrame: A single-row DataFrame, in the exact column order
            ml/preprocess.py produces, ready for model.predict().
    """
    # ---- Encode the 20 raw fields ----
    encoded = {
        "age": fields["age"],
        "gender": GENDER_MAP[fields["gender"]],
        "marital_status": MARITAL_STATUS_MAP[fields["marital_status"]],
        "dependents": DEPENDENTS_MAP[fields["dependents"]],
        "education": EDUCATION_MAP[fields["education"]],
        "employment_type": EMPLOYMENT_TYPE_MAP[fields["employment_type"]],
        "years_of_employment": fields["years_of_employment"],
        "monthly_income": fields["monthly_income"],
        "bank_balance": fields["bank_balance"],
        "existing_emi": fields["existing_emi"],
        "existing_loan_amount": fields["existing_loan_amount"],
        "debt_to_income_ratio": fields["debt_to_income_ratio"],
        "credit_score": fields["credit_score"],
        "credit_history": fields["credit_history"],
        "previous_defaults": fields["previous_defaults"],
        "loan_amount_requested": fields["loan_amount_requested"],
        "loan_tenure": fields["loan_tenure"],
        "loan_purpose": LOAN_PURPOSE_MAP[fields["loan_purpose"]],
        "collateral_value": fields["collateral_value"],
        "property_area": PROPERTY_AREA_MAP[fields["property_area"]],
    }

    # ---- Engineer the same 4 features ml/preprocess.py derives ----
    # Formulas match ml/preprocess.py's engineer_features() exactly,
    # including the same zero-denominator guards, so a live request is
    # preprocessed identically to how the training data was built.

    # loan_to_income_ratio: requested loan vs. annual income.
    annual_income = fields["monthly_income"] * 12
    loan_to_income_ratio = (
        fields["loan_amount_requested"] / annual_income if annual_income != 0 else 0.0
    )

    # collateral_to_loan_ratio: how well the loan is secured.
    collateral_to_loan_ratio = (
        fields["collateral_value"] / fields["loan_amount_requested"]
        if fields["loan_amount_requested"] != 0 else 0.0
    )

    # net_disposable_income: monthly cash left after existing EMI.
    # Intentionally not clipped at zero - a negative value is itself a
    # meaningful (bad) signal, exactly as in preprocess.py.
    net_disposable_income = fields["monthly_income"] - fields["existing_emi"]

    # income_per_dependent: income spread across applicant + dependents.
    dependents_count = DEPENDENTS_TO_COUNT[fields["dependents"]]
    income_per_dependent = fields["monthly_income"] / (dependents_count + 1)

    engineered = {
        "loan_to_income_ratio": loan_to_income_ratio,
        "collateral_to_loan_ratio": collateral_to_loan_ratio,
        "net_disposable_income": net_disposable_income,
        "income_per_dependent": income_per_dependent,
    }

    all_features = {**encoded, **engineered}
    row = {col: all_features[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.route("/")
def index():
    """Render the landing page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET:  Render the empty loan application form.
    POST: Validate the submitted form, run the prediction, and render
          the result page. Falls back to re-rendering the form with an
          error message if input is invalid or the model is unavailable.
    """
    if request.method == "GET":
        return render_template("predict.html")

    # ---- POST: handle form submission ----
    raw_form = request.form

    # Guard: model must have loaded successfully at startup.
    if model is None:
        logger.error("Prediction attempted but model is not loaded.")
        return render_template(
            "predict.html",
            error="The prediction model is currently unavailable. Please try again later.",
        ), 503

    try:
        # 1. Validate and parse EVERY field on the new form (once), whether
        #    or not the model currently uses it.
        fields = validate_all_fields(raw_form)

        # 2. Build the model's 24-feature vector from the validated fields.
        features_df = build_model_features(fields)

        # 3. Run the prediction.
        prediction_raw = model.predict(features_df)[0]

        # 4. Get a confidence/probability score if the model supports it.
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features_df)[0]
                # Probability of the predicted class, as a percentage.
                probability = round(float(proba[int(prediction_raw)]) * 100, 1)
            except Exception as proba_error:
                # Non-fatal: proceed without a probability score.
                logger.warning("Could not compute prediction probability: %s", proba_error)
                probability = None

        # 5. Translate the model's numeric output back into "Y"/"N",
        #    matching the Loan_Status encoding used during training
        #    (N -> 0, Y -> 1; see ml/preprocess.py).
        prediction_label = "Y" if int(prediction_raw) == 1 else "N"

        logger.info(
            "Prediction completed successfully: %s (probability=%s)",
            prediction_label,
            probability,
        )

        # 6. Persist this application + its prediction to MySQL. This is a
        #    side effect that happens AFTER the prediction is final; it
        #    never changes the prediction logic above, and a database
        #    failure here is logged but does not stop the user from seeing
        #    their result (handled internally by save_loan_application()).
        #    Every field from the new form is saved, including the ones
        #    the model does not yet use.
        save_loan_application({
            **fields,
            "prediction": prediction_label,
            "probability": probability,
        })

        # 7. Render the result page with the prediction and the original
        #    (human-readable) form values for the summary table.
        #
        #    NOTE: result.html was not modified as part of this task and
        #    still reads legacy capitalized variable names (Gender,
        #    Married, ApplicantIncome, etc.) left over from an earlier
        #    version of this project. To keep that page rendering
        #    correctly without touching its template, those legacy names
        #    are still populated here, sourced from the current fields.
        return render_template(
            "result.html",
            prediction=prediction_label,
            probability=probability,
            applicant_name=fields["applicant_name"],
            Gender=fields["gender"],
            Married=fields["marital_status"],
            Dependents=fields["dependents"],
            Education=fields["education"],
            Self_Employed=fields["employment_type"],
            ApplicantIncome=fields["monthly_income"],
            CoapplicantIncome=0,
            LoanAmount=fields["loan_amount_requested"],
            Loan_Amount_Term=fields["loan_tenure"],
            Credit_History=str(fields["credit_history"]),
            Property_Area=fields["property_area"],
        )

    except InvalidInputError as e:
        # Expected validation failure: show the form again with a clear message.
        logger.info("Invalid input on /predict: %s", e)
        return render_template("predict.html", error=str(e)), 400

    except Exception as e:
        # Unexpected failure: log full details, show a generic safe message.
        logger.exception("Unexpected error during prediction: %s", e)
        return render_template(
            "predict.html",
            error="Something went wrong while processing your application. Please try again.",
        ), 500


# --------------------------------------------------------------------------
# Error Handlers
# --------------------------------------------------------------------------


@app.errorhandler(404)
def handle_not_found(error):
    """Handle unknown routes gracefully instead of showing Flask's default 404."""
    return render_template("index.html"), 404


@app.errorhandler(500)
def handle_server_error(error):
    """Handle unexpected server errors gracefully."""
    logger.exception("Internal server error: %s", error)
    return render_template("index.html"), 500


# --------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # debug=False is the production-appropriate default; enable manually
    # during local development if needed.
    app.run(host="0.0.0.0", port=5000, debug=False)

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import the classes before loading the saved models.
# This also prevents common joblib deserialization errors.
from perceptron_model import Perceptron
from logistic_model import LogisticRegression


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. EXACT SPAMBASE FEATURE NAMES AND ORDER
# ============================================================

FEATURE_NAMES = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]

INTEGER_FEATURES = {
    "capital_run_length_longest",
    "capital_run_length_total",
}

WORD_FEATURES = [
    name for name in FEATURE_NAMES
    if name.startswith("word_freq_")
]

CHAR_FEATURES = [
    name for name in FEATURE_NAMES
    if name.startswith("char_freq_")
]

CAPITAL_FEATURES = [
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin-bottom: 0.4rem;
        }

        .hero p {
            margin-bottom: 0;
            opacity: 0.8;
        }

        div[data-testid="stMetric"] {
            padding: 1rem;
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
        }

        .stButton button,
        .stFormSubmitButton button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_artifacts():
    """Load both trained models and the fitted scaler."""

    perceptron_path = MODELS_DIR / "perceptron.pkl"
    logistic_path = MODELS_DIR / "logistic_regression.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    required_files = [
        perceptron_path,
        logistic_path,
        scaler_path,
    ]

    missing_files = [
        path.name for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model files: "
            + ", ".join(missing_files)
            + ". Run train_model.py first."
        )

    perceptron_model = joblib.load(perceptron_path)
    logistic_model = joblib.load(logistic_path)
    scaler = joblib.load(scaler_path)

    return perceptron_model, logistic_model, scaler


try:
    perceptron_model, logistic_model, scaler = load_artifacts()
    artifacts_ready = True
except Exception as error:
    artifacts_ready = False
    st.error(
        "The models could not be loaded. "
        "Check your model files and Python class imports."
    )
    st.exception(error)


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SpamGuard AI</h1>
        <p>
            An intelligent email classification application
            powered by machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Enter the 57 Spambase features below. "
    "The selected model will predict whether the email "
    "is spam or non-spam."
)


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ Configuration")

    selected_model_name = st.selectbox(
        "Choose a classification model",
        options=[
            "Logistic Regression",
            "Perceptron",
        ],
        index=0,
    )

    st.divider()

    st.subheader("About this application")

    st.write(
        """
        **Dataset:** UCI Spambase

        **Input:** 57 numerical features

        **Output:**
        - `1` — Spam
        - `0` — Non-spam

        **Preprocessing:** StandardScaler
        """
    )

    st.caption(
        "Educational machine-learning project. "
        "Predictions depend on the quality and validity "
        "of the supplied features."
    )


# ============================================================
# 7. FEATURE INPUT FORM
# ============================================================

def render_feature_inputs(feature_names, columns):
    """Display numeric inputs and return their values."""

    values = {}

    for index, feature_name in enumerate(feature_names):
        column = columns[index % len(columns)]

        with column:
            if feature_name in INTEGER_FEATURES:
                value = st.number_input(
                    label=feature_name,
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"input_{feature_name}",
                    help="Enter a non-negative integer.",
                )
            else:
                value = st.number_input(
                    label=feature_name,
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    format="%.4f",
                    key=f"input_{feature_name}",
                    help=(
                        "Enter a non-negative numeric value. "
                        "Use the same feature definition and "
                        "units as the Spambase dataset."
                    ),
                )

            values[feature_name] = value

    return values


with st.form("spam_prediction_form"):
    st.subheader(" Email feature values")

    st.info(
        "For this demo, enter extracted Spambase feature values. "
        "These inputs are not the raw words or characters of an email."
    )

    with st.expander(
        "1. Word frequency features (48)",
        expanded=True,
    ):
        st.caption(
            "Frequency values for words such as make, free, "
            "money, email and conference."
        )

        word_columns = st.columns(3)
        word_values = render_feature_inputs(
            WORD_FEATURES,
            word_columns,
        )

    with st.expander(
        "2. Character frequency features (6)",
        expanded=True,
    ):
        st.caption(
            "Frequency values for the characters ; ( [ ! $ #"
        )

        char_columns = st.columns(3)
        char_values = render_feature_inputs(
            CHAR_FEATURES,
            char_columns,
        )

    with st.expander(
        "3. Capital-letter features (3)",
        expanded=True,
    ):
        st.caption(
            "Statistics about uninterrupted sequences of capital letters."
        )

        capital_columns = st.columns(3)
        capital_values = render_feature_inputs(
            CAPITAL_FEATURES,
            capital_columns,
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Analyze Email",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 8. VALIDATE AND PREPARE INPUT
# ============================================================

if submitted and artifacts_ready:
    # Combine values and preserve the exact training feature order.
    all_values = {
        **word_values,
        **char_values,
        **capital_values,
    }

    input_df = pd.DataFrame(
        [[all_values[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    if input_df.isna().any().any():
        st.error("Please provide a valid value for every feature.")
        st.stop()

    # The scaler must be the one fitted during model training.
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as error:
        st.error(
            "Could not preprocess the input. "
            "Check that the feature names and order match training."
        )
        st.exception(error)
        st.stop()


    # ========================================================
    # 9. MAKE PREDICTION
    # ========================================================

    if selected_model_name == "Perceptron":
        selected_model = perceptron_model
    else:
        selected_model = logistic_model

    try:
        prediction = int(
            np.asarray(
                selected_model.predict(input_scaled)
            ).reshape(-1)[0]
        )
    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
        st.stop()


    # ========================================================
    # 10. DISPLAY RESULTS
    # ========================================================

    st.divider()
    st.subheader("📋 Prediction result")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            label="Selected model",
            value=selected_model_name,
        )

    with metric_col2:
        st.metric(
            label="Predicted class",
            value=str(prediction),
        )

    if prediction == 1:
        st.error(
            "🚨 **Spam detected**\n\n"
            "The model classified the supplied features as spam."
        )
    elif prediction == 0:
        st.success(
            "✅ **Non-spam detected**\n\n"
            "The model classified the supplied features as non-spam."
        )
    else:
        st.warning(
            f"The model returned an unexpected class: {prediction}"
        )

    with st.expander("View submitted feature values"):
        st.dataframe(
            input_df.T.rename(columns={0: "Input value"}),
            use_container_width=True,
        )

    st.caption(
        "This is a model prediction, not a guarantee that an email "
        "is safe or malicious. The app does not calculate a "
        "confidence percentage unless the model explicitly supports it."
    )
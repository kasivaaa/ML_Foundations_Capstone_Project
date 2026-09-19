
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Student Support Predictor",
    page_icon="🎓",
    layout="centered"
)

RANDOM_STATE = 42
TARGET = "G3"

# The ML model uses the same non-leakage predictors as the capstone.
# G1 and G2 are deliberately excluded.
ALL_FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]

NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]

CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    possible_paths = [
        Path("student-mat.csv"),
        Path(__file__).parent / "student-mat.csv",
        Path("data") / "student-mat.csv",
        Path(__file__).parent / "data" / "student-mat.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return pd.read_csv(path, sep=";"), path

    raise FileNotFoundError(
        "student-mat.csv was not found. Place it in the same folder as app.py."
    )

# =========================================================
# TRAIN MODEL
# =========================================================
@st.cache_resource
def train_model(df):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return pipeline, metrics

try:
    df, dataset_path = load_data()
    model, metrics = train_model(df)
except Exception as e:
    st.error(f"Could not start the application: {e}")
    st.stop()

# =========================================================
# SIMPLE INPUT -> FULL MODEL INPUT
# =========================================================
def create_student(
    age,
    studytime,
    failures,
    absences,
    higher,
    schoolsup,
    famsup,
    internet
):
    """
    Creates a complete model input from a small number of
    user-friendly questions.

    The remaining variables are assigned sensible reference
    values based on the most common category/median in the
    training dataset. This keeps the interface simple while
    preserving the same trained model used in the capstone.
    """

    # Start with dataset reference values.
    student = {}

    for col in ALL_FEATURES:
        if col in NUMERIC_FEATURES:
            student[col] = float(df[col].median())
        else:
            student[col] = df[col].mode()[0]

    # Replace only the simple, user-entered variables.
    student.update({
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "higher": higher,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "internet": internet,
    })

    return pd.DataFrame([student], columns=ALL_FEATURES)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎓 Student Support Predictor")
page = st.sidebar.radio(
    "Choose a page",
    ["🏠 Make a Prediction"]
)

# =========================================================
# HOME
# =========================================================
if page == "🏠 Make a Prediction":

    st.title("🎓 Student Support Predictor")

    st.markdown(
        """
        ### Estimate a student's final academic grade

        Answer a few simple questions about the student and the model
        will estimate their **final grade out of 20**.

        
        """
    )

    st.info(
        "💡 This is an early-support tool. It gives an estimate that can "
        "help a teacher decide whether a student may need a closer look. "
        "It does not replace a teacher's judgement."
    )

    st.divider()

    st.subheader("👩‍🎓 Tell us about the student")

    age = st.slider(
        "Student's age",
        min_value=15,
        max_value=22,
        value=17,
        help="Age of the student."
    )

    studytime_label = st.select_slider(
        "How much time does the student spend studying each week?",
        options=[
            "Less than 2 hours",
            "2–5 hours",
            "5–10 hours",
            "More than 10 hours"
        ],
        value="2–5 hours"
    )

    studytime_map = {
        "Less than 2 hours": 1,
        "2–5 hours": 2,
        "5–10 hours": 3,
        "More than 10 hours": 4
    }
    studytime = studytime_map[studytime_label]

    failures = st.selectbox(
        "How many times has the student previously failed a class?",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: "None" if x == 0 else str(x)
    )

    absences = st.number_input(
        "How many days has the student been absent?",
        min_value=0,
        max_value=100,
        value=4,
        step=1,
        help="Approximate number of school absences."
    )

    higher = st.radio(
        "Does the student want to continue to higher education?",
        ["yes", "no"],
        format_func=lambda x: "Yes" if x == "yes" else "No",
        horizontal=True
    )

    schoolsup = st.radio(
        "Does the student receive extra academic support at school?",
        ["yes", "no"],
        format_func=lambda x: "Yes" if x == "yes" else "No",
        horizontal=True
    )

    famsup = st.radio(
        "Does the student receive academic support from their family?",
        ["yes", "no"],
        format_func=lambda x: "Yes" if x == "yes" else "No",
        horizontal=True
    )

    internet = st.radio(
        "Does the student have internet access at home?",
        ["yes", "no"],
        format_func=lambda x: "Yes" if x == "yes" else "No",
        horizontal=True
    )

    st.divider()

    with st.expander("🔎 What does this prediction use?"):
        st.write(
            """
            The project dataset contains many demographic, family, social and
            school-related variables. To keep this application easy to use,
            this interface asks only a small set of practical questions.

            The trained model still uses the full set of non-leakage predictors.
            The additional variables that are not asked here are automatically
            filled using typical reference values from the dataset.
            """
        )

    if st.button(
        "🔮 Estimate Final Grade",
        type="primary",
        use_container_width=True
    ):

        student = create_student(
            age=age,
            studytime=studytime,
            failures=failures,
            absences=absences,
            higher=higher,
            schoolsup=schoolsup,
            famsup=famsup,
            internet=internet
        )

        prediction = float(model.predict(student)[0])
        prediction = float(np.clip(prediction, 0, 20))

        st.divider()
        st.subheader("📚 Estimated Result")

        st.metric(
            "Estimated Final Grade",
            f"{prediction:.1f} / 20"
        )

        # Simple, non-technical interpretation.
        if prediction < 8:
            st.error(
                "⚠️ The estimate is in a low range. "
                "This student may benefit from closer academic support."
            )
            support_message = (
                "Consider reviewing attendance, study habits, previous "
                "academic difficulties and available support with the student."
            )
        elif prediction < 10:
            st.warning(
                "⚠️ The estimate is below 10/20. "
                "This may be an early-support signal."
            )
            support_message = (
                "A teacher may want to monitor the student's progress and "
                "consider whether additional academic support is needed."
            )
        elif prediction < 15:
            st.info(
                "ℹ️ The estimate is in the middle range. "
                "Continued monitoring may be useful."
            )
            support_message = (
                "Encourage consistent study habits and continue monitoring "
                "the student's progress."
            )
        else:
            st.success(
                "✅ The estimate is in a higher range."
            )
            support_message = (
                "Continue supporting the student's current learning habits "
                "and progress."
            )

        st.write("**Suggested next step:**")
        st.write(support_message)

        st.caption(
            "Important: this is a statistical estimate, not a final judgement "
            "of the student's ability, intelligence or future."
        )

# =========================================================
# MODEL PAGE
# =========================================================
elif page == "📊 About the Model":

    st.title("📊 About the Prediction Model")

    st.write(
        """
        This application uses a **Random Forest Regression model**.
        It was selected because it performed best among the tree-based
        models tested in the capstone.
        """
    )

    st.subheader("How accurate is the model?")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Error (MAE)", f"{metrics['MAE']:.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col3.metric("R²", f"{metrics['R2']:.3f}")

    st.markdown(
        f"""
        ### What these numbers mean

        **Average Error (MAE): {metrics['MAE']:.2f} grade points**

        On average, the model's predictions differ from the actual final
        grade by about **{metrics['MAE']:.2f} points**.

        **RMSE: {metrics['RMSE']:.2f} grade points**

        RMSE gives more weight to large mistakes, so it helps us understand
        whether the model sometimes makes particularly large errors.

        **R²: {metrics['R2']:.3f}**

        This measures how much variation in final grades is explained by
        the model on the test data.
        """
    )

    st.warning(
        "The model is designed for decision support, not automatic "
        "student classification or high-stakes decisions."
    )

# =========================================================
# PROJECT PAGE
# =========================================================
else:

    st.title("ℹ️ About the Project")

    st.subheader("Problem")

    st.write(
        """
        Schools may need an early indication of which students could benefit
        from additional academic support. This project uses machine learning
        to estimate a student's final academic grade from information about
        their demographic, family, behavioural and school circumstances.
        """
    )

    st.subheader("Why is the app simple?")

    st.write(
        """
        A machine-learning application should be usable by people who do not
        know how the underlying algorithm works. Instead of asking a user to
        understand technical dataset column names, the application asks a
        small number of simple questions in normal language.
        """
    )

    st.subheader("Why are G1 and G2 excluded?")

    st.write(
        """
        G1 and G2 are previous academic grades. They were analysed during
        exploratory analysis but excluded from the main prediction model
        because they are very directly related to the final grade G3.
        Excluding them makes the project more suitable for early academic
        support.
        """
    )

    st.subheader("How should the prediction be used?")

    st.write(
        """
        The prediction should be treated as an early-support signal. A teacher
        or academic support team should consider the prediction together with
        the student's actual circumstances before taking action.
        """
    )

    st.subheader("Dataset")

    st.write(
        f"The application is using the Student Performance dataset loaded from: `{dataset_path}`."
    )

    st.dataframe(
        df[["school", "age", "studytime", "failures", "absences", "G1", "G2", "G3"]].head(10),
        use_container_width=True
    )

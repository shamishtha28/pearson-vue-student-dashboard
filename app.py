
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(
    page_title="Pearson VUE Student Analytics",
    page_icon="📊",
    layout="wide"
)

FEATURES = [
    "Midterm_Score",
    "Quizzes_Avg",
    "Attendance (%)",
    "Assignments_Avg"
]

CATEGORY_ORDER = [
    "Needs Improvement",
    "Average Performer",
    "High Performer"
]

st.title("📊 Pearson VUE Student Performance Analytics")
st.write(
    "Analyze student performance, compare machine learning models, "
    "and explore academic score patterns."
)


@st.cache_data
def load_data():
    df = pd.read_csv("student_performance_categories.csv")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=FEATURES)

    if "Performance_Category" not in df.columns:
        df["Overall_Score"] = df[FEATURES].mean(axis=1)
        low = df["Overall_Score"].quantile(0.25)
        high = df["Overall_Score"].quantile(0.75)

        def classify(score):
            if score < low:
                return "Needs Improvement"
            elif score < high:
                return "Average Performer"
            return "High Performer"

        df["Performance_Category"] = (
            df["Overall_Score"].apply(classify)
        )

    return df


try:
    df = load_data()
    X = df[FEATURES]
    y = df["Performance_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy (%)": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "Precision (%)": round(
                precision_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                ) * 100, 2
            ),
            "Recall (%)": round(
                recall_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                ) * 100, 2
            ),
            "F1-Score (%)": round(
                f1_score(
                    y_test, predictions,
                    average="macro", zero_division=0
                ) * 100, 2
            )
        })

    results_df = pd.DataFrame(results)

    best_model_name = results_df.loc[
        results_df["Accuracy (%)"].idxmax(), "Model"
    ]

    counts = df["Performance_Category"].value_counts()

    st.subheader("Key Statistics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", len(df))
    c2.metric(
        "Needs Improvement",
        int(counts.get("Needs Improvement", 0))
    )
    c3.metric("Best Model", best_model_name)
    c4.metric(
        "Best Model Accuracy",
        f"{results_df['Accuracy (%)'].max():.2f}%"
    )

    st.subheader("1. Student Performance Category Distribution")

    category_counts = (
        df["Performance_Category"]
        .value_counts()
        .reindex(CATEGORY_ORDER, fill_value=0)
        .rename_axis("Performance Category")
        .reset_index(name="Number of Students")
    )

    fig1 = px.bar(
        category_counts,
        x="Performance Category",
        y="Number of Students",
        color="Performance Category",
        text="Number of Students"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2. Machine Learning Model Comparison")

    fig2 = px.bar(
        results_df,
        x="Model",
        y="Accuracy (%)",
        color="Model",
        text="Accuracy (%)",
        title="Test Accuracy by Model"
    )
    fig2.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. Average Academic Scores by Category")

    averages = (
        df.groupby("Performance_Category")[FEATURES]
        .mean()
        .reindex(CATEGORY_ORDER)
        .reset_index()
        .melt(
            id_vars="Performance_Category",
            var_name="Academic Measure",
            value_name="Average Score"
        )
    )

    fig3 = px.bar(
        averages,
        x="Academic Measure",
        y="Average Score",
        color="Performance_Category",
        barmode="group"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("4. Detailed Model Metrics")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("5. Predict Student Performance")
    st.write("Enter a student's academic scores to get a category.")

    with st.form("prediction_form"):
        a, b = st.columns(2)

        with a:
            midterm = st.slider("Midterm Score", 0, 100, 85)
            quiz = st.slider("Quiz Average", 0, 100, 80)

        with b:
            attendance = st.slider("Attendance (%)", 0, 100, 90)
            assignment = st.slider(
                "Assignment Average", 0, 100, 88
            )

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        student = pd.DataFrame(
            [[midterm, quiz, attendance, assignment]],
            columns=FEATURES
        )

        model_path = Path("best_logistic_regression_model.pkl")

        if model_path.exists():
            prediction_model = joblib.load(model_path)
        else:
            prediction_model = models["Logistic Regression"]

        prediction = prediction_model.predict(student)[0]
        average_score = student[FEATURES].mean(axis=1).iloc[0]

        st.success(f"Predicted Category: {prediction}")
        st.write(f"Average of the four inputs: {average_score:.2f}")

        st.caption(
            "This category is based on the project's constructed labels. "
            "It is not a validated prediction of future academic results."
        )

    with st.expander("View student data"):
        st.dataframe(df, use_container_width=True)

    st.info(
        "Methodology limitation: The categories were created using "
        "quartiles of the same academic features supplied to the models. "
        "The reported metrics measure how well models reproduce these "
        "project-defined labels, not independently verified future "
        "student outcomes."
    )

except Exception as e:
    st.error(
        "The dashboard could not load. Check that the CSV and model "
        "files exist in your GitHub repository and that their names "
        "are correct."
    )
    st.code(str(e))
    from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Pearson VUE Student Analytics",
    page_icon="📊",
    layout="wide"
)

FEATURES = [
    "Midterm_Score",
    "Quizzes_Avg",
    "Attendance (%)",
    "Assignments_Avg"
]

CATEGORY_ORDER = [
    "Needs Improvement",
    "Average Performer",
    "High Performer"
]

st.title("📊 Pearson VUE Student Performance Analytics")
st.write(
    "Analyze student performance, compare machine learning models, "
    "and explore academic score patterns."
)

# Load the dataset uploaded to GitHub
@st.cache_data
def load_data():
    df = pd.read_csv("student_performance_categories.csv")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=FEATURES)

    # Recreate project-defined categories only if labels are absent
    if "Performance_Category" not in df.columns:
        df["Overall_Score"] = df[FEATURES].mean(axis=1)
        low = df["Overall_Score"].quantile(0.25)
        high = df["Overall_Score"].quantile(0.75)

        def classify(score):
            if score < low:
                return "Needs Improvement"
            elif score < high:
                return "Average Performer"
            return "High Performer"

        df["Performance_Category"] = (
            df["Overall_Score"].apply(classify)
        )

    return df


try:
    df = load_data()

    X = df[FEATURES]
    y = df["Performance_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy (%)": round(
                accuracy_score(y_test, predictions) * 100, 2
            )
        })

    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[
        results_df["Accuracy (%)"].idxmax(), "Model"
    ]

    # Summary statistics
    counts = df["Performance_Category"].value_counts()

    st.subheader("Key Statistics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", len(df))
    c2.metric(
        "Needs Improvement",
        int(counts.get("Needs Improvement", 0))
    )
    c3.metric("Best Model", best_model_name)
    c4.metric(
        "Best Model Accuracy",
        f"{results_df['Accuracy (%)'].max():.2f}%"
    )

    # Graph 1
    st.subheader("1. Student Performance Category Distribution")

    category_counts = (
        df["Performance_Category"]
        .value_counts()
        .reindex(CATEGORY_ORDER, fill_value=0)
        .rename_axis("Performance Category")
        .reset_index(name="Number of Students")
    )

    fig1 = px.bar(
        category_counts,
        x="Performance Category",
        y="Number of Students",
        color="Performance Category",
        text="Number of Students"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Graph 2
    st.subheader("2. Machine Learning Model Comparison")

    fig2 = px.bar(
        results_df,
        x="Model",
        y="Accuracy (%)",
        color="Model",
        text="Accuracy (%)",
        title="Test Accuracy by Model"
    )
    fig2.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

    # Graph 3
    st.subheader("3. Average Academic Scores by Category")

    averages = (
        df.groupby("Performance_Category")[FEATURES]
        .mean()
        .reindex(CATEGORY_ORDER)
        .reset_index()
        .melt(
            id_vars="Performance_Category",
            var_name="Academic Measure",
            value_name="Average Score"
        )
    )

    fig3 = px.bar(
        averages,
        x="Academic Measure",
        y="Average Score",
        color="Performance_Category",
        barmode="group"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Student prediction tool
    st.subheader("4. Predict Student Performance")
    st.write("Enter a student's academic scores to get a prediction.")

    with st.form("prediction_form"):
        a, b = st.columns(2)

        with a:
            midterm = st.slider("Midterm Score", 0, 100, 85)
            quiz = st.slider("Quiz Average", 0, 100, 80)

        with b:
            attendance = st.slider("Attendance (%)", 0, 100, 90)
            assignment = st.slider("Assignment Average", 0, 100, 88)

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        student = pd.DataFrame([[
            midterm, quiz, attendance, assignment
        ]], columns=FEATURES)

        model = models["Logistic Regression"]
        prediction = model.predict(student)[0]
        average_score = student[FEATURES].mean(axis=1).iloc[0]

        st.success(f"Predicted Category: {prediction}")
        st.write(f"Average of the four inputs: {average_score:.2f}")

    # Show model results
    with st.expander("View model comparison table"):
        st.dataframe(results_df, use_container_width=True)

    st.info(
        "Methodology note: These categories are project-defined using "
        "quartiles of the selected academic features. Model accuracy "
        "measures reproduction of these constructed labels, not "
        "validated real-world Pearson VUE outcomes."
    )

except Exception as e:
    st.error(
        "The dashboard could not load. Check that "
        "'student_performance_categories.csv' exists in your GitHub "
        "repository and contains the required columns."
    )
    st.code(str(e))

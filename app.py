"""Streamlit user interface for the SentimentScope AI dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import analyse_dataset
from src.predict import load_model, predict_text, summarize_predictions
from src.preprocessing import preprocess_text
from src.report_generator import generate_insights_report
from train_model import train_and_evaluate_model

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_reviews.csv"
METRICS_PATH = ROOT_DIR / "reports" / "model_metrics.json"


def ensure_model() -> object:
    """Ensure the locally trained model exists before serving the app."""
    if not MODEL_PATH.exists():
        st.info("Training a fresh model from the sample dataset to get the app ready...")
        train_and_evaluate_model(SAMPLE_DATA_PATH, MODEL_PATH, METRICS_PATH)
    return load_model(MODEL_PATH)


st.set_page_config(page_title="SentimentScope AI", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, #132238 0%, #0b1220 35%, #070c14 100%);
            color: #e6edf7;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .section-title {
            font-size: 2rem;
            font-weight: 800;
            color: #f3f7ff;
            margin: 0.2rem 0 0.1rem 0;
            letter-spacing: -0.03em;
        }
        .section-subtitle {
            color: #a9bbd4;
            margin-bottom: 1rem;
            font-size: 1rem;
        }
        .panel {
            background: rgba(17, 25, 38, 0.9);
            border: 1px solid rgba(146, 168, 196, 0.18);
            border-radius: 18px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 12px 28px rgba(3, 7, 18, 0.35);
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(25, 41, 62, 0.95) 0%, rgba(18, 31, 47, 0.95) 100%);
            border: 1px solid rgba(103, 135, 190, 0.28);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 20px rgba(11, 17, 28, 0.26);
            min-height: 110px;
        }
        .small-label {
            color: #8ca9d5;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }
        .feature-box {
            background: linear-gradient(135deg, rgba(32, 54, 81, 0.9) 0%, rgba(19, 29, 43, 0.9) 100%);
            border-radius: 14px;
            border-left: 4px solid #65a7ff;
            padding: 1rem;
            height: 100%;
            color: #edf5ff;
            box-shadow: inset 0 0 0 1px rgba(130, 168, 224, 0.15);
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1726 0%, #0b1320 100%);
        }
        .stButton > button {
            background: linear-gradient(180deg, #4f8ef7 0%, #2d6fe6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.65rem 1rem;
            box-shadow: 0 8px 18px rgba(47, 108, 227, 0.4);
        }
        .stButton > button:hover {
            background: linear-gradient(180deg, #5f9bff 0%, #2f73ec 100%);
            filter: brightness(1.04);
        }
        .stDataFrame, .stTable {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(147, 176, 220, 0.15);
        }
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #5ea4ff 0%, #7dc2ff 100%);
        }
        .css-1d391kg, .css-1v0mbdj, .css-17lntkn {
            color: white;
        }
        h1, h2, h3, h4, p, li, div, span {
            color: #edf5ff;
        }
        .stAlert {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-title'>SentimentScope AI</div>", unsafe_allow_html=True)
st.caption("Sentiment Analysis and Customer Insights Dashboard")

model = ensure_model()

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = pd.DataFrame()

if "analysis_meta" not in st.session_state:
    st.session_state.analysis_meta = {}


def reset_analysis():
    st.session_state.analysis_df = pd.DataFrame()
    st.session_state.analysis_meta = {}


with st.sidebar:
    st.title("Navigation")
    st.caption("SentimentScope AI")
    page = st.radio(
        "Go to",
        [
            "Home / Overview",
            "Analyse Single Text",
            "Upload and Analyse Dataset",
            "Sentiment Dashboard",
            "Model Performance",
            "Insights Report",
            "About and Responsible AI",
        ],
    )
    st.markdown("---")
    st.caption("Synthetic demo data only. Model results should be reviewed by a human analyst.")

if page == "Home / Overview":
    st.markdown("<div class='section-subtitle'>A clear, practical ML workflow for analysing customer sentiment.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><div class='small-label'>Model</div><div style='font-size:1.5rem;font-weight:700;'>TF-IDF + Logistic Regression</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='small-label'>Classes</div><div style='font-size:1.5rem;font-weight:700;'>Positive / Neutral / Negative</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='small-label'>Purpose</div><div style='font-size:1.5rem;font-weight:700;'>Customer insight generation</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Workflow</div>", unsafe_allow_html=True)
    steps = [
        "Enter a review or upload a CSV dataset.",
        "Clean and standardise the text.",
        "Predict sentiment using the trained model.",
        "Review dashboard metrics and downloadable outputs.",
    ]
    cols = st.columns(2)
    for idx, step in enumerate(steps):
        col = cols[idx % 2]
        with col:
            st.markdown(f"<div class='feature-box'><strong>{idx + 1}.</strong> {step}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Analyse Single Text":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Analyse a single review</div>", unsafe_allow_html=True)
    text_input = st.text_area(
        "Enter a product review, customer comment, or message:",
        height=160,
        placeholder="Example: I love this product and the customer service was excellent.",
    )
    if st.button("Analyse Sentiment"):
        if not preprocess_text(text_input):
            st.warning("Please enter a valid review before analysing.")
        else:
            try:
                result = predict_text(text_input, model=model)
                label = result["label"].title()
                confidence = result["confidence"]
                st.success(f"Predicted sentiment: {label}")
                st.metric("Confidence score", f"{confidence:.2f}")
                st.info(
                    "This prediction is based on patterns learned from labelled training data and should be treated as an informed signal rather than a fact about a person's intention."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Upload and Analyse Dataset":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Upload and analyse a CSV dataset</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Dataset preview")
            st.dataframe(df.head(10), use_container_width=True)

            columns = list(df.columns)
            text_column = st.selectbox("Select the text column", columns)
            date_column = st.selectbox("Select an optional date column", ["None"] + columns)
            source_column = st.selectbox("Select an optional source/category column", ["None"] + columns)

            if st.button("Run analysis"):
                try:
                    date_value = None if date_column == "None" else date_column
                    source_value = None if source_column == "None" else source_column
                    analysis = analyse_dataset(df, text_column=text_column, model=model, date_column=date_value, source_column=source_value)
                    result_df = analysis["dataframe"]
                    st.session_state.analysis_df = result_df
                    st.session_state.analysis_meta = {
                        "text_column": text_column,
                        "date_column": date_value,
                        "source_column": source_value,
                        "processed": analysis["processed"],
                        "skipped": analysis["skipped"],
                        "invalid": analysis["invalid"],
                    }

                    st.success(
                        f"Analysis complete: {analysis['processed']} records processed, {analysis['skipped']} skipped, {analysis['invalid']} invalid."
                    )
                    st.dataframe(result_df.head(10), use_container_width=True)
                    st.download_button(
                        label="Download analysed results as CSV",
                        data=result_df.to_csv(index=False),
                        file_name="analysed_results.csv",
                        mime="text/csv",
                    )
                except ValueError as exc:
                    st.error(str(exc))
        except Exception as exc:
            st.error(f"The CSV file could not be read: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Sentiment Dashboard":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Sentiment dashboard</div>", unsafe_allow_html=True)
    if st.session_state.analysis_df.empty:
        st.info("Upload a dataset and run analysis to populate the dashboard.")
    else:
        result_df = st.session_state.analysis_df.copy()
        summary = summarize_predictions(result_df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total analysed", summary["total"])
        with col2:
            st.metric("Positive", summary["positive"])
        with col3:
            st.metric("Negative", summary["negative"])
        with col4:
            st.metric("Neutral", summary["neutral"])

        sentiment_counts = result_df["predicted_sentiment"].fillna("neutral").str.lower().value_counts().reindex(["positive", "negative", "neutral"], fill_value=0)
        bar_chart = px.bar(
            sentiment_counts.reset_index().rename(columns={"index": "Sentiment", "predicted_sentiment": "Count"}),
            x="Sentiment",
            y="Count",
            color="Sentiment",
            title="Sentiment distribution",
            color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"},
        )
        pie_chart = px.pie(
            sentiment_counts.reset_index().rename(columns={"index": "Sentiment", "predicted_sentiment": "Count"}),
            values="Count",
            names="Sentiment",
            title="Sentiment share",
            color="Sentiment",
            color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"},
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(bar_chart, use_container_width=True)
        with chart_right:
            st.plotly_chart(pie_chart, use_container_width=True)

        filter_values = st.multiselect("Filter sentiment", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
        search_term = st.text_input("Search comments")
        filtered_df = result_df[result_df["predicted_sentiment"].str.lower().isin(filter_values)].copy()
        if search_term:
            filtered_df = filtered_df[filtered_df["text"].str.lower().str.contains(search_term.lower(), na=False)]

        st.dataframe(filtered_df, use_container_width=True)

        if st.button("Clear analysis"):
            reset_analysis()

        if "date" in result_df.columns:
            try:
                trend = pd.to_datetime(result_df["date"], errors="coerce").dropna().to_frame(name="date")
                trend["predicted_sentiment"] = result_df.loc[trend.index, "predicted_sentiment"]
                trend_group = (
                    trend.groupby([trend["date"].dt.to_period("M").astype(str), "predicted_sentiment"])
                    .size()
                    .reset_index(name="count")
                )
                trend_group = trend_group.rename(columns={"date": "Month"})
                trend_chart = px.line(trend_group, x="Month", y="count", color="predicted_sentiment", title="Sentiment trends over time")
                st.plotly_chart(trend_chart, use_container_width=True)
            except Exception:
                st.info("No valid date information was available for trend analysis.")
        else:
            st.info("No valid date information was available for trend analysis.")

        if "source" in result_df.columns:
            source_breakdown = result_df.groupby("source")["predicted_sentiment"].value_counts().unstack(fill_value=0)
            st.write("### Sentiment by source")
            st.dataframe(source_breakdown, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Performance":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Model performance</div>", unsafe_allow_html=True)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with col4:
            st.metric("F1-score", f"{metrics.get('f1', 0):.4f}")
        st.text(metrics.get("classification_report", "Classification report unavailable."))
        confusion = pd.DataFrame(
            metrics.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            index=["positive", "negative", "neutral"],
            columns=["positive", "negative", "neutral"],
        )
        st.dataframe(confusion)
    else:
        st.info("No metrics file is present yet. Train the model first to generate performance details.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Insights Report":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Insights report</div>", unsafe_allow_html=True)
    if st.session_state.analysis_df.empty:
        st.info("Generate a dataset analysis before producing a report.")
    else:
        report = generate_insights_report(st.session_state.analysis_df)
        st.text_area("Generated markdown report", report, height=450)
        st.download_button(
            label="Download insights report as Markdown",
            data=report,
            file_name="sentimentscope_insights_report.md",
            mime="text/markdown",
        )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About and Responsible AI":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>About SentimentScope AI</div>", unsafe_allow_html=True)
    st.markdown(
        """
        Sentiment analysis is a text classification task that tries to determine whether a message is mostly positive, negative, or neutral.
        The model learns patterns from labelled examples, such as words and phrases that often appear in positive or negative comments.
        """
    )

    st.markdown("### Responsible AI considerations")
    st.markdown(
        """
        - Predictions are not perfect and can be wrong when sarcasm, slang, spelling errors, mixed emotions, or cultural language differences are present.
        - A sentiment model estimates the likely tone in a text; it does not prove a person's intentions or feelings with certainty.
        - Human review remains important before making business or customer-service decisions based on the output.
        - The quality and balance of the training data strongly influence performance. Poor or biased training data can lead to weak results.
        """
    )

    st.markdown("### How the model works")
    st.markdown(
        """
        The model converts text into numerical features using TF-IDF and then applies logistic regression to classify the message.
        TF-IDF gives more importance to words that appear often in a comment but unusually often in the broader dataset, which helps the classifier focus on meaningful vocabulary.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

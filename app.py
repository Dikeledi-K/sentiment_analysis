"""Streamlit interface for the Quality & Returns Intelligence platform."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import analyse_dataset, guess_column, read_csv_file
from src.business_metrics import (
    calculate_product_return_rate,
    flag_products_for_review,
    has_return_info,
    has_unit_totals,
    returned_mask,
)
from src.issue_classifier import classify_issue
from src.predict import load_model, predict_text, summarize_predictions
from src.preprocessing import preprocess_text
from src.report_generator import generate_quality_report

ROOT_DIR = Path(__file__).resolve().parent
PLATFORM_NAME = "Quality & Returns Intelligence"
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "product_quality_demo.csv"
METRICS_PATH = ROOT_DIR / "reports" / "model_metrics.json"


@st.cache_resource(show_spinner="Loading sentiment model...")
def ensure_model() -> object:
    """Load the model once per server process, training it first if no saved model exists."""
    return load_model(MODEL_PATH)


st.set_page_config(page_title=PLATFORM_NAME, page_icon="📊", layout="wide")

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
        div[data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: 0.4rem;
            padding: 0.55rem 0 0.8rem;
            margin: 0 0 1.35rem;
            border-bottom: 1px solid rgba(146, 168, 196, 0.2);
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            gap: 0;
            padding: 0.55rem 0.9rem;
            border: 1px solid rgba(146, 168, 196, 0.14);
            border-radius: 9px;
            background: rgba(17, 25, 38, 0.68);
            transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(58, 115, 186, 0.24);
            border-color: rgba(103, 169, 255, 0.65);
            box-shadow: inset 0 -2px 0 #68a9ff;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: rgba(71, 101, 143, 0.22);
            border-color: rgba(146, 185, 239, 0.4);
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child > div {
            display: none;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:focus-visible) {
            outline: 2px solid #a5cbff;
            outline-offset: 2px;
        }
        div[data-testid="stSidebar"] [data-testid="stMetric"] {
            padding: 0.7rem 0;
            border-top: 1px solid rgba(146, 168, 196, 0.14);
            border-bottom: 1px solid rgba(146, 168, 196, 0.14);
        }
        .workspace-kicker {
            color: #8ca9d5;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
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

st.markdown(f"<div class='section-title'>{PLATFORM_NAME}</div>", unsafe_allow_html=True)
st.caption("Product feedback, defect trends, and return risk")

model = ensure_model()

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = pd.DataFrame()

if "analysis_meta" not in st.session_state:
    st.session_state.analysis_meta = {}


def reset_analysis():
    st.session_state.analysis_df = pd.DataFrame()
    st.session_state.analysis_meta = {}


def analyze_and_store(df: pd.DataFrame, text_column: str, product_column: str | None = None,
                      category_column: str | None = None, date_column: str | None = None) -> dict:
    """Analyze a dataset and retain the result for the dashboard pages."""
    analysis = analyse_dataset(
        df,
        text_column=text_column,
        model=model,
        date_column=date_column,
        category_column=category_column,
        product_column=product_column,
    )
    st.session_state.analysis_df = analysis["dataframe"]
    st.session_state.analysis_meta = {
        "text_column": text_column,
        "product_column": product_column,
        "category_column": category_column,
        "date_column": date_column,
        "processed": analysis["processed"],
        "skipped": analysis["skipped"],
        "invalid": analysis["invalid"],
    }
    return analysis


with st.sidebar:
    st.markdown("<div class='workspace-kicker'>Workspace</div>", unsafe_allow_html=True)
    st.markdown("### Quality operations")
    st.caption("Analysis workspace")
    st.markdown("---")
    # Filled at the end of the script so it reflects an analysis run during this rerun.
    records_metric = st.empty()
    st.caption("Synthetic demo data only. Validate quality signals with a human analyst.")

navigation_options = [
    "Home / Overview",
    "Analyse Single Text",
    "Upload and Analyse Dataset",
    "Quality Dashboard",
    "Model Performance",
    "Insights Report",
    "About and Responsible AI",
]
navigation_labels = {
    "Home / Overview": "⌂  Overview",
    "Analyse Single Text": "✳  Feedback",
    "Upload and Analyse Dataset": "▦  Dataset",
    "Quality Dashboard": "◈  Quality",
    "Model Performance": "▤  Model",
    "Insights Report": "▧  Reports",
    "About and Responsible AI": "ⓘ  About",
}
page = st.radio(
    "Workspace navigation",
    navigation_options,
    format_func=lambda option: navigation_labels[option],
    horizontal=True,
    label_visibility="collapsed",
    key="workspace_navigation",
)

if page == "Home / Overview":
    st.markdown("<div class='section-subtitle'>An operations view of customer-reported defects, returns, and product-quality signals.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><div class='small-label'>Model</div><div style='font-size:1.5rem;font-weight:700;'>TF-IDF + Logistic Regression</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='small-label'>Classes</div><div style='font-size:1.5rem;font-weight:700;'>Positive / Neutral / Negative</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='small-label'>Purpose</div><div style='font-size:1.5rem;font-weight:700;'>Quality & returns intelligence</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Workflow</div>", unsafe_allow_html=True)
    steps = [
        "Load the demo or upload customer and return records.",
        "Classify feedback sentiment and recurring issue themes.",
        "Compare defect signals by product and category.",
        "Track reported returns and download an investigation report.",
    ]
    cols = st.columns(2)
    for idx, step in enumerate(steps):
        col = cols[idx % 2]
        with col:
            st.markdown(f"<div class='feature-box'><strong>{idx + 1}.</strong> {step}</div>", unsafe_allow_html=True)

elif page == "Analyse Single Text":
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Analyse a single review</div>", unsafe_allow_html=True)
    text_input = st.text_area(
        "Enter a product review, customer comment, or message:",
        height=160,
                placeholder="Example: The battery stopped charging after two days and the return process was confusing.",
    )
    if st.button("Analyse feedback"):
        if not preprocess_text(text_input):
            st.warning("Please enter a valid review before analysing.")
        else:
            try:
                result = predict_text(text_input, model=model)
                label = result["label"].title()
                confidence = result["confidence"]
                issue = classify_issue(text_input)
                result_col, issue_col = st.columns(2)
                with result_col:
                    st.metric("Predicted sentiment", label)
                    st.caption(f"Model confidence: {confidence:.2f}")
                with issue_col:
                    st.metric("Likely issue category", issue)
                st.info(
                    "This prediction is based on patterns learned from labelled training data and should be treated as an informed signal rather than a fact about a person's intention."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

elif page == "Upload and Analyse Dataset":
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Dataset Explorer</div>", unsafe_allow_html=True)
    st.caption("Start with the built-in demonstration data, or upload a CSV with customer feedback and optional product, date, and return fields.")
    if st.button("Analyze built-in demo dataset"):
        try:
            demo_df = read_csv_file(SAMPLE_DATA_PATH)
            analysis = analyze_and_store(
                demo_df,
                text_column="text",
                product_column="product_name",
                category_column="category",
                date_column="date",
            )
            st.success(f"Demo analysis complete: {analysis['processed']} records processed.")
        except Exception as exc:
            st.error(f"The demo dataset could not be analysed: {exc}")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = read_csv_file(uploaded_file)
            st.write("### Dataset preview")
            st.dataframe(df.head(10), width="stretch")

            columns = list(df.columns)
            optional_columns = ["None"] + columns

            def default_index(role: str, options: list[str]) -> int:
                guess = guess_column(df, role)
                return options.index(guess) if guess in options else 0

            st.caption("Columns are pre-selected from their names. Check each mapping before running the analysis.")
            text_column = st.selectbox(
                "Feedback text column",
                columns,
                index=default_index("text", columns),
                help="The free-text customer comment to classify, for example review_text.",
            )
            product_column = st.selectbox(
                "Product name column (optional)",
                optional_columns,
                index=default_index("product", optional_columns),
                help="The product each comment is about, for example product_name. Not the return reason.",
            )
            category_column = st.selectbox(
                "Product category column (optional)",
                optional_columns,
                index=default_index("category", optional_columns),
            )
            date_column = st.selectbox(
                "Date column (optional)",
                optional_columns,
                index=default_index("date", optional_columns),
            )

            if st.button("Run analysis"):
                try:
                    date_value = None if date_column == "None" else date_column
                    product_value = None if product_column == "None" else product_column
                    category_value = None if category_column == "None" else category_column
                    analysis = analyze_and_store(
                        df,
                        text_column,
                        product_column=product_value,
                        category_column=category_value,
                        date_column=date_value,
                    )
                    result_df = analysis["dataframe"]

                    st.success(
                        f"Analysis complete: {analysis['processed']} records processed, {analysis['skipped']} skipped because the text was empty."
                    )
                    st.dataframe(result_df.head(10), width="stretch")
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

elif page == "Quality Dashboard":
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Operations overview</div>", unsafe_allow_html=True)
    if st.session_state.analysis_df.empty:
        st.info("Analyze the built-in demo dataset or upload your own data in Dataset Explorer to populate the dashboard.")
    else:
        result_df = st.session_state.analysis_df.copy()
        summary = summarize_predictions(result_df)
        st.caption("Executive summary · based on the currently analysed records")

        negative_share = summary["negative_pct"]
        if has_return_info(result_df):
            returned_cases_label = f"{int(returned_mask(result_df).sum()):,}"
        else:
            returned_cases_label = "Not provided"

        if has_unit_totals(result_df):
            return_rate_label = f"{calculate_product_return_rate(result_df)['percentage']:.2f}%"
            return_rate_help = "Distinct product totals: returned units divided by sold units"
        else:
            return_rate_label = "Not provided"
            return_rate_help = "Add units_sold and units_returned columns to calculate a rate"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Feedback analysed", f"{summary['total']:,}")
        kpi2.metric("Negative feedback", f"{negative_share:.1f}%", help="Share of analysed comments predicted as negative")
        kpi3.metric("Reported return cases", returned_cases_label, help="Rows with return_status returned/refunded or return_requested = Yes; not a unit return rate")
        kpi4.metric("Unit return rate", return_rate_label, help=return_rate_help)

        st.markdown("### Quality signals")
        left, right = st.columns([1.2, 1])
        with left:
            if {"product_name", "issue_category"}.issubset(result_df.columns):
                issue_counts = result_df[
                    ~result_df["issue_category"].fillna("Needs Review").isin(["Other", "Needs Review"])
                ]
                if issue_counts.empty:
                    st.info("No classified defect signals are available in these records.")
                else:
                    pivot = issue_counts.pivot_table(
                        index="issue_category",
                        columns="product_name",
                        values="text",
                        aggfunc="count",
                        fill_value=0,
                    )
                    heatmap = px.imshow(
                        pivot,
                        labels={"x": "Product", "y": "Issue category", "color": "Reports"},
                        color_continuous_scale=["#172536", "#317a86", "#f0b45b"],
                        aspect="auto",
                        title="Defect reports by product",
                    )
                    heatmap.update_layout(margin={"t": 55, "b": 20}, coloraxis_colorbar_title="Reports")
                    st.plotly_chart(heatmap, width="stretch")
            else:
                st.info("Select a product name column when analysing data to show a defect heatmap.")

        with right:
            if "date" in result_df.columns:
                trend = result_df.copy()
                trend["date"] = pd.to_datetime(trend["date"], errors="coerce")
                if has_return_info(trend):
                    trend = trend[returned_mask(trend)]
                    trend_label = "Reported return cases"
                else:
                    trend_label = "Dated feedback records"
                trend = trend.dropna(subset=["date"])
                if trend.empty:
                    st.info("No dated return records are available for a monthly trend.")
                else:
                    trend["month"] = trend["date"].dt.to_period("M").dt.to_timestamp()
                    trend_counts = trend.groupby("month").size().reset_index(name="records")
                    trend_chart = px.line(
                        trend_counts,
                        x="month",
                        y="records",
                        markers=True,
                        title=f"Monthly {trend_label.lower()}",
                        labels={"month": "Month", "records": trend_label},
                    )
                    trend_chart.update_traces(line_color="#f0b45b", line_width=3)
                    trend_chart.update_layout(margin={"t": 55, "b": 20})
                    st.plotly_chart(trend_chart, width="stretch")
            else:
                st.info("Select a date column when analysing data to show a returns trend.")

        st.markdown("### Products to investigate")
        flagged = flag_products_for_review(result_df)
        if flagged:
            st.dataframe(pd.DataFrame(flagged), width="stretch", hide_index=True)
        else:
            st.success("No products currently meet the review thresholds in this dataset.")

        with st.expander("Explore analysed feedback"):
            filter_values = st.multiselect("Sentiment", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
            search_term = st.text_input("Search feedback")
            filtered_df = result_df[result_df["predicted_sentiment"].str.lower().isin(filter_values)].copy()
            if search_term:
                filtered_df = filtered_df[filtered_df["text"].str.lower().str.contains(search_term.lower(), na=False)]
            st.dataframe(filtered_df, width="stretch")

        if st.button("Clear analysis"):
            reset_analysis()
            st.rerun()

elif page == "Model Performance":
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
        if "cv_f1_mean" in metrics:
            st.caption(
                f"Held-out test set: {metrics.get('test_size', '?')} records. "
                f"Cross-validated macro F1 on the training split: {metrics['cv_f1_mean']:.4f} (± {metrics['cv_f1_std']:.4f})."
            )
        st.text(metrics.get("classification_report", "Classification report unavailable."))
        labels = metrics.get("labels", ["positive", "negative", "neutral"])
        confusion = pd.DataFrame(
            metrics.get("confusion_matrix", [[0] * len(labels) for _ in labels]),
            index=[f"actual {label}" for label in labels],
            columns=[f"predicted {label}" for label in labels],
        )
        st.dataframe(confusion)
    else:
        st.info("No metrics file is present yet. Train the model first to generate performance details.")

elif page == "Insights Report":
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Insights report</div>", unsafe_allow_html=True)
    if st.session_state.analysis_df.empty:
        st.info("Generate a dataset analysis before producing a report.")
    else:
        report = generate_quality_report(st.session_state.analysis_df)
        st.text_area("Generated markdown report", report, height=450)
        st.download_button(
            label="Download insights report as Markdown",
            data=report,
            file_name="product_quality_insights_report.md",
            mime="text/markdown",
        )

elif page == "About and Responsible AI":
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>About the Product Quality Platform</div>", unsafe_allow_html=True)
    st.markdown(
        """
        This platform combines text sentiment classification with transparent keyword-based issue categories and operational return metrics. It helps teams organise customer quality signals for investigation.
        """
    )

    st.markdown("### Responsible AI considerations")
    st.markdown(
        """
        - Predictions are not perfect and can be wrong when sarcasm, slang, spelling errors, mixed emotions, or cultural language differences are present.
        - Sentiment and keyword categories are estimates, not proof of product fault or customer intent.
        - Human review remains important before warranty, returns, supplier, or manufacturing decisions.
        - Monthly return charts count dated records and do not represent a unit return rate.
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

records_metric.metric("Records analyzed", f"{len(st.session_state.analysis_df):,}")

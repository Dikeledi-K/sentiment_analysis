"""Streamlit interface for the Quality & Returns Intelligence platform."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
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

OVERVIEW = "Home / Overview"
LOAD = "Upload and Analyse Dataset"
QUALITY = "Quality Dashboard"
REPORT = "Insights Report"
SINGLE = "Analyse Single Text"
MODEL = "Model Performance"
ABOUT = "About and Responsible AI"
NAVIGATION_LABELS = {
    OVERVIEW: "Overview",
    LOAD: "1 · Load data",
    QUALITY: "2 · Quality",
    REPORT: "3 · Report",
    SINGLE: "Try a comment",
    MODEL: "Model",
    ABOUT: "About",
}
EXAMPLE_COMMENTS = {
    "Damaged delivery": "The blender arrived cracked and the box was crushed.",
    "Happy customer": "Works perfectly, easy to clean, and great value for money.",
    "Mixed review": "The sound is good but the case feels a little cheap.",
}


@st.cache_resource(show_spinner="Loading sentiment model...")
def ensure_model() -> object:
    """Load the model once per server process, training it first if no saved model exists."""
    return load_model(MODEL_PATH)


st.set_page_config(page_title=PLATFORM_NAME, page_icon="🛡️", layout="wide")
ui.inject_css()
model = ensure_model()

st.session_state.setdefault("analysis_df", pd.DataFrame())
st.session_state.setdefault("analysis_meta", {})
st.session_state.setdefault("visited", set())
st.session_state.setdefault("workspace_navigation", OVERVIEW)


# ---------------------------------------------------------------------------
# Actions. These run as button callbacks, before the page is drawn, so every
# part of the page (sidebar, stepper, status chip) reflects the new state.
# ---------------------------------------------------------------------------
def go_to(page: str) -> None:
    st.session_state.workspace_navigation = page


def store_analysis(df: pd.DataFrame, source: str, text_column: str, product_column: str | None = None,
                   category_column: str | None = None, date_column: str | None = None) -> None:
    """Analyse a dataset and keep the result for the dashboard and report."""
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
        "source": source,
        "processed": analysis["processed"],
        "skipped": analysis["skipped"],
        "columns": {"Text": text_column, "Product": product_column, "Category": category_column, "Date": date_column},
    }
    st.session_state.flash = f"Analysed {analysis['processed']} records from {source}."
    if analysis["skipped"]:
        st.session_state.flash += f" {analysis['skipped']} rows were skipped because the text was empty."


def run_demo(open_dashboard: bool = False) -> None:
    store_analysis(
        read_csv_file(SAMPLE_DATA_PATH),
        source="the built-in demo dataset",
        text_column="text",
        product_column="product_name",
        category_column="category",
        date_column="date",
    )
    if open_dashboard:
        go_to(QUALITY)


def run_upload(df: pd.DataFrame, file_name: str) -> None:
    optional = lambda key: None if st.session_state[key] == "None" else st.session_state[key]  # noqa: E731
    try:
        store_analysis(
            df,
            source=file_name,
            text_column=st.session_state.map_text,
            product_column=optional("map_product"),
            category_column=optional("map_category"),
            date_column=optional("map_date"),
        )
    except ValueError as exc:
        st.session_state.flash_error = str(exc)


def clear_analysis() -> None:
    st.session_state.analysis_df = pd.DataFrame()
    st.session_state.analysis_meta = {}
    st.session_state.visited = set()


def use_example(text: str) -> None:
    st.session_state.single_text = text


def show_flash() -> None:
    if message := st.session_state.pop("flash", None):
        st.success(message, icon="✅")
    if error := st.session_state.pop("flash_error", None):
        st.error(error, icon="⚠️")


def next_step(label: str, page: str, key: str) -> None:
    st.markdown("")
    _, right = st.columns([3, 1])
    right.button(label, type="primary", on_click=go_to, args=(page,), key=key, width="stretch")


def empty_state(message: str, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### Nothing analysed yet\n{message}")
        left, right, _ = st.columns([1, 1, 2])
        left.button("▶ Use demo data", type="primary", on_click=run_demo, key=f"{key}_demo", width="stretch")
        right.button("Upload a CSV", on_click=go_to, args=(LOAD,), key=f"{key}_upload", width="stretch")


# ---------------------------------------------------------------------------
# Shell: brand bar, navigation, workflow stepper, sidebar
# ---------------------------------------------------------------------------
analysis_df: pd.DataFrame = st.session_state.analysis_df
meta: dict = st.session_state.analysis_meta
has_data = not analysis_df.empty

ui.brand_bar(
    f"{len(analysis_df):,} records · {meta.get('source', '')}" if has_data else "No data loaded yet",
    ready=has_data,
)

with st.container(key="nav"):
    page = st.radio(
        "Workspace navigation",
        list(NAVIGATION_LABELS),
        format_func=NAVIGATION_LABELS.get,
        horizontal=True,
        label_visibility="collapsed",
        key="workspace_navigation",
    )
st.session_state.visited.add(page)

if page in (OVERVIEW, LOAD, QUALITY, REPORT):
    def step_state(step_page: str, done: bool) -> str:
        return "done" if done else ("current" if page == step_page else "todo")

    ui.stepper(
        [
            ("Load data", "Demo data or your own CSV", step_state(LOAD, has_data)),
            ("Review quality", "Sentiment, defects, returns", step_state(QUALITY, has_data and QUALITY in st.session_state.visited and page != QUALITY)),
            ("Share the report", "Download a Markdown summary", step_state(REPORT, has_data and REPORT in st.session_state.visited and page != REPORT)),
        ]
    )

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Quality operations</div>', unsafe_allow_html=True)
    st.caption("Analysis workspace")
    st.metric("Records analyzed", f"{len(analysis_df):,}")
    if has_data:
        st.caption(f"Source: {meta.get('source', 'unknown')}")
        st.button("Clear analysis", on_click=clear_analysis, key="sidebar_clear", width="stretch")
    else:
        st.button("▶ Load demo data", type="primary", on_click=run_demo, args=(True,), key="sidebar_demo", width="stretch")
    st.divider()
    demo_note = "The demo dataset is synthetic. " if meta.get("source") == "the built-in demo dataset" else ""
    st.markdown(
        f'<div class="sidebar-note">{demo_note}Predictions and issue tags are signals for a human analyst, '
        "not proof of a defect.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == OVERVIEW:
    with st.container(key="hero"):
        st.markdown(
            '<div class="hero-title">Turn customer feedback into <em>product-quality decisions</em>.</div>'
            '<div class="hero-sub">Classify sentiment, spot recurring defects, and see which products drive returns, '
            "then hand your team a ready-made report. Start with the demo data or bring your own CSV.</div>",
            unsafe_allow_html=True,
        )
        left, right, _ = st.columns([1.1, 1, 2.4])
        left.button("▶ Try with demo data", type="primary", on_click=run_demo, args=(True,), key="hero_demo", width="stretch")
        right.button("Upload your CSV", on_click=go_to, args=(LOAD,), key="hero_upload", width="stretch")

    st.markdown("")
    ui.section_header("How it works", "Three steps from raw comments to a quality report")
    cols = st.columns(3)
    with cols[0]:
        ui.card("1", "Load data", "Use the demo dataset or upload a CSV. Text, product, category, and date columns are detected for you.")
    with cols[1]:
        ui.card("2", "Review quality", "See the sentiment mix, top defect themes, a product-by-issue heatmap, return trends, and products to investigate.")
    with cols[2]:
        ui.card("3", "Share the report", "Download a Markdown summary and the analysed CSV for your quality or operations team.")

    st.markdown("")
    ui.section_header("Under the hood", "What the platform combines")
    cols = st.columns(3)
    with cols[0]:
        ui.card("◆", "Sentiment model", "TF-IDF features with logistic regression label each comment positive, neutral, or negative.")
    with cols[1]:
        ui.card("◆", "Issue rules", "Transparent keyword rules tag themes such as transit damage, missing parts, and faulty components.")
    with cols[2]:
        ui.card("◆", "Return metrics", "Return flags and unit totals become return cases, unit return rates, and monthly trends.")

elif page == LOAD:
    ui.section_header("Step 1", "Load data", "Analyse the built-in demo dataset or upload a CSV of customer feedback.")
    show_flash()

    if has_data:
        with st.container(border=True):
            st.markdown("#### Current analysis")
            st.caption(f"Source: {meta.get('source', 'unknown')}")
            stats = st.columns(2)
            stats[0].metric("Records analysed", f"{meta.get('processed', len(analysis_df)):,}")
            stats[1].metric("Rows skipped", f"{meta.get('skipped', 0):,}")
            mapped = ", ".join(f"{role}: `{column}`" for role, column in meta.get("columns", {}).items() if column)
            if mapped:
                st.caption(f"Columns used: {mapped}")
            with st.expander("Preview analysed records"):
                st.dataframe(analysis_df.head(20), width="stretch", hide_index=True)
            actions = st.columns([1, 1, 2])
            actions[0].download_button(
                "Download analysed CSV",
                data=analysis_df.to_csv(index=False),
                file_name="analysed_results.csv",
                mime="text/csv",
                width="stretch",
            )
            actions[1].button("Clear analysis", on_click=clear_analysis, key="load_clear", width="stretch")

    demo_col, upload_col = st.columns(2, gap="large")
    with demo_col:
        with st.container(border=True):
            st.markdown("#### Use the demo dataset")
            st.caption("24 synthetic reviews across 8 products, with return statuses and unit totals, so every chart has data.")
            st.button("▶ Analyze built-in demo dataset", type="primary", on_click=run_demo, key="load_demo", width="stretch")

    with upload_col:
        with st.container(border=True):
            st.markdown("#### Upload your own CSV")
            st.caption("Only a feedback text column is required. UTF-8 and Excel CSV files both work.")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            upload_df = read_csv_file(uploaded_file)
        except Exception as exc:
            st.error(f"The CSV file could not be read: {exc}", icon="⚠️")
        else:
            columns = list(upload_df.columns)
            optional_columns = ["None"] + columns

            def default_index(role: str, options: list[str]) -> int:
                guess = guess_column(upload_df, role)
                return options.index(guess) if guess in options else 0

            with st.container(border=True):
                st.markdown(f"#### Map the columns in `{uploaded_file.name}`")
                st.caption(f"{len(upload_df):,} rows · {len(columns)} columns. Columns are pre-selected from their names; check each one before running.")
                first, second = st.columns(2)
                first.selectbox(
                    "Feedback text column",
                    columns,
                    index=default_index("text", columns),
                    key="map_text",
                    help="The free-text customer comment to classify, for example review_text.",
                )
                second.selectbox(
                    "Product name column (optional)",
                    optional_columns,
                    index=default_index("product", optional_columns),
                    key="map_product",
                    help="The product each comment is about, for example product_name. Not the return reason.",
                )
                first.selectbox("Product category column (optional)", optional_columns, index=default_index("category", optional_columns), key="map_category")
                second.selectbox("Date column (optional)", optional_columns, index=default_index("date", optional_columns), key="map_date")
                with st.expander("Preview the first 10 rows"):
                    st.dataframe(upload_df.head(10), width="stretch", hide_index=True)
                st.button(
                    "Run analysis",
                    type="primary",
                    on_click=run_upload,
                    args=(upload_df, uploaded_file.name),
                    key="run_upload",
                )

    if has_data:
        next_step("Next: review quality →", QUALITY, "next_quality")

elif page == QUALITY:
    ui.section_header("Step 2", "Quality dashboard", "Where customers are unhappy, what is going wrong, and which products need a closer look.")
    show_flash()
    if not has_data:
        empty_state("Load the demo dataset or upload a CSV to fill the dashboard.", "quality_empty")
    else:
        result_df = analysis_df.copy()
        summary = summarize_predictions(result_df)
        returned_cases = f"{int(returned_mask(result_df).sum()):,}" if has_return_info(result_df) else "Not provided"
        if has_unit_totals(result_df):
            return_rate_label = f"{calculate_product_return_rate(result_df)['percentage']:.2f}%"
            return_rate_help = "Distinct product totals: returned units divided by sold units"
        else:
            return_rate_label = "Not provided"
            return_rate_help = "Add units_sold and units_returned columns to calculate a rate"

        kpis = st.columns(4)
        kpis[0].metric("Feedback analysed", f"{summary['total']:,}", border=True)
        kpis[1].metric("Negative feedback", f"{summary['negative_pct']:.1f}%", help="Share of analysed comments predicted as negative", border=True)
        kpis[2].metric(
            "Reported return cases",
            returned_cases,
            help="Rows with return_status returned/refunded or return_requested = Yes; not a unit return rate",
            border=True,
        )
        kpis[3].metric("Unit return rate", return_rate_label, help=return_rate_help, border=True)

        left, right = st.columns(2, gap="large")
        with left:
            sentiments = ["positive", "neutral", "negative"]
            counts = [summary[label] for label in sentiments]
            shares = [summary[f"{label}_pct"] for label in sentiments]
            fig = go.Figure(
                go.Bar(
                    x=counts,
                    y=[label.title() for label in sentiments],
                    orientation="h",
                    marker_color=[ui.SENTIMENT_COLORS[label] for label in sentiments],
                    text=[f"{count} · {share:.0f}%" for count, share in zip(counts, shares)],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}: %{x} comments<extra></extra>",
                )
            )
            fig.update_yaxes(autorange="reversed", showgrid=False)
            # Leave room past the longest bar for its outside label.
            fig.update_xaxes(title_text="Comments", range=[0, max(max(counts), 1) * 1.3])
            ui.show_chart(ui.style_figure(fig, height=280, title="Sentiment mix"))

        with right:
            issues = result_df["issue_category"].fillna("Needs Review")
            issue_counts = issues[~issues.isin(["Other", "Needs Review"])].value_counts().head(6).sort_values()
            if issue_counts.empty:
                st.info("No classified defect themes in these records.")
            else:
                fig = go.Figure(
                    go.Bar(
                        x=issue_counts.values,
                        y=issue_counts.index,
                        orientation="h",
                        marker_color=ui.SERIES_BLUE,
                        hovertemplate="%{y}: %{x} comments<extra></extra>",
                    )
                )
                fig.update_yaxes(showgrid=False)
                fig.update_xaxes(title_text="Comments", dtick=1, rangemode="tozero")
                ui.show_chart(ui.style_figure(fig, height=280, title="Top defect themes"))

        left, right = st.columns([1.25, 1], gap="large")
        with left:
            if "product_name" in result_df.columns:
                defects = result_df[~result_df["issue_category"].fillna("Needs Review").isin(["Other", "Needs Review"])]
                if defects.empty:
                    st.info("No classified defect signals are available in these records.")
                else:
                    pivot = defects.pivot_table(index="issue_category", columns="product_name", values="text", aggfunc="count", fill_value=0)
                    fig = go.Figure(
                        go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns,
                            y=pivot.index,
                            colorscale=ui.heatmap_scale(),
                            zmin=0,
                            xgap=2,
                            ygap=2,
                            colorbar={"title": {"text": "Reports"}, "thickness": 10, "outlinewidth": 0, "dtick": 1},
                            hovertemplate="%{x}<br>%{y}: %{z} reports<extra></extra>",
                        )
                    )
                    fig.update_xaxes(showgrid=False, tickangle=-30)
                    fig.update_yaxes(showgrid=False)
                    ui.show_chart(ui.style_figure(fig, height=380, title="Defect reports by product"))
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
                elif trend["date"].dt.to_period("M").nunique() < 2:
                    # One month is not a trend; state the number instead of plotting a lone point.
                    month = trend["date"].dt.to_period("M").iloc[0].strftime("%B %Y")
                    with st.container(border=True):
                        st.metric(f"{trend_label} in {month}", f"{len(trend):,}")
                        st.caption("All dated records fall in one month. Add data from more months to see a trend.")
                else:
                    trend["month"] = trend["date"].dt.to_period("M").dt.to_timestamp()
                    trend_counts = trend.groupby("month").size()
                    fig = go.Figure(
                        go.Scatter(
                            x=trend_counts.index,
                            y=trend_counts.values,
                            mode="lines+markers",
                            line={"color": ui.SERIES_BLUE, "width": 2},
                            marker={"size": 9, "color": ui.SERIES_BLUE, "line": {"color": ui.PAPER, "width": 2}},
                            fill="tozeroy",
                            fillcolor="rgba(42,120,214,0.10)",
                            hovertemplate="%{x|%b %Y}: %{y}<extra></extra>",
                        )
                    )
                    fig.update_xaxes(showgrid=False, dtick="M1", tickformat="%b %Y")
                    fig.update_yaxes(title_text=trend_label, rangemode="tozero", dtick=1 if trend_counts.max() <= 6 else None)
                    ui.show_chart(ui.style_figure(fig, height=380, title=f"Monthly {trend_label.lower()}"))
            else:
                st.info("Select a date column when analysing data to show a returns trend.")

        st.markdown("### Products to investigate")
        flagged = flag_products_for_review(result_df)
        if flagged:
            st.dataframe(
                pd.DataFrame(flagged),
                width="stretch",
                hide_index=True,
                column_config={
                    "product_name": st.column_config.TextColumn("Product"),
                    "category": st.column_config.TextColumn("Category"),
                    "records": st.column_config.NumberColumn("Records"),
                    "negative_share": st.column_config.ProgressColumn("Negative share", format="percent", min_value=0, max_value=1),
                    "issue_rate": st.column_config.ProgressColumn("Issue rate", format="percent", min_value=0, max_value=1),
                    "reason": st.column_config.TextColumn("Why it was flagged", width="large"),
                },
            )
        else:
            st.success("No products currently meet the review thresholds in this dataset.", icon="✅")

        with st.expander("Explore analysed feedback"):
            filter_col, search_col = st.columns([1, 1.4])
            selected = filter_col.pills(
                "Sentiment",
                ["positive", "neutral", "negative"],
                selection_mode="multi",
                default=["positive", "neutral", "negative"],
                format_func=lambda label: f"{ui.SENTIMENT_ICONS[label]} {label.title()}",
                key="explore_sentiment",
            )
            search_term = search_col.text_input("Search feedback", placeholder="e.g. battery, lid, refund")
            filtered_df = result_df[result_df["predicted_sentiment"].str.lower().isin(selected or [])]
            if search_term:
                filtered_df = filtered_df[filtered_df["text"].str.contains(search_term, case=False, na=False, regex=False)]
            preferred = ["text", "product_name", "category", "predicted_sentiment", "confidence", "issue_category"]
            ordered = [column for column in preferred if column in filtered_df.columns]
            ordered += [column for column in filtered_df.columns if column not in ordered]
            st.caption(f"{len(filtered_df):,} of {len(result_df):,} records")
            st.dataframe(
                filtered_df[ordered],
                width="stretch",
                hide_index=True,
                column_config={
                    "text": st.column_config.TextColumn("Feedback", width="large"),
                    "product_name": "Product",
                    "category": "Category",
                    "predicted_sentiment": "Sentiment",
                    "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1),
                    "issue_category": "Issue",
                },
            )

        next_step("Next: build the report →", REPORT, "next_report")

elif page == REPORT:
    ui.section_header("Step 3", "Insights report", "A Markdown summary of the current analysis, ready to share with your team.")
    if not has_data:
        empty_state("Run an analysis first; the report is generated from it.", "report_empty")
    else:
        report = generate_quality_report(analysis_df)
        actions = st.columns([1, 1, 2])
        actions[0].download_button(
            "Download report (.md)",
            data=report,
            file_name="product_quality_insights_report.md",
            mime="text/markdown",
            type="primary",
            width="stretch",
        )
        actions[1].download_button(
            "Download analysed CSV",
            data=analysis_df.to_csv(index=False),
            file_name="analysed_results.csv",
            mime="text/csv",
            width="stretch",
        )
        preview_tab, source_tab = st.tabs(["Preview", "Markdown source"])
        with preview_tab:
            with st.container(border=True):
                st.markdown(report)
        with source_tab:
            st.text_area("Generated markdown report", report, height=450, label_visibility="collapsed")
        next_step("← Back to the dashboard", QUALITY, "back_quality")

elif page == SINGLE:
    ui.section_header("Quick check", "Analyse a single comment", "Paste one customer comment to see its sentiment, model confidence, and likely issue.")
    st.caption("Try an example:")
    with st.container(horizontal=True, key="examples", gap="small"):
        for label, text in EXAMPLE_COMMENTS.items():
            st.button(label, on_click=use_example, args=(text,), key=f"example_{label}", width="content")
    text_input = st.text_area(
        "Customer comment",
        height=140,
        key="single_text",
        placeholder="Example: The battery stopped charging after two days and the return process was confusing.",
    )
    if st.button("Analyse feedback", type="primary"):
        if not preprocess_text(text_input):
            st.warning("Please enter a comment before analysing.", icon="✏️")
        else:
            try:
                result = predict_text(text_input, model=model)
                label = result["label"]
                issue = classify_issue(text_input)
                left, right = st.columns(2)
                with left:
                    ui.result_card(
                        "Predicted sentiment",
                        label.title(),
                        accent=ui.SENTIMENT_COLORS.get(label, ui.MUTED),
                        icon=ui.SENTIMENT_ICONS.get(label, ""),
                        meter=result["confidence"],
                        note=f"Model confidence {result['confidence']:.0%}",
                    )
                with right:
                    ui.result_card(
                        "Likely issue category",
                        issue,
                        accent=ui.TEAL if issue not in ("Other", "Needs Review") else ui.MUTED,
                        note="From transparent keyword rules, not the sentiment model.",
                    )
                st.caption(
                    "Predictions come from patterns in labelled training data. Treat them as a signal, not a fact about the customer's intent."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}", icon="⚠️")

elif page == MODEL:
    ui.section_header("Transparency", "Model performance", "How the sentiment model scored on reviews it did not see during training.")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        scores = st.columns(4)
        scores[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}", border=True)
        scores[1].metric("Precision", f"{metrics.get('precision', 0):.1%}", border=True)
        scores[2].metric("Recall", f"{metrics.get('recall', 0):.1%}", border=True)
        scores[3].metric("F1-score", f"{metrics.get('f1', 0):.1%}", border=True)
        if "cv_f1_mean" in metrics:
            st.caption(
                f"Held-out test set: {metrics.get('test_size', '?')} records from {metrics.get('train_size', 0) + metrics.get('test_size', 0)} labelled reviews. "
                f"Cross-validated macro F1 on the training split: {metrics['cv_f1_mean']:.1%} (± {metrics['cv_f1_std']:.1%})."
            )
        labels = metrics.get("labels", ["positive", "negative", "neutral"])
        matrix = metrics.get("confusion_matrix", [[0] * len(labels) for _ in labels])
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            fig = go.Figure(
                go.Heatmap(
                    z=matrix,
                    x=[label.title() for label in labels],
                    y=[label.title() for label in labels],
                    colorscale=ui.heatmap_scale(),
                    zmin=0,
                    xgap=2,
                    ygap=2,
                    showscale=False,
                    text=matrix,
                    texttemplate="%{text}",
                    hovertemplate="Actual %{y}, predicted %{x}: %{z}<extra></extra>",
                )
            )
            fig.update_xaxes(title_text="Predicted", side="bottom", showgrid=False)
            fig.update_yaxes(title_text="Actual", autorange="reversed", showgrid=False)
            ui.show_chart(ui.style_figure(fig, height=360, title="Confusion matrix"))
        with right:
            st.markdown("#### Reading the scores")
            st.markdown(
                "- **Accuracy**: share of test reviews labelled correctly.\n"
                "- **Precision**: when the model says a label, how often it is right.\n"
                "- **Recall**: how many reviews of each label the model finds.\n"
                "- **F1**: balance of precision and recall, averaged across labels.\n\n"
                "The diagonal of the matrix counts correct predictions. The training set is small and synthetic, "
                "so treat these numbers as rough indicators."
            )
            with st.expander("Full classification report"):
                st.code(metrics.get("classification_report", "Classification report unavailable."), language=None)
    else:
        st.info("No metrics file is present yet. Run `python train_model.py` to generate performance details.")

elif page == ABOUT:
    ui.section_header("About", "About the platform", "What it does, how it works, and how to use it responsibly.")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### How the model works")
            st.markdown(
                "Each comment is cleaned and turned into TF-IDF features, which weight words that are frequent in the comment "
                "but rarer across all comments. A logistic regression classifier then predicts positive, neutral, or negative.\n\n"
                "Issue categories come from separate, transparent keyword rules, so you can see exactly why a comment was tagged."
            )
    with right:
        with st.container(border=True):
            st.markdown("#### Responsible use")
            st.markdown(
                "- Predictions can be wrong with sarcasm, slang, spelling errors, or mixed feelings.\n"
                "- Sentiment and issue tags are estimates, not proof of a product fault or customer intent.\n"
                "- Have a person review findings before warranty, returns, supplier, or manufacturing decisions.\n"
                "- Monthly return charts count dated records; they are not a unit return rate.\n"
                "- Remove personal information before uploading customer data."
            )

import pandas as pd

from src.analysis import analyse_dataset, summarize_sentiment_counts


def test_summarize_sentiment_counts():
    df = pd.DataFrame({
        "predicted_sentiment": ["positive", "negative", "positive", "neutral"],
    })
    summary = summarize_sentiment_counts(df)
    assert summary["positive"] == 2
    assert summary["negative"] == 1
    assert summary["neutral"] == 1
    assert summary["total"] == 4
    assert summary["positive_pct"] == 50.0


def test_analyse_dataset_handles_invalid_rows():
    df = pd.DataFrame(
        {
            "review": [
                "I love this product.",
                "",
                "The delivery was terrible.",
                None,
                "Price is okay.",
            ]
        }
    )
    result = analyse_dataset(df, text_column="review", model=None)
    assert result["processed"] == 3
    assert result["skipped"] == 2
    assert result["invalid"] == 0


def test_analyse_dataset_maps_selected_columns_to_standard_names():
    df = pd.DataFrame(
        {
            "comment": ["The motor stopped working.", "Great blender."],
            "item": ["Nova Blender", "Nova Blender"],
            "created": ["2024-03-02", "not a date"],
        }
    )
    result = analyse_dataset(df, text_column="comment", product_column="item", date_column="created")
    result_df = result["dataframe"]
    assert result_df["product_name"].tolist() == ["Nova Blender", "Nova Blender"]
    assert pd.api.types.is_datetime64_any_dtype(result_df["date"])
    assert result_df["date"].isna().tolist() == [False, True]
    assert result["summary"]["return_rate"] is None

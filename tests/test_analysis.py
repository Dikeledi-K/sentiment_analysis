import io

import pandas as pd

from src.analysis import analyse_dataset, guess_column, read_csv_file, summarize_sentiment_counts


def test_guess_column_picks_columns_by_name():
    df = pd.DataFrame(
        columns=["review_id", "review_date", "product_id", "product_name", "product_category", "review_text", "return_reason"]
    )
    assert guess_column(df, "text") == "review_text"
    assert guess_column(df, "product") == "product_name"
    assert guess_column(df, "category") == "product_category"
    assert guess_column(df, "date") == "review_date"


def test_guess_column_falls_back_to_longest_text_field():
    df = pd.DataFrame({"id": ["R1", "R2"], "notes": ["The motor stopped after a week.", "Lovely and quiet."]})
    assert guess_column(df, "text") == "notes"
    assert guess_column(df, "product") is None


def test_selecting_another_column_keeps_original_product_names():
    df = pd.DataFrame(
        {
            "review_text": ["The phone arrived with a cracked screen."],
            "product_name": ["Nova X Smartphone"],
            "return_reason": ["Damaged in transit"],
        }
    )
    result_df = analyse_dataset(df, "review_text", product_column="return_reason")["dataframe"]
    assert result_df["product_name"].tolist() == ["Damaged in transit"]
    assert result_df["product_name_original"].tolist() == ["Nova X Smartphone"]


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


def test_results_keep_original_comment_wording():
    df = pd.DataFrame({"review_text": ["  Excellent phone, clear camera!  "]})
    result_df = analyse_dataset(df, "review_text")["dataframe"]
    assert result_df["text"].tolist() == ["Excellent phone, clear camera!"]


def test_guess_column_ignores_empty_text_columns():
    df = pd.DataFrame({"empty": [None, None], "notes": ["Short", "A longer comment"]}).astype(object)
    assert guess_column(df, "text") == "notes"


def test_read_csv_file_accepts_excel_windows_encoding():
    raw = "review_text,product_name\nThe café grinder’s motor died,Grinder\n".encode("cp1252")
    df = read_csv_file(io.BytesIO(raw))
    assert df["review_text"].tolist() == ["The café grinder’s motor died"]


def test_read_csv_file_strips_utf8_byte_order_mark():
    raw = "review_text\nGood\n".encode("utf-8-sig")
    assert list(read_csv_file(io.BytesIO(raw)).columns) == ["review_text"]

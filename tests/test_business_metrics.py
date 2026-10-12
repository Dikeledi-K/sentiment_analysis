import pandas as pd

from src.business_metrics import calculate_product_return_rate, flag_products_for_review, has_return_info, returned_mask


def test_product_return_rate_counts_each_product_total_once():
    df = pd.DataFrame(
        {
            "product_name": ["A", "A", "A", "B"],
            "units_sold": [100, 100, 100, 100],
            "units_returned": [10, 10, 10, 30],
        }
    )
    assert calculate_product_return_rate(df)["percentage"] == 20.0


def test_returned_mask_matches_known_statuses():
    df = pd.DataFrame({"return_status": ["Returned", " refunded ", "open", None]})
    assert returned_mask(df).tolist() == [True, True, False, False]


def test_returned_mask_reads_yes_no_return_flag():
    df = pd.DataFrame({"return_requested": ["Yes", "No", "yes", None]})
    assert has_return_info(df)
    assert returned_mask(df).tolist() == [True, False, True, False]


def test_flags_ignore_needs_review_rows():
    df = pd.DataFrame(
        {
            "product_name": ["A", "A"],
            "predicted_sentiment": ["positive", "positive"],
            "issue_category": ["Needs Review", "Needs Review"],
        }
    )
    assert flag_products_for_review(df) == []


def test_flags_with_only_category_column_do_not_mislabel_product():
    df = pd.DataFrame(
        {
            "category": ["Audio", "Audio"],
            "predicted_sentiment": ["negative", "negative"],
            "issue_category": ["Faulty components", "Other"],
        }
    )
    flags = flag_products_for_review(df)
    assert len(flags) == 1
    assert flags[0]["category"] == "Audio"
    assert flags[0]["product_name"] == "Not provided"

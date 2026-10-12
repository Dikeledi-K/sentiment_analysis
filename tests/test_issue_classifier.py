from src.issue_classifier import classify_issue


def test_keywords_match_whole_words_only():
    # "bent" must not match inside "absent".
    assert classify_issue("The USB dongle was absent from the box.") == "Missing parts or accessories"


def test_specific_categories_win_over_generic_terms():
    assert classify_issue("Missing the lid and the jar arrived scratched.") == "Missing parts or accessories"
    assert classify_issue("Product was damaged in transit and the box arrived crushed.") == "Damaged in transit"


def test_unmatched_and_empty_text():
    assert classify_issue("Lovely colour and a nice design.") == "Other"
    assert classify_issue("   ") == "Needs Review"

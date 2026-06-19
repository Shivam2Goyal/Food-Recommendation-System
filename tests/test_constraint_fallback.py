# tests/test_constraint_fallback.py

from src.constraint_fallback import apply_constraint_fallback, describe_fallback_reason

GOOD_COVERAGE_DISH = {
    "name": "Paneer Butter Masala",
    "coverage_score": 0.9,
    "missing_ingredients": ["cream"],
}

LOW_COVERAGE_DISH = {
    "name": "Mutton Rogan Josh",
    "coverage_score": 0.1,
    "missing_ingredients": ["mutton", "yogurt", "kashmiri chili"],
}

ZERO_COVERAGE_DISH = {
    "name": "Rasgulla",
    "coverage_score": 0.0,
    "missing_ingredients": ["chenna", "sugar", "water"],
}

EXACTLY_THRESHOLD_DISH = {
    "name": "Borderline Dish",
    "coverage_score": 0.4,
    "missing_ingredients": ["salt"],
}


# ── apply_constraint_fallback ──────────────────────────────────────────────────

def test_none_input_returns_none():
    assert apply_constraint_fallback(None) is None

def test_never_filters_out_low_coverage_dish():
    # The whole point: low coverage NEVER results in no recommendation
    result = apply_constraint_fallback(LOW_COVERAGE_DISH)
    assert result is not None
    assert result["name"] == "Mutton Rogan Josh"

def test_never_filters_out_zero_coverage_dish():
    result = apply_constraint_fallback(ZERO_COVERAGE_DISH)
    assert result is not None
    assert result["name"] == "Rasgulla"

def test_low_coverage_flagged_true():
    result = apply_constraint_fallback(LOW_COVERAGE_DISH)
    assert result["low_coverage_warning"] is True

def test_good_coverage_flagged_false():
    result = apply_constraint_fallback(GOOD_COVERAGE_DISH)
    assert result["low_coverage_warning"] is False

def test_exactly_at_threshold_not_flagged():
    # 0.4 is the threshold; "< 0.4" means exactly 0.4 should NOT be flagged
    result = apply_constraint_fallback(EXACTLY_THRESHOLD_DISH)
    assert result["low_coverage_warning"] is False

def test_just_below_threshold_flagged():
    dish = {"name": "X", "coverage_score": 0.39, "missing_ingredients": ["a"]}
    result = apply_constraint_fallback(dish)
    assert result["low_coverage_warning"] is True

def test_original_dish_fields_preserved():
    result = apply_constraint_fallback(GOOD_COVERAGE_DISH)
    assert result["name"] == GOOD_COVERAGE_DISH["name"]
    assert result["coverage_score"] == GOOD_COVERAGE_DISH["coverage_score"]
    assert result["missing_ingredients"] == GOOD_COVERAGE_DISH["missing_ingredients"]

def test_missing_coverage_score_defaults_safely():
    # Defensive: if coverage_score key is somehow absent, don't crash
    dish = {"name": "Mystery Dish", "missing_ingredients": []}
    result = apply_constraint_fallback(dish)
    assert result["low_coverage_warning"] is True  # defaults to 0.0 -> flagged

# ── describe_fallback_reason ───────────────────────────────────────────────────

def test_describe_returns_none_when_nothing_missing():
    dish = {"name": "X", "coverage_score": 1.0, "missing_ingredients": []}
    assert describe_fallback_reason(dish) is None

def test_describe_includes_missing_ingredients():
    desc = describe_fallback_reason(LOW_COVERAGE_DISH)
    assert "mutton" in desc
    assert "yogurt" in desc
    assert "kashmiri chili" in desc

def test_describe_includes_coverage_percentage():
    desc = describe_fallback_reason(LOW_COVERAGE_DISH)
    assert "10%" in desc

def test_describe_is_a_string():
    desc = describe_fallback_reason(LOW_COVERAGE_DISH)
    assert isinstance(desc, str)
    assert len(desc) > 0
# tests/test_matcher.py

from src.matcher import match_ingredient, compute_coverage

# ── match_ingredient ───────────────────────────────────────────────────────────

def test_exact_match():
    assert match_ingredient("tomato", "tomato") is True

def test_synonym_match():
    # tomatoes → tomato via synonyms.py
    assert match_ingredient("tomatoes", "tomato") is True

def test_hindi_synonym():
    # tamatar → tomato via synonyms.py
    assert match_ingredient("tamatar", "tomato") is True

def test_fuzzy_match():
    # slight typo / spacing
    assert match_ingredient("corriander", "coriander") is True

def test_substring_match():
    # "basmati rice" satisfies "rice"
    assert match_ingredient("basmati rice", "rice") is True

def test_substitute_ghee_butter():
    assert match_ingredient("ghee", "butter") is True

def test_substitute_paneer_tofu():
    assert match_ingredient("tofu", "paneer") is True

def test_no_match():
    assert match_ingredient("sugar", "salt") is False

def test_no_match_unrelated():
    assert match_ingredient("chicken", "lentils") is False

# ── compute_coverage ──────────────────────────────────────────────────────────

def test_full_coverage():
    result = compute_coverage(["tomato", "onion", "garlic"], ["tomato", "onion", "garlic"])
    assert result["coverage_score"] == 1.0
    assert result["missing_ingredients"] == []

def test_partial_coverage():
    result = compute_coverage(["tomato", "onion"], ["tomato", "onion", "garlic", "cumin"])
    assert result["coverage_score"] == 0.5
    assert "garlic" in result["missing_ingredients"]
    assert "cumin" in result["missing_ingredients"]

def test_zero_coverage():
    result = compute_coverage(["sugar", "milk"], ["lentils", "cumin", "turmeric"])
    assert result["coverage_score"] == 0.0
    assert len(result["missing_ingredients"]) == 3

def test_synonym_coverage():
    # user says "tomatoes" and "dahi", dish needs "tomato" and "yogurt"
    result = compute_coverage(["tomatoes", "dahi"], ["tomato", "yogurt"])
    assert result["coverage_score"] == 1.0

def test_coverage_score_rounds_to_3dp():
    result = compute_coverage(["tomato"], ["tomato", "onion", "garlic"])
    assert result["coverage_score"] == round(1/3, 3)

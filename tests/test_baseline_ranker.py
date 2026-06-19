# tests/test_baseline_ranker.py

from src.baseline_ranker import (
    rank_dishes,
    score_meal_type,
    score_prep_time,
    score_user_type,
)

SAMPLE_DISHES = [
    {
        "name": "Paneer Butter Masala",
        "ingredients": ["paneer", "butter", "tomato", "onion", "cream"],
        "meal_type": "dinner",
        "prep_time_mins": 30,
        "dietary_tags": ["vegetarian", "comfort"],
    },
    {
        "name": "Poha",
        "ingredients": ["flattened rice", "onion", "mustard seeds", "turmeric"],
        "meal_type": "breakfast",
        "prep_time_mins": 15,
        "dietary_tags": ["vegetarian", "quick", "easy"],
    },
    {
        "name": "Mutton Curry",
        "ingredients": ["mutton", "onion", "tomato", "garam masala", "ginger"],
        "meal_type": "dinner",
        "prep_time_mins": 90,
        "dietary_tags": ["non-vegetarian", "traditional"],
    },
]

# ── score_meal_type ────────────────────────────────────────────────────────────

def test_meal_type_exact_match():
    assert score_meal_type(SAMPLE_DISHES[0], "dinner") == 1.0

def test_meal_type_mismatch():
    assert score_meal_type(SAMPLE_DISHES[0], "breakfast") == 0.0

def test_meal_type_no_preference():
    assert score_meal_type(SAMPLE_DISHES[0], None) == 0.5

# ── score_prep_time ───────────────────────────────────────────────────────────

def test_prep_time_within_budget():
    assert score_prep_time(SAMPLE_DISHES[1], 30) == 1.0  # 15 mins, budget 30

def test_prep_time_no_budget():
    assert score_prep_time(SAMPLE_DISHES[0], None) == 0.5

def test_prep_time_over_budget_falloff():
    # Mutton curry = 90 mins, budget = 30 -> way over, should be 0
    score = score_prep_time(SAMPLE_DISHES[2], 30)
    assert score == 0.0

def test_prep_time_slightly_over_budget():
    # 30 min dish, 25 min budget -> slightly over, score between 0 and 1
    score = score_prep_time(SAMPLE_DISHES[0], 25)
    assert 0.0 < score < 1.0

# ── score_user_type ───────────────────────────────────────────────────────────

def test_user_type_match_bonus():
    # Poha has 'quick' and 'easy' tags, bachelor hints include both
    assert score_user_type(SAMPLE_DISHES[1], "bachelor") == 1.0

def test_user_type_no_overlap():
    # Mutton curry has no 'quick'/'easy'/'snack' tags
    assert score_user_type(SAMPLE_DISHES[2], "bachelor") == 0.3

def test_user_type_none_given():
    assert score_user_type(SAMPLE_DISHES[0], None) == 0.5

def test_user_type_unknown_type():
    assert score_user_type(SAMPLE_DISHES[0], "alien") == 0.5

# ── rank_dishes (integration) ─────────────────────────────────────────────────

def test_rank_dishes_returns_top_k():
    results = rank_dishes(
        SAMPLE_DISHES,
        user_ingredients=["paneer", "tomato", "onion"],
        top_k=2,
    )
    assert len(results) == 2

def test_rank_dishes_sorted_descending():
    results = rank_dishes(
        SAMPLE_DISHES,
        user_ingredients=["paneer", "butter", "tomato", "onion", "cream"],
        top_k=3,
    )
    scores = [d["baseline_score"] for d in results]
    assert scores == sorted(scores, reverse=True)

def test_rank_dishes_full_coverage_ranks_first():
    # User has everything for Paneer Butter Masala, nothing for the others
    results = rank_dishes(
        SAMPLE_DISHES,
        user_ingredients=["paneer", "butter", "tomato", "onion", "cream"],
        requested_meal_type="dinner",
        top_k=3,
    )
    assert results[0]["name"] == "Paneer Butter Masala"
    assert results[0]["coverage_score"] == 1.0

def test_rank_dishes_includes_missing_ingredients():
    results = rank_dishes(
        SAMPLE_DISHES,
        user_ingredients=["onion"],
        top_k=1,
    )
    assert "missing_ingredients" in results[0]
    assert "ingredients_available" in results[0]

def test_rank_dishes_respects_meal_type_filter_softly():
    # Breakfast preference should boost Poha even with partial ingredient match
    results = rank_dishes(
        SAMPLE_DISHES,
        user_ingredients=["onion", "turmeric"],
        requested_meal_type="breakfast",
        top_k=3,
    )
    top_names = [d["name"] for d in results]
    assert "Poha" in top_names

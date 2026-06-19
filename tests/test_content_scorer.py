# tests/test_content_scorer.py

import pytest
from src.content_scorer import ContentScorer, _dish_to_feature_string

SAMPLE_DISHES = [
    {
        "name": "Paneer Butter Masala",
        "ingredients": ["paneer", "butter", "tomato", "onion", "cream"],
        "region": "North",
        "state": "Punjab",
        "flavor_profile": "spicy",
        "meal_type": "dinner",
        "dietary_tags": ["vegetarian"],
    },
    {
        "name": "Shahi Paneer",
        "ingredients": ["paneer", "cream", "tomato", "cashews", "onion"],
        "region": "North",
        "state": "Punjab",
        "flavor_profile": "spicy",
        "meal_type": "dinner",
        "dietary_tags": ["vegetarian"],
    },
    {
        "name": "Rasgulla",
        "ingredients": ["chenna", "sugar", "water"],
        "region": "East",
        "state": "West Bengal",
        "flavor_profile": "sweet",
        "meal_type": "snack",
        "dietary_tags": ["vegetarian", "dessert"],
    },
    {
        "name": "Mutton Rogan Josh",
        "ingredients": ["mutton", "yogurt", "onion", "kashmiri chili"],
        "region": "North",
        "state": "Kashmir",
        "flavor_profile": "spicy",
        "meal_type": "dinner",
        "dietary_tags": ["non-vegetarian"],
    },
]

# ── _dish_to_feature_string ───────────────────────────────────────────────────

def test_feature_string_contains_ingredients():
    s = _dish_to_feature_string(SAMPLE_DISHES[0])
    assert "paneer" in s
    assert "tomato" in s

def test_feature_string_contains_region_and_state():
    s = _dish_to_feature_string(SAMPLE_DISHES[0])
    assert "region_north" in s
    assert "state_punjab" in s

def test_feature_string_contains_meal_type():
    s = _dish_to_feature_string(SAMPLE_DISHES[0])
    assert "meal_dinner" in s

# ── ContentScorer.similar_to ──────────────────────────────────────────────────

def test_scorer_initializes():
    scorer = ContentScorer(SAMPLE_DISHES)
    assert scorer.tfidf_matrix.shape[0] == len(SAMPLE_DISHES)

def test_similar_to_excludes_self():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = scorer.similar_to("Paneer Butter Masala", top_k=3)
    names = [r["name"] for r in results]
    assert "Paneer Butter Masala" not in names

def test_similar_to_ranks_closest_dish_first():
    # Shahi Paneer shares the most ingredients/region/flavor with Paneer Butter Masala
    scorer = ContentScorer(SAMPLE_DISHES)
    results = scorer.similar_to("Paneer Butter Masala", top_k=3)
    assert results[0]["name"] == "Shahi Paneer"

def test_similar_to_dessert_ranks_low_for_curry():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = scorer.similar_to("Paneer Butter Masala", top_k=3)
    names_in_order = [r["name"] for r in results]
    # Rasgulla (sweet dessert) should rank below Shahi Paneer and Rogan Josh
    assert names_in_order.index("Rasgulla") > names_in_order.index("Shahi Paneer")

def test_similar_to_invalid_name_raises():
    scorer = ContentScorer(SAMPLE_DISHES)
    with pytest.raises(ValueError):
        scorer.similar_to("Nonexistent Dish")

def test_similar_to_returns_similarity_field():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = scorer.similar_to("Paneer Butter Masala", top_k=1)
    assert "content_similarity" in results[0]
    assert 0.0 <= results[0]["content_similarity"] <= 1.0

# ── ContentScorer.score_against_profile ───────────────────────────────────────

def test_score_against_profile_excludes_liked():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = scorer.score_against_profile(["Paneer Butter Masala"])
    names = [r["name"] for r in results]
    assert "Paneer Butter Masala" not in names

def test_score_against_profile_ranks_similar_dish_high():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = scorer.score_against_profile(["Paneer Butter Masala"])
    assert results[0]["name"] == "Shahi Paneer"

def test_score_against_profile_invalid_liked_raises():
    scorer = ContentScorer(SAMPLE_DISHES)
    with pytest.raises(ValueError):
        scorer.score_against_profile(["Nonexistent Dish"])

def test_score_against_profile_multiple_liked_dishes():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = scorer.score_against_profile(["Paneer Butter Masala", "Rasgulla"])
    # Should return remaining 2 dishes, each with a content_score
    assert len(results) == 2
    assert all("content_score" in r for r in results)

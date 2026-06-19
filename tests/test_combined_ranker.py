# tests/test_combined_ranker.py

import pytest
from src.combined_ranker import get_blend_weights, combined_rank
from src.content_scorer import ContentScorer

SAMPLE_DISHES = [
    {
        "name": "Paneer Butter Masala",
        "ingredients": ["paneer", "butter", "tomato", "onion", "cream"],
        "region": "North",
        "state": "Punjab",
        "flavor_profile": "spicy",
        "meal_type": "dinner",
        "prep_time_mins": 30,
        "dietary_tags": ["vegetarian"],
    },
    {
        "name": "Shahi Paneer",
        "ingredients": ["paneer", "cream", "tomato", "cashews", "onion"],
        "region": "North",
        "state": "Punjab",
        "flavor_profile": "spicy",
        "meal_type": "dinner",
        "prep_time_mins": 35,
        "dietary_tags": ["vegetarian"],
    },
    {
        "name": "Rasgulla",
        "ingredients": ["chenna", "sugar", "water"],
        "region": "East",
        "state": "West Bengal",
        "flavor_profile": "sweet",
        "meal_type": "snack",
        "prep_time_mins": 45,
        "dietary_tags": ["vegetarian", "dessert"],
    },
    {
        "name": "Mutton Rogan Josh",
        "ingredients": ["mutton", "yogurt", "onion", "kashmiri chili"],
        "region": "North",
        "state": "Kashmir",
        "flavor_profile": "spicy",
        "meal_type": "dinner",
        "prep_time_mins": 90,
        "dietary_tags": ["non-vegetarian"],
    },
]

# ── get_blend_weights ──────────────────────────────────────────────────────────

def test_zero_liked_is_pure_baseline():
    w = get_blend_weights(0)
    assert w == {"baseline": 1.0, "content": 0.0}

def test_one_to_two_liked_mostly_baseline():
    w = get_blend_weights(1)
    assert w["baseline"] > w["content"]
    assert w["baseline"] + w["content"] == 1.0

def test_three_to_four_liked_even_split():
    w = get_blend_weights(3)
    assert w == {"baseline": 0.5, "content": 0.5}

def test_five_plus_liked_mostly_content():
    w = get_blend_weights(5)
    assert w["content"] > w["baseline"]
    assert w["baseline"] + w["content"] == 1.0

def test_weights_always_sum_to_one():
    for n in [0, 1, 2, 3, 4, 5, 10, 100]:
        w = get_blend_weights(n)
        assert abs((w["baseline"] + w["content"]) - 1.0) < 1e-9

# ── combined_rank ──────────────────────────────────────────────────────────────

def test_combined_rank_no_liked_dishes_uses_pure_baseline():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = combined_rank(
        SAMPLE_DISHES,
        content_scorer=scorer,
        user_ingredients=["paneer", "butter", "tomato", "onion", "cream"],
        liked_dish_names=[],
        top_k=3,
    )
    # With 0 liked dishes, combined_score should equal baseline_score
    for d in results:
        assert d["combined_score"] == d["baseline_score"]
        assert d["content_score"] == 0.0

def test_combined_rank_excludes_liked_dishes():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = combined_rank(
        SAMPLE_DISHES,
        content_scorer=scorer,
        user_ingredients=["paneer", "tomato"],
        liked_dish_names=["Paneer Butter Masala"],
        top_k=10,
    )
    names = [d["name"] for d in results]
    assert "Paneer Butter Masala" not in names

def test_combined_rank_boosts_similar_dish_when_liked():
    scorer = ContentScorer(SAMPLE_DISHES)

    # Without any liked dishes
    baseline_only = combined_rank(
        SAMPLE_DISHES,
        content_scorer=scorer,
        user_ingredients=["onion"],  # weak ingredient signal for everyone
        liked_dish_names=[],
        top_k=10,
    )
    baseline_rasgulla_rank = next(
        i for i, d in enumerate(baseline_only) if d["name"] == "Rasgulla"
    )

    # After liking Paneer Butter Masala, Shahi Paneer (very similar) should
    # rank above Rasgulla (a sweet dessert, very different)
    with_like = combined_rank(
        SAMPLE_DISHES,
        content_scorer=scorer,
        user_ingredients=["onion"],
        liked_dish_names=["Paneer Butter Masala"],
        top_k=10,
    )
    names_with_like = [d["name"] for d in with_like]
    assert names_with_like.index("Shahi Paneer") < names_with_like.index("Rasgulla")

def test_combined_rank_returns_top_k():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = combined_rank(
        SAMPLE_DISHES,
        content_scorer=scorer,
        user_ingredients=["onion"],
        liked_dish_names=[],
        top_k=2,
    )
    assert len(results) == 2

def test_combined_rank_invalid_liked_dish_falls_back_to_baseline():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = combined_rank(
        SAMPLE_DISHES,
        content_scorer=scorer,
        user_ingredients=["paneer", "tomato"],
        liked_dish_names=["Nonexistent Dish"],
        top_k=3,
    )
    # Should not crash, and should fall back to pure baseline weighting
    for d in results:
        assert d["_blend_weights"] == {"baseline": 1.0, "content": 0.0}

def test_combined_rank_includes_score_breakdown_fields():
    scorer = ContentScorer(SAMPLE_DISHES)
    results = combined_rank(
        SAMPLE_DISHES,
        content_scorer=scorer,
        user_ingredients=["paneer"],
        liked_dish_names=[],
        top_k=1,
    )
    d = results[0]
    assert "baseline_score" in d
    assert "content_score" in d
    assert "combined_score" in d
    assert "_blend_weights" in d

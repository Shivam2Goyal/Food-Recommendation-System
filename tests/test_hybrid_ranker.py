# tests/test_hybrid_ranker.py

import pytest
from pathlib import Path
from src.hybrid_ranker import hybrid_rank, get_cf_weight
from src.content_scorer import ContentScorer
from src.feedback_store import FeedbackStore

SAMPLE_DISHES = [
    {
        "name": "Paneer Butter Masala",
        "ingredients": ["paneer", "butter", "tomato", "onion", "cream"],
        "region": "North", "state": "Punjab", "flavor_profile": "spicy",
        "meal_type": "dinner", "prep_time_mins": 30,
        "dietary_tags": ["vegetarian"],
    },
    {
        "name": "Shahi Paneer",
        "ingredients": ["paneer", "cream", "tomato", "cashews", "onion"],
        "region": "North", "state": "Punjab", "flavor_profile": "spicy",
        "meal_type": "dinner", "prep_time_mins": 35,
        "dietary_tags": ["vegetarian"],
    },
    {
        "name": "Rasgulla",
        "ingredients": ["chenna", "sugar", "water"],
        "region": "East", "state": "West Bengal", "flavor_profile": "sweet",
        "meal_type": "snack", "prep_time_mins": 45,
        "dietary_tags": ["vegetarian", "dessert"],
    },
    {
        "name": "Mutton Rogan Josh",
        "ingredients": ["mutton", "yogurt", "onion", "kashmiri chili"],
        "region": "North", "state": "Kashmir", "flavor_profile": "spicy",
        "meal_type": "dinner", "prep_time_mins": 90,
        "dietary_tags": ["non-vegetarian"],
    },
    {
        "name": "Poha",
        "ingredients": ["flattened rice", "onion", "mustard seeds", "turmeric"],
        "region": "West", "state": "Maharashtra", "flavor_profile": "savory",
        "meal_type": "breakfast", "prep_time_mins": 15,
        "dietary_tags": ["vegetarian", "quick"],
    },
]


@pytest.fixture
def scorer():
    return ContentScorer(SAMPLE_DISHES)


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=tmp_path / "test_hybrid.db")


# ── get_cf_weight ──────────────────────────────────────────────────────────────

def test_cf_weight_zero_at_no_feedback():
    assert get_cf_weight(0) == 0.0

def test_cf_weight_small_at_low_feedback():
    assert 0 < get_cf_weight(1) < 0.3
    assert 0 < get_cf_weight(4) < 0.3

def test_cf_weight_moderate_at_mid_feedback():
    assert get_cf_weight(5) > get_cf_weight(4)

def test_cf_weight_high_at_rich_feedback():
    assert get_cf_weight(20) > get_cf_weight(5)

def test_cf_weight_monotonically_increases():
    counts = [0, 1, 5, 10, 15, 20, 50]
    weights = [get_cf_weight(c) for c in counts]
    assert weights == sorted(weights)

# ── hybrid_rank: cold start (no feedback) ──────────────────────────────────────

def test_hybrid_rank_no_feedback_behaves_like_combined_rank(scorer, store):
    results = hybrid_rank(
        SAMPLE_DISHES,
        content_scorer=scorer,
        feedback_store=store,
        user_id="brand_new_user",
        user_ingredients=["paneer", "tomato", "onion"],
        top_k=5,
    )
    for d in results:
        assert d["cf_score"] == 0.0
        assert d["_cf_weight"] == 0.0
        # with cf_weight 0, hybrid_score should equal combined_score
        assert d["hybrid_score"] == d["combined_score"]

def test_hybrid_rank_returns_feedback_count(scorer, store):
    results = hybrid_rank(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_id="new_user", user_ingredients=["onion"], top_k=1,
    )
    assert results[0]["_feedback_count"] == 0

# ── hybrid_rank: with feedback + CF signal ────────────────────────────────────

def test_hybrid_rank_excludes_liked_dishes(scorer, store):
    store.add_feedback("u1", "Paneer Butter Masala", 1)
    results = hybrid_rank(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_id="u1", user_ingredients=["paneer", "tomato"], top_k=10,
    )
    names = [d["name"] for d in results]
    assert "Paneer Butter Masala" not in names

def test_hybrid_rank_excludes_disliked_dishes(scorer, store):
    store.add_feedback("u1", "Rasgulla", -1)
    results = hybrid_rank(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_id="u1", user_ingredients=["onion"], top_k=10,
    )
    names = [d["name"] for d in results]
    assert "Rasgulla" not in names

def test_hybrid_rank_respects_explicit_exclude_list(scorer, store):
    results = hybrid_rank(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_id="u1", user_ingredients=["onion"],
        exclude_dish_names=["Poha"], top_k=10,
    )
    names = [d["name"] for d in results]
    assert "Poha" not in names

def test_hybrid_rank_uses_cf_when_sufficient_data(scorer, store):
    # u1 and u2 share taste; u2 also liked Shahi Paneer which u1 hasn't rated
    store.add_feedback("u1", "Paneer Butter Masala", 1)
    store.add_feedback("u1", "Rasgulla", -1)
    store.add_feedback("u2", "Paneer Butter Masala", 1)
    store.add_feedback("u2", "Rasgulla", -1)
    store.add_feedback("u2", "Shahi Paneer", 1)

    results = hybrid_rank(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_id="u1", user_ingredients=["onion"], top_k=10,
    )
    shahi = next(d for d in results if d["name"] == "Shahi Paneer")
    assert shahi["cf_score"] > 0  # CF should have picked up the signal

def test_hybrid_rank_falls_back_when_no_cf_neighbors(scorer, store):
    # u1 has feedback but is the only user -> no neighbors -> cf_weight forced to 0
    store.add_feedback("u1", "Paneer Butter Masala", 1)
    store.add_feedback("u1", "Rasgulla", -1)

    results = hybrid_rank(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_id="u1", user_ingredients=["onion"], top_k=10,
    )
    for d in results:
        assert d["_cf_weight"] == 0.0

def test_hybrid_rank_top_k_respected(scorer, store):
    results = hybrid_rank(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_id="u1", user_ingredients=["onion"], top_k=2,
    )
    assert len(results) == 2

def test_hybrid_rank_sorted_descending(scorer, store):
    results = hybrid_rank(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_id="u1", user_ingredients=["paneer", "onion"], top_k=10,
    )
    scores = [d["hybrid_score"] for d in results]
    assert scores == sorted(scores, reverse=True)

# tests/test_session.py

import pytest
from pathlib import Path
from src.session import RecommendationSession
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
]


@pytest.fixture
def scorer():
    return ContentScorer(SAMPLE_DISHES)


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=tmp_path / "test_session.db")


@pytest.fixture
def session():
    return RecommendationSession(user_id="test_user")


# ── get_next_recommendation ────────────────────────────────────────────────────

def test_first_recommendation_returns_a_dish(session, scorer, store):
    dish = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer", "tomato"],
    )
    assert dish is not None
    assert "name" in dish

def test_recommendation_added_to_shown_list(session, scorer, store):
    dish = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer", "tomato"],
    )
    assert dish["name"] in session.shown_dish_names

def test_same_dish_never_repeats_in_session(session, scorer, store):
    seen = set()
    for _ in range(len(SAMPLE_DISHES)):
        dish = session.get_next_recommendation(
            SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
            user_ingredients=["onion"],
        )
        if dish is None:
            break
        assert dish["name"] not in seen
        seen.add(dish["name"])

def test_returns_none_when_catalogue_exhausted(session, scorer, store):
    # Catalogue has 4 dishes — after 4 recommendations, nothing left
    for _ in range(4):
        session.get_next_recommendation(
            SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
            user_ingredients=["onion"],
        )
    result = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["onion"],
    )
    assert result is None

# ── reject_current ─────────────────────────────────────────────────────────────

def test_reject_records_permanent_dislike(session, scorer, store):
    dish = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer"],
    )
    session.reject_current(dish["name"], store)
    assert dish["name"] in store.get_disliked_dishes("test_user")

def test_reject_tracked_in_session(session, scorer, store):
    dish = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer"],
    )
    session.reject_current(dish["name"], store)
    assert dish["name"] in session.rejected_this_session

def test_rejected_dish_not_recommended_again_after_reject(session, scorer, store):
    first = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer", "tomato", "onion"],
    )
    session.reject_current(first["name"], store)

    second = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer", "tomato", "onion"],
    )
    assert second["name"] != first["name"]

def test_rejected_dish_excluded_in_future_session(scorer, store):
    # Reject in session 1
    session1 = RecommendationSession(user_id="persistent_user")
    dish = session1.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer"],
    )
    session1.reject_current(dish["name"], store)

    # New session, same user -> rejected dish should still be excluded
    session2 = RecommendationSession(user_id="persistent_user")
    for _ in range(len(SAMPLE_DISHES)):
        next_dish = session2.get_next_recommendation(
            SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
            user_ingredients=["paneer"],
        )
        if next_dish is None:
            break
        assert next_dish["name"] != dish["name"]

# ── accept_current ─────────────────────────────────────────────────────────────

def test_accept_records_permanent_like(session, scorer, store):
    dish = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer"],
    )
    session.accept_current(dish["name"], store)
    assert dish["name"] in store.get_liked_dishes("test_user")

# ── reset ───────────────────────────────────────────────────────────────────────

def test_reset_clears_shown_dishes(session, scorer, store):
    session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer"],
    )
    assert len(session.shown_dish_names) == 1
    session.reset()
    assert session.shown_dish_names == []
    assert session.rejected_this_session == []

def test_dish_can_repeat_after_reset_if_not_persistently_disliked(session, scorer, store):
    dish = session.get_next_recommendation(
        SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
        user_ingredients=["paneer"],
    )
    # No reject/accept — just reset (e.g. user closed app without reacting)
    session.reset()

    results = []
    for _ in range(len(SAMPLE_DISHES)):
        next_dish = session.get_next_recommendation(
            SAMPLE_DISHES, content_scorer=scorer, feedback_store=store,
            user_ingredients=["paneer"],
        )
        if next_dish is None:
            break
        results.append(next_dish["name"])
    assert dish["name"] in results  # can come back since it wasn't disliked

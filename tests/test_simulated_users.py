# tests/test_simulated_users.py

import pytest
from pathlib import Path
from src.simulated_users import (
    PERSONAS,
    run_persona_sessions,
    run_all_personas,
    _is_quick_or_easy,
    _is_healthy,
    _is_comfort_traditional,
)
from src.content_scorer import ContentScorer
from src.feedback_store import FeedbackStore

SAMPLE_DISHES = [
    {
        "name": "Poha",
        "ingredients": ["flattened rice", "onion", "mustard seeds", "turmeric"],
        "region": "West", "state": "Maharashtra", "flavor_profile": "savory",
        "meal_type": "snack", "prep_time_mins": 15,
        "dietary_tags": ["vegetarian", "quick", "easy"],
    },
    {
        "name": "Mutton Rogan Josh",
        "ingredients": ["mutton", "yogurt", "onion", "kashmiri chili"],
        "region": "North", "state": "Kashmir", "flavor_profile": "spicy",
        "meal_type": "dinner", "prep_time_mins": 90,
        "dietary_tags": ["non-vegetarian", "comfort", "traditional"],
    },
    {
        "name": "Sprouts Salad",
        "ingredients": ["sprouts", "tomato", "cucumber", "lemon"],
        "region": "West", "state": "Gujarat", "flavor_profile": "savory",
        "meal_type": "snack", "prep_time_mins": 10,
        "dietary_tags": ["vegetarian", "healthy", "high-protein"],
    },
    {
        "name": "Rasgulla",
        "ingredients": ["chenna", "sugar", "water"],
        "region": "East", "state": "West Bengal", "flavor_profile": "sweet",
        "meal_type": "snack", "prep_time_mins": 45,
        "dietary_tags": ["vegetarian", "dessert"],
    },
    {
        "name": "Dal Makhani",
        "ingredients": ["lentils", "butter", "cream", "tomato"],
        "region": "North", "state": "Punjab", "flavor_profile": "spicy",
        "meal_type": "dinner", "prep_time_mins": 60,
        "dietary_tags": ["vegetarian", "comfort", "traditional"],
    },
]


@pytest.fixture
def scorer():
    return ContentScorer(SAMPLE_DISHES)


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=tmp_path / "test_sim.db")


# ── ground truth rule functions ───────────────────────────────────────────────

def test_is_quick_or_easy_true_for_tagged_dish():
    assert _is_quick_or_easy(SAMPLE_DISHES[0]) is True  # Poha: quick+easy tags

def test_is_quick_or_easy_true_for_low_prep_time():
    dish = {"dietary_tags": [], "prep_time_mins": 10}
    assert _is_quick_or_easy(dish) is True

def test_is_quick_or_easy_false_for_long_untagged_dish():
    assert _is_quick_or_easy(SAMPLE_DISHES[1]) is False  # Mutton, 90 min, no quick tag

def test_is_healthy_true_for_tagged_dish():
    assert _is_healthy(SAMPLE_DISHES[2]) is True  # Sprouts Salad

def test_is_healthy_false_for_dessert():
    assert _is_healthy(SAMPLE_DISHES[3]) is False  # Rasgulla

def test_is_comfort_traditional_true():
    assert _is_comfort_traditional(SAMPLE_DISHES[4]) is True  # Dal Makhani

def test_is_comfort_traditional_false_for_salad():
    assert _is_comfort_traditional(SAMPLE_DISHES[2]) is False

# ── PERSONAS config sanity ─────────────────────────────────────────────────────

def test_personas_list_has_at_least_three():
    assert len(PERSONAS) >= 3

def test_each_persona_has_required_fields():
    for p in PERSONAS:
        assert p.name
        assert p.user_type
        assert isinstance(p.sample_ingredients, list)
        assert callable(p.likes_rule)

def test_persona_names_are_unique():
    names = [p.name for p in PERSONAS]
    assert len(names) == len(set(names))

# ── run_persona_sessions ───────────────────────────────────────────────────────

def test_run_persona_sessions_returns_requested_count(scorer, store):
    persona = PERSONAS[0]
    records = run_persona_sessions(
        persona, SAMPLE_DISHES, scorer, store, num_sessions=5,
    )
    assert len(records) <= 5  # may be fewer if catalogue exhausted

def test_run_persona_sessions_same_user_id_throughout(scorer, store):
    persona = PERSONAS[0]
    records = run_persona_sessions(
        persona, SAMPLE_DISHES, scorer, store, num_sessions=3,
    )
    user_ids = {r.user_id for r in records}
    assert len(user_ids) == 1
    assert user_ids.pop() == f"sim_{persona.name}"

def test_run_persona_sessions_records_have_correct_shape(scorer, store):
    persona = PERSONAS[0]
    records = run_persona_sessions(
        persona, SAMPLE_DISHES, scorer, store, num_sessions=3,
    )
    for r in records:
        assert isinstance(r.was_accepted, bool)
        assert isinstance(r.recommended_dish, str)

def test_run_persona_sessions_deterministic_with_same_seed(scorer, store, tmp_path):
    persona = PERSONAS[0]
    store1 = FeedbackStore(db_path=tmp_path / "a.db")
    store2 = FeedbackStore(db_path=tmp_path / "b.db")

    records1 = run_persona_sessions(persona, SAMPLE_DISHES, scorer, store1, num_sessions=5, seed=1)
    records2 = run_persona_sessions(persona, SAMPLE_DISHES, scorer, store2, num_sessions=5, seed=1)

    dishes1 = [r.recommended_dish for r in records1]
    dishes2 = [r.recommended_dish for r in records2]
    assert dishes1 == dishes2

# ── run_all_personas ────────────────────────────────────────────────────────────

def test_run_all_personas_covers_every_persona(scorer, store):
    results = run_all_personas(SAMPLE_DISHES, scorer, store, num_sessions_per_persona=3)
    assert set(results["per_persona"].keys()) == {p.name for p in PERSONAS}

def test_run_all_personas_combined_has_train_test_overall(scorer, store):
    results = run_all_personas(SAMPLE_DISHES, scorer, store, num_sessions_per_persona=3)
    assert set(results["combined"].keys()) == {"train", "test", "overall"}

def test_run_all_personas_per_persona_results_well_formed(scorer, store):
    results = run_all_personas(SAMPLE_DISHES, scorer, store, num_sessions_per_persona=3)
    for persona_name, result in results["per_persona"].items():
        assert "precision_at_1" in result["overall"]
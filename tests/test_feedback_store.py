# tests/test_feedback_store.py

import pytest
from pathlib import Path
from src.feedback_store import FeedbackStore


@pytest.fixture
def store(tmp_path):
    """Fresh, isolated SQLite DB per test (tmp_path is pytest's built-in temp dir)."""
    db_path = tmp_path / "test_feedback.db"
    return FeedbackStore(db_path=db_path)


# ── add_feedback ───────────────────────────────────────────────────────────────

def test_add_feedback_thumbs_up(store):
    result = store.add_feedback("u1", "Dal Makhani", 1)
    assert result["rating"] == 1
    assert result["dish_name"] == "Dal Makhani"

def test_add_feedback_thumbs_down(store):
    result = store.add_feedback("u1", "Rasgulla", -1)
    assert result["rating"] == -1

def test_add_feedback_invalid_rating_raises(store):
    with pytest.raises(ValueError):
        store.add_feedback("u1", "Dal Makhani", 0)

def test_add_feedback_invalid_rating_5_raises(store):
    with pytest.raises(ValueError):
        store.add_feedback("u1", "Dal Makhani", 5)

# ── get_user_history ───────────────────────────────────────────────────────────

def test_get_user_history_empty_for_new_user(store):
    assert store.get_user_history("ghost_user") == []

def test_get_user_history_returns_all_events(store):
    store.add_feedback("u1", "Dal Makhani", 1)
    store.add_feedback("u1", "Rasgulla", -1)
    history = store.get_user_history("u1")
    assert len(history) == 2

def test_get_user_history_oldest_first(store):
    store.add_feedback("u1", "Dish A", 1)
    store.add_feedback("u1", "Dish B", 1)
    history = store.get_user_history("u1")
    assert history[0]["dish_name"] == "Dish A"
    assert history[1]["dish_name"] == "Dish B"

def test_get_user_history_isolated_per_user(store):
    store.add_feedback("u1", "Dish A", 1)
    store.add_feedback("u2", "Dish B", 1)
    assert len(store.get_user_history("u1")) == 1
    assert len(store.get_user_history("u2")) == 1

# ── get_liked_dishes / get_disliked_dishes ──────────────────────────────────────

def test_get_liked_dishes(store):
    store.add_feedback("u1", "Dal Makhani", 1)
    store.add_feedback("u1", "Rasgulla", -1)
    store.add_feedback("u1", "Shahi Paneer", 1)
    liked = store.get_liked_dishes("u1")
    assert set(liked) == {"Dal Makhani", "Shahi Paneer"}

def test_get_disliked_dishes(store):
    store.add_feedback("u1", "Dal Makhani", 1)
    store.add_feedback("u1", "Rasgulla", -1)
    disliked = store.get_disliked_dishes("u1")
    assert disliked == ["Rasgulla"]

def test_most_recent_rating_wins_for_same_dish(store):
    # User initially liked it, then changed their mind
    store.add_feedback("u1", "Dal Makhani", 1)
    store.add_feedback("u1", "Dal Makhani", -1)
    assert store.get_liked_dishes("u1") == []
    assert store.get_disliked_dishes("u1") == ["Dal Makhani"]

# ── feedback_count ──────────────────────────────────────────────────────────────

def test_feedback_count_zero_for_new_user(store):
    assert store.feedback_count("ghost_user") == 0

def test_feedback_count_counts_all_events_including_changed_mind(store):
    store.add_feedback("u1", "Dal Makhani", 1)
    store.add_feedback("u1", "Dal Makhani", -1)  # same dish, different rating
    store.add_feedback("u1", "Rasgulla", 1)
    # feedback_count counts raw events, not unique dishes
    assert store.feedback_count("u1") == 3

# ── get_user_item_matrix ──────────────────────────────────────────────────────

def test_user_item_matrix_empty_when_no_data(store):
    matrix = store.get_user_item_matrix()
    assert matrix.empty

def test_user_item_matrix_shape(store):
    store.add_feedback("u1", "Dal Makhani", 1)
    store.add_feedback("u1", "Rasgulla", -1)
    store.add_feedback("u2", "Dal Makhani", 1)
    matrix = store.get_user_item_matrix()
    assert "u1" in matrix.index
    assert "u2" in matrix.index
    assert "Dal Makhani" in matrix.columns
    assert "Rasgulla" in matrix.columns

def test_user_item_matrix_missing_entries_are_zero(store):
    store.add_feedback("u1", "Dal Makhani", 1)
    store.add_feedback("u2", "Rasgulla", -1)
    matrix = store.get_user_item_matrix()
    # u1 never rated Rasgulla -> should be 0, not NaN
    assert matrix.loc["u1", "Rasgulla"] == 0

def test_user_item_matrix_uses_latest_rating(store):
    store.add_feedback("u1", "Dal Makhani", 1)
    store.add_feedback("u1", "Dal Makhani", -1)
    matrix = store.get_user_item_matrix()
    assert matrix.loc["u1", "Dal Makhani"] == -1

# ── persistence across instances (same db_path) ────────────────────────────────

def test_data_persists_across_store_instances(tmp_path):
    db_path = tmp_path / "persist_test.db"
    store1 = FeedbackStore(db_path=db_path)
    store1.add_feedback("u1", "Dal Makhani", 1)

    store2 = FeedbackStore(db_path=db_path)  # new instance, same file
    history = store2.get_user_history("u1")
    assert len(history) == 1

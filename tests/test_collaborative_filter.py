# tests/test_collaborative_filter.py

import pytest
import pandas as pd
from src.collaborative_filter import CollaborativeFilter


def make_matrix(data: dict) -> pd.DataFrame:
    """Helper: build a user-item matrix from a nested dict for readable tests."""
    return pd.DataFrame(data).T.fillna(0)


# ── has_sufficient_data ─────────────────────────────────────────────────────────

def test_no_data_returns_false():
    cf = CollaborativeFilter(pd.DataFrame())
    assert cf.has_sufficient_data("u1") is False

def test_unknown_user_returns_false():
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1, "Rasgulla": -1},
    })
    cf = CollaborativeFilter(matrix)
    assert cf.has_sufficient_data("ghost_user") is False

def test_single_user_no_neighbors_returns_false():
    # Only one user in the whole matrix -> nobody to compare against
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1, "Rasgulla": -1},
    })
    cf = CollaborativeFilter(matrix)
    assert cf.has_sufficient_data("u1") is False

def test_two_similar_users_returns_true():
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1, "Rasgulla": -1, "Poha": 1},
        "u2": {"Dal Makhani": 1, "Rasgulla": -1, "Poha": 1},
    })
    cf = CollaborativeFilter(matrix)
    assert cf.has_sufficient_data("u1") is True

def test_completely_opposite_users_returns_false():
    # u1 and u2 have perfectly opposite tastes -> negative similarity, filtered out
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1, "Rasgulla": 1},
        "u2": {"Dal Makhani": -1, "Rasgulla": -1},
    })
    cf = CollaborativeFilter(matrix)
    assert cf.has_sufficient_data("u1") is False

# ── recommend ────────────────────────────────────────────────────────────────

def test_recommend_empty_matrix_returns_empty():
    cf = CollaborativeFilter(pd.DataFrame())
    assert cf.recommend("u1") == []

def test_recommend_unknown_user_returns_empty():
    matrix = make_matrix({"u1": {"Dal Makhani": 1}})
    cf = CollaborativeFilter(matrix)
    assert cf.recommend("ghost_user") == []

def test_recommend_suggests_dish_liked_by_similar_user():
    # u1 and u2 have identical taste on shared dishes.
    # u2 also liked "Shahi Paneer", which u1 hasn't rated yet.
    # u1 should get Shahi Paneer recommended.
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1, "Rasgulla": -1},
        "u2": {"Dal Makhani": 1, "Rasgulla": -1, "Shahi Paneer": 1},
    })
    cf = CollaborativeFilter(matrix)
    recs = cf.recommend("u1")
    names = [r["dish_name"] for r in recs]
    assert "Shahi Paneer" in names

def test_recommend_excludes_already_rated_dishes():
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1, "Rasgulla": -1},
        "u2": {"Dal Makhani": 1, "Rasgulla": -1, "Shahi Paneer": 1},
    })
    cf = CollaborativeFilter(matrix)
    recs = cf.recommend("u1")
    names = [r["dish_name"] for r in recs]
    assert "Dal Makhani" not in names
    assert "Rasgulla" not in names

def test_recommend_respects_top_k():
    matrix = make_matrix({
        "u1": {},
        "u2": {"A": 1, "B": 1, "C": 1, "D": 1},
    })
    # Make u1 similar to u2 by sharing at least one rating
    matrix.loc["u1", "A"] = 1
    cf = CollaborativeFilter(matrix)
    recs = cf.recommend("u1", top_k=2)
    assert len(recs) <= 2

def test_recommend_returns_cf_score_field():
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1},
        "u2": {"Dal Makhani": 1, "Shahi Paneer": 1},
    })
    cf = CollaborativeFilter(matrix)
    recs = cf.recommend("u1")
    assert all("cf_score" in r for r in recs)

def test_recommend_negative_similarity_excluded():
    # u2 has opposite taste to u1 -> shouldn't influence u1's recommendations at all
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1, "Rasgulla": 1},
        "u2": {"Dal Makhani": -1, "Rasgulla": -1, "Poha": 1},
    })
    cf = CollaborativeFilter(matrix)
    recs = cf.recommend("u1")
    assert recs == []  # no positively-similar neighbors exist

def test_recommend_sorted_descending_by_score():
    matrix = make_matrix({
        "u1": {"Dal Makhani": 1},
        "u2": {"Dal Makhani": 1, "A": 1, "B": -1},
        "u3": {"Dal Makhani": 1, "A": 1, "B": 1},
    })
    cf = CollaborativeFilter(matrix)
    recs = cf.recommend("u1")
    scores = [r["cf_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)

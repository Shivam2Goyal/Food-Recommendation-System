# tests/test_api.py

"""
API tests using FastAPI's TestClient (no need to run a live server).

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app, load_data


@pytest.fixture(scope="module", autouse=True)
def setup_data():
    # Manually trigger the startup event since TestClient doesn't
    # always fire it depending on pytest/fastapi version
    load_data()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dishes_loaded"] > 0


def test_recommend_basic_request(client):
    response = client.post("/recommend", json={
        "user_ingredients": ["paneer", "tomato", "onion"],
        "top_k": 5,
    })
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert body["count"] <= 5
    assert len(body["results"]) == body["count"]


def test_recommend_empty_ingredients_returns_400(client):
    response = client.post("/recommend", json={
        "user_ingredients": [],
    })
    assert response.status_code == 400


def test_recommend_missing_ingredients_field_returns_422(client):
    # user_ingredients is required by the pydantic model
    response = client.post("/recommend", json={})
    assert response.status_code == 422


def test_recommend_response_matches_contract_shape(client):
    response = client.post("/recommend", json={
        "user_ingredients": ["paneer", "butter"],
        "top_k": 1,
    })
    assert response.status_code == 200
    dish = response.json()["results"][0]

    expected_fields = {
        "dish_name", "cuisine", "region", "state", "meal_type",
        "flavor_profile", "ingredients_required", "ingredients_available",
        "missing_ingredients", "coverage_score", "dietary_tags",
        "prep_time_mins",
    }
    assert expected_fields.issubset(dish.keys())


def test_recommend_with_user_type_and_meal_type(client):
    response = client.post("/recommend", json={
        "user_ingredients": ["paneer", "tomato"],
        "user_type": "bachelor",
        "requested_meal_type": "dinner",
        "max_prep_time": 40,
        "top_k": 3,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["count"] <= 3


def test_recommend_with_liked_dishes(client):
    # First get any dish name from the catalogue
    initial = client.post("/recommend", json={
        "user_ingredients": ["onion"],
        "top_k": 1,
    }).json()
    liked_name = initial["results"][0]["dish_name"]

    response = client.post("/recommend", json={
        "user_ingredients": ["onion"],
        "liked_dish_names": [liked_name],
        "top_k": 5,
    })
    assert response.status_code == 200
    names = [d["dish_name"] for d in response.json()["results"]]
    assert liked_name not in names  # shouldn't recommend what they already liked


def test_recommend_coverage_score_in_valid_range(client):
    response = client.post("/recommend", json={
        "user_ingredients": ["paneer", "tomato", "onion", "garlic"],
        "top_k": 10,
    })
    for dish in response.json()["results"]:
        assert 0.0 <= dish["coverage_score"] <= 1.0

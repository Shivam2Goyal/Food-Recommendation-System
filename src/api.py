# src/api.py

"""
Stub FastAPI endpoint for the food recommender.

This is what Intern B's LLM layer calls over HTTP. Returns recommendations
in the exact shape defined in contract.json.

Run with:
    uvicorn src.api:app --reload --port 8000

Then test at:
    http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""

import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.combined_ranker import combined_rank
from src.content_scorer import ContentScorer

app = FastAPI(title="Food Recommender API", version="0.1.0")

# ── Load dataset + fit content scorer once at startup ─────────────────────────

DISHES: List[dict] = []
CONTENT_SCORER: Optional[ContentScorer] = None


@app.on_event("startup")
def load_data():
    global DISHES, CONTENT_SCORER
    clean_path = Path("data/clean/dishes.json")
    if not clean_path.exists():
        raise RuntimeError(
            "data/clean/dishes.json not found. Run `python -m src.clean` first."
        )
    with open(clean_path) as f:
        DISHES = json.load(f)
    CONTENT_SCORER = ContentScorer(DISHES)
    print(f"Loaded {len(DISHES)} dishes. Content scorer fitted.")


# ── Request / response schemas (mirrors contract.json) ────────────────────────

class RecommendRequest(BaseModel):
    user_ingredients: List[str]
    user_type: Optional[str] = None          # bachelor | student | family | health-focused
    requested_meal_type: Optional[str] = None  # breakfast | lunch | dinner | snack
    max_prep_time: Optional[int] = None
    liked_dish_names: Optional[List[str]] = []
    top_k: Optional[int] = 10


class DishResponse(BaseModel):
    dish_name: str
    cuisine: str
    region: Optional[str] = None
    state: Optional[str] = None
    meal_type: str
    flavor_profile: Optional[str] = None
    ingredients_required: List[str]
    ingredients_available: List[str]
    missing_ingredients: List[str]
    coverage_score: float
    dietary_tags: List[str]
    prep_time_mins: int


class RecommendResponse(BaseModel):
    results: List[DishResponse]
    count: int


# ── Helper: map internal dish dict -> contract.json shape ─────────────────────

def to_contract_shape(dish: dict) -> dict:
    flavor = dish.get("flavor_profile")
    if not flavor:
        # fallback: pull from dietary_tags if flavor wasn't stored as its own field
        tags = dish.get("dietary_tags", [])
        known_flavors = {"sweet", "spicy", "savory", "bitter", "sour"}
        flavor = next((t for t in tags if t in known_flavors), None)

    return {
        "dish_name": dish["name"],
        "cuisine": dish.get("cuisine", "Indian"),
        "region": dish.get("region"),
        "state": dish.get("state"),
        "meal_type": dish["meal_type"],
        "flavor_profile": flavor,
        "ingredients_required": dish["ingredients"],
        "ingredients_available": dish.get("ingredients_available", []),
        "missing_ingredients": dish.get("missing_ingredients", []),
        "coverage_score": dish.get("coverage_score", 0.0),
        "dietary_tags": dish.get("dietary_tags", []),
        "prep_time_mins": dish.get("prep_time_mins", 30),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "dishes_loaded": len(DISHES)}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not DISHES or CONTENT_SCORER is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded yet")

    if not req.user_ingredients:
        raise HTTPException(status_code=400, detail="user_ingredients cannot be empty")

    results = combined_rank(
        DISHES,
        content_scorer=CONTENT_SCORER,
        user_ingredients=req.user_ingredients,
        liked_dish_names=req.liked_dish_names or [],
        user_type=req.user_type,
        requested_meal_type=req.requested_meal_type,
        max_prep_time=req.max_prep_time,
        top_k=req.top_k or 10,
    )

    shaped = [to_contract_shape(d) for d in results]

    return {"results": shaped, "count": len(shaped)}
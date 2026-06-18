# src/schema.py

from dataclasses import dataclass, field
from typing import List

@dataclass
class Dish:
    name: str
    cuisine: str                    # e.g. "Indian", "Italian"
    ingredients: List[str]          # normalised ingredient names
    meal_type: str                  # "breakfast" | "snack" | "dinner" | "lunch"
    prep_time_mins: int
    dietary_tags: List[str]         # e.g. ["vegetarian", "gluten-free"]
    user_type_affinity: List[str]   # e.g. ["bachelor", "family", "health-focused"]

# The JSON contract object returned by the recommender to Intern B
CONTRACT_SCHEMA = {
    "dish_name": str,
    "cuisine": str,
    "ingredients_required": list,   # all ingredients the dish needs
    "ingredients_available": list,  # subset the user already has
    "user_type": str,
    "dietary_tags": list,
    # optional fields:
    "missing_ingredients": list,    # flagged when coverage < 40%
    "coverage_score": float,        # 0.0 – 1.0
}
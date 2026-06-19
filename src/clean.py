# src/clean.py

import pandas as pd
import json
from pathlib import Path
from src.synonyms import normalise

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Map the dataset's 'course' values → our standard meal_type
COURSE_TO_MEAL_TYPE = {
    "breakfast": "breakfast",
    "main course": "dinner",
    "dessert": "snack",
    "snack": "snack",
    "side dish": "dinner",
    "starter": "snack",
    "lunch": "lunch",
    "dinner": "dinner",
    "beverage": "snack",
    "bread": "breakfast",
    "rice": "lunch",
    "biryani": "lunch",
}

# Values that mean "no data" across any column, as either int or string —
# the source CSV is inconsistent about which type -1 shows up as depending
# on the column, so we check both forms everywhere we read a cell.
NULL_SENTINELS = {"-1", "nan", "none", ""}

def is_null_value(value) -> bool:
    return str(value).strip().lower() in NULL_SENTINELS

def parse_ingredients(raw) -> list:
    """Parse comma-separated ingredient string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []

# Heuristic tags derived from ingredients/name/prep_time, since the source
# CSV only gives us diet + flavor_profile. These are intentionally simple,
# hand-written rules — good enough to give personas (Week 4) real signal
# to work with, not meant to be a precise classifier.
HEAVY_DAIRY_OR_FRIED_KEYWORDS = {
    "ghee", "butter", "cream", "khoya", "paneer", "sugar", "oil", "deep fried",
}

# Words that should NEVER count as "healthy" even if a healthy keyword also
# matches — sugar/frying dominates the nutritional profile of the dish.
# This is a hard exclusion, checked before any healthy keyword inclusion.
UNHEALTHY_OVERRIDE_KEYWORDS = {
    "sugar", "sugar syrup", "jaggery", "ghee", "deep fried", "fried",
    "khoya", "condensed milk", "syrup",
}

# Ingredients that must be a MAIN ingredient (not just "yogurt as a minor
# binding agent in a dessert") to count as a healthy signal. Kept narrower
# than before — generic dairy (yogurt, milk) removed since they appear in
# both healthy AND unhealthy dishes and aren't a reliable signal alone.
HEALTHY_KEYWORDS = {
    "sprouts", "spinach", "salad", "cucumber", "vegetable", "lauki",
    "broccoli", "carrot", "beans", "gourd", "tomato soup", "karela",
    "palak",
}
TRADITIONAL_DESSERT_OR_CURRY_KEYWORDS = {
    "halwa", "kheer", "ladoo", "barfi", "curry", "biryani", "dal", "sabzi",
}

def infer_extra_tags(name: str, ingredients: list, prep_time: int) -> list:
    """Backfill tags the raw dataset doesn't provide: quick, easy, healthy,
    comfort, traditional. Based on ingredient keywords, dish name, and
    prep time — not from any source column."""
    tags = []
    joined = (name + " " + " ".join(ingredients)).lower()

    if prep_time <= 20:
        tags.append("quick")
    if prep_time <= 20 and len(ingredients) <= 5:
        tags.append("easy")

    # Hard exclusion first: sugar/fried dishes are never "healthy" no
    # matter what else is in the ingredient list (e.g. yogurt-based fried
    # sweets like Balu Shahi should not qualify just because of yogurt).
    is_unhealthy = any(kw in joined for kw in UNHEALTHY_OVERRIDE_KEYWORDS)
    if not is_unhealthy and any(kw in joined for kw in HEALTHY_KEYWORDS):
        tags.append("healthy")


    if any(kw in joined for kw in HEAVY_DAIRY_OR_FRIED_KEYWORDS):
        tags.append("comfort")

    if any(kw in joined for kw in TRADITIONAL_DESSERT_OR_CURRY_KEYWORDS):
        tags.append("traditional")

    return tags

def clean_indian_food(input_csv: str, output_json: str):
    df = pd.read_csv(RAW_DIR / input_csv)
    print(f"Loaded {len(df)} rows")
    print("Columns:", df.columns.tolist())

    dishes = []
    seen_names = set()

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        # Ingredients
        raw_ingredients = parse_ingredients(row.get("ingredients", ""))
        normalised_ingredients = [normalise(i) for i in raw_ingredients if i]

        # meal_type from 'course' column
        course_raw = str(row.get("course", "")).strip().lower()
        meal_type = COURSE_TO_MEAL_TYPE.get(course_raw, "dinner")

        # prep_time: use prep_time, fallback to cook_time, fallback to 30
        prep_time_raw = row.get("prep_time", None)
        if is_null_value(prep_time_raw):
            prep_time_raw = row.get("cook_time", 30)
        try:
            prep_time = int(prep_time_raw)
            if prep_time <= 0 or prep_time > 300:
                prep_time = 30
        except (ValueError, TypeError):
            prep_time = 30

        # dietary_tags from 'diet' column — skip null sentinels in ANY form
        diet_raw = row.get("diet", "")
        dietary_tags = []
        if not is_null_value(diet_raw):
            dietary_tags.append(str(diet_raw).strip().lower().replace(" ", "-"))

        # flavor_profile as an extra tag — same null check
        flavor_raw = row.get("flavor_profile", "")
        flavor_profile = None
        if not is_null_value(flavor_raw):
            flavor_profile = str(flavor_raw).strip().lower()
            dietary_tags.append(flavor_profile)

        # Backfill quick/easy/healthy/comfort/traditional from heuristics,
        # since the source CSV has no equivalent column
        dietary_tags.extend(
            infer_extra_tags(name, normalised_ingredients, prep_time)
        )

        # cuisine: use state → region as a proxy
        state_raw = row.get("state", "")
        region_raw = row.get("region", "")
        state = None if is_null_value(state_raw) else str(state_raw).strip()
        region = None if is_null_value(region_raw) else str(region_raw).strip()
        cuisine = "Indian"

        dish = {
            "name": name,
            "cuisine": cuisine,
            "state": state,          # bonus field — useful for filtering later
            "region": region,        # bonus field
            "ingredients": normalised_ingredients,
            "meal_type": meal_type,
            "flavor_profile": flavor_profile,
            "prep_time_mins": prep_time,
            "dietary_tags": dietary_tags,
            "user_type_affinity": [],
        }
        dishes.append(dish)

    print(f"Clean dishes: {len(dishes)}")

    with open(CLEAN_DIR / output_json, "w") as f:
        json.dump(dishes, f, indent=2)

    print(f"Saved → data/clean/{output_json}")
    return dishes

if __name__ == "__main__":
    clean_indian_food("indian_food.csv", "dishes.json")
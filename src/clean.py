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

def parse_ingredients(raw) -> list:
    """Parse comma-separated ingredient string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []

def clean_indian_food(input_csv: str, output_json: str):
    df = pd.read_csv(RAW_DIR / input_csv)
    print(f"Loaded {len(df)} rows")
    print("Columns:", df.columns.tolist())

    # Replace -1 (null sentinel) with NaN across the board
    df.replace(-1, pd.NA, inplace=True)

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
        prep_time = row.get("prep_time", pd.NA)
        if pd.isna(prep_time):
            prep_time = row.get("cook_time", 30)
        try:
            prep_time = int(prep_time)
            if prep_time <= 0 or prep_time > 300:
                prep_time = 30
        except (ValueError, TypeError):
            prep_time = 30

        # dietary_tags from 'diet' column
        diet = str(row.get("diet", "")).strip().lower()
        dietary_tags = [diet] if diet and diet != "nan" else []

        # flavor_profile as an extra tag if present
        flavor = str(row.get("flavor_profile", "")).strip().lower()
        if flavor and flavor != "nan":
            dietary_tags.append(flavor)

        # cuisine: use state → region as a proxy
        state = str(row.get("state", "")).strip()
        region = str(row.get("region", "")).strip()
        cuisine = f"Indian"  # always Indian; state/region stored separately
        if state == "nan": state = None
        if region == "nan": region = None

        dish = {
            "name": name,
            "cuisine": cuisine,
            "state": state,          # bonus field — useful for filtering later
            "region": region,        # bonus field
            "ingredients": normalised_ingredients,
            "meal_type": meal_type,
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
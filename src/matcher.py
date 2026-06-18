# src/matcher.py

from rapidfuzz import fuzz
from src.synonyms import normalise

# Minimum similarity score (0-100) to count as a match
FUZZY_THRESHOLD = 75

# Close-enough pairs that are acceptable substitutes
# (these are treated as matches even if fuzzy score is borderline)
SUBSTITUTES = {
    ("paneer", "tofu"),
    ("ghee", "butter"),
    ("ghee", "clarified butter"),
    ("cream", "fresh cream"),
    ("curd", "yogurt"),
    ("dahi", "yogurt"),
    ("whole milk", "milk"),
    ("refined oil", "oil"),
    ("mustard oil", "oil"),
    ("coconut oil", "oil"),
    ("gram flour", "chickpea flour"),
    ("maida", "all-purpose flour"),
    ("atta", "whole wheat flour"),
    ("chenna", "cottage cheese"),
    ("khoya", "milk powder"),
    ("mawa", "milk powder"),
}

def _is_substitute(a: str, b: str) -> bool:
    """Check if two ingredients are acceptable substitutes for each other."""
    pair = (a, b)
    reverse = (b, a)
    return pair in SUBSTITUTES or reverse in SUBSTITUTES

def match_ingredient(user_ingredient: str, dish_ingredient: str) -> bool:
    """
    Returns True if user_ingredient satisfies dish_ingredient.
    Uses: exact match → synonym normalisation → fuzzy match → substitute check.
    """
    u = normalise(user_ingredient)
    d = normalise(dish_ingredient)

    # 1. Exact match after normalisation
    if u == d:
        return True

    # 2. Substring match (e.g. "basmati rice" satisfies "rice")
    if u in d or d in u:
        return True

    # 3. Fuzzy match
    score = fuzz.token_sort_ratio(u, d)
    if score >= FUZZY_THRESHOLD:
        return True

    # 4. Substitute check
    if _is_substitute(u, d):
        return True

    return False

def compute_coverage(user_ingredients: list[str], dish_ingredients: list[str]) -> dict:
    """
    Given what the user has and what the dish needs,
    return a coverage breakdown.

    Returns:
        {
            "coverage_score": float,          # 0.0 - 1.0
            "ingredients_available": list,    # dish ingredients the user CAN cover
            "missing_ingredients": list,      # dish ingredients the user CANNOT cover
        }
    """
    if not dish_ingredients:
        return {
            "coverage_score": 0.0,
            "ingredients_available": [],
            "missing_ingredients": [],
        }

    available = []
    missing = []

    for dish_ing in dish_ingredients:
        matched = any(
            match_ingredient(user_ing, dish_ing)
            for user_ing in user_ingredients
        )
        if matched:
            available.append(dish_ing)
        else:
            missing.append(dish_ing)

    coverage_score = round(len(available) / len(dish_ingredients), 3)

    return {
        "coverage_score": coverage_score,
        "ingredients_available": available,
        "missing_ingredients": missing,
    }


# ── Quick manual test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    user_has = ["tomatoes", "dahi", "ghee", "onion", "garlic"]
    dish_needs = ["tomato", "yogurt", "butter", "onion", "garlic", "cumin", "turmeric"]

    result = compute_coverage(user_has, dish_needs)

    print("User has:    ", user_has)
    print("Dish needs:  ", dish_needs)
    print()
    print(f"Coverage:    {result['coverage_score']} ({int(result['coverage_score']*100)}%)")
    print(f"Available:   {result['ingredients_available']}")
    print(f"Missing:     {result['missing_ingredients']}")

# src/baseline_ranker.py

"""
Cold-start baseline ranker.

For a brand-new user with no history, rank dishes using only:
  - ingredient coverage (do they have what's needed?)
  - meal_type match (did they ask for breakfast/lunch/dinner/snack?)
  - user_type affinity (bachelor/student/family/health-focused — soft boost)
  - prep_time fit (does it fit their stated time budget?)

No ML, no learning — just transparent, explainable rules.
This becomes the fallback whenever personalization (later weeks) has no
signal to work with.
"""

from src.matcher import compute_coverage

# Weights — tune these later once Intern B's LLM layer is in place
WEIGHTS = {
    "coverage": 0.55,
    "meal_type": 0.20,
    "prep_time": 0.15,
    "user_type": 0.10,
}

# Soft affinity boost: dishes tagged with these dietary_tags get a small
# bonus for matching user_type, even though we have no real personalization yet.
USER_TYPE_AFFINITY_HINTS = {
    "bachelor":        {"quick", "easy", "snack"},
    "student":         {"quick", "easy", "snack", "budget"},
    "family":          {"comfort", "traditional"},
    "health-focused":  {"healthy", "low-fat", "high-protein", "gluten-free"},
}

def score_meal_type(dish: dict, requested_meal_type: str | None) -> float:
    """1.0 if exact match, 0.5 if no preference given, 0.0 if mismatch."""
    if not requested_meal_type:
        return 0.5  # neutral — user didn't specify
    return 1.0 if dish["meal_type"] == requested_meal_type else 0.0

def score_prep_time(dish: dict, max_prep_time: int | None) -> float:
    """
    1.0 if well within budget, scaling down to 0.0 if it blows the budget.
    No budget given → neutral 0.5.
    """
    if max_prep_time is None:
        return 0.5
    dish_time = dish.get("prep_time_mins", 30)
    if dish_time <= max_prep_time:
        return 1.0
    # Linear falloff: fully over by 2x budget = 0 score
    overage_ratio = (dish_time - max_prep_time) / max_prep_time
    return max(0.0, 1.0 - overage_ratio)

def score_user_type(dish: dict, user_type: str | None) -> float:
    """Soft boost if dish's dietary_tags overlap with user_type hints."""
    if not user_type or user_type not in USER_TYPE_AFFINITY_HINTS:
        return 0.5
    hints = USER_TYPE_AFFINITY_HINTS[user_type]
    dish_tags = set(t.lower() for t in dish.get("dietary_tags", []))
    overlap = hints & dish_tags
    if overlap:
        return 1.0
    return 0.3  # no penalty, just no bonus

def rank_dishes(
    dishes: list[dict],
    user_ingredients: list[str],
    user_type: str | None = None,
    requested_meal_type: str | None = None,
    max_prep_time: int | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Cold-start ranking. Returns top_k dishes with scores and coverage info attached.
    """
    scored = []

    for dish in dishes:
        coverage = compute_coverage(user_ingredients, dish["ingredients"])

        s_coverage = coverage["coverage_score"]
        s_meal = score_meal_type(dish, requested_meal_type)
        s_prep = score_prep_time(dish, max_prep_time)
        s_user = score_user_type(dish, user_type)

        final_score = (
            WEIGHTS["coverage"] * s_coverage +
            WEIGHTS["meal_type"] * s_meal +
            WEIGHTS["prep_time"] * s_prep +
            WEIGHTS["user_type"] * s_user
        )

        scored.append({
            **dish,
            "coverage_score": s_coverage,
            "ingredients_available": coverage["ingredients_available"],
            "missing_ingredients": coverage["missing_ingredients"],
            "baseline_score": round(final_score, 4),
            "_score_breakdown": {
                "coverage": round(s_coverage, 3),
                "meal_type": s_meal,
                "prep_time": round(s_prep, 3),
                "user_type": s_user,
            },
        })

    scored.sort(key=lambda d: d["baseline_score"], reverse=True)
    return scored[:top_k]


# ── Quick manual test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path

    clean_path = Path("data/clean/dishes.json")
    if not clean_path.exists():
        print("Run src/clean.py first to generate data/clean/dishes.json")
        exit()

    with open(clean_path) as f:
        dishes = json.load(f)

    results = rank_dishes(
        dishes,
        user_ingredients=["paneer", "tomato", "onion", "garlic", "ghee"],
        user_type="bachelor",
        requested_meal_type="dinner",
        max_prep_time=40,
        top_k=5,
    )

    print(f"Top {len(results)} dishes:\n")
    for i, dish in enumerate(results, 1):
        print(f"{i}. {dish['name']}  (score={dish['baseline_score']})")
        print(f"   coverage={dish['coverage_score']}  missing={dish['missing_ingredients']}")
        print(f"   breakdown={dish['_score_breakdown']}")
        print()

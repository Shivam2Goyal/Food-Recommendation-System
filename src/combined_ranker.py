# src/combined_ranker.py

"""
Combined ranker.

Blends:
  - baseline_score   (from baseline_ranker.py — coverage, meal_type, prep_time, user_type)
  - content_score     (from content_scorer.py — similarity to dishes the user liked)

The blend ratio is NOT fixed. It shifts based on how much preference signal
we have about the user (i.e. how many dishes they've liked so far):

  0 liked dishes   -> 100% baseline   (pure cold-start)
  1-2 liked dishes -> mostly baseline, small content nudge
  3-4 liked dishes -> roughly even mix
  5+ liked dishes  -> mostly content, baseline still acts as a floor/filter

This is intentionally simple (a step function) so it's easy to reason about
and tune later. Replace with a smooth function or learned weights once
Intern B's feedback loop is producing real data.
"""

from src.baseline_ranker import rank_dishes
from src.content_scorer import ContentScorer


def get_blend_weights(num_liked_dishes: int) -> dict:
    """
    Returns {"baseline": w1, "content": w2} where w1 + w2 == 1.0
    """
    if num_liked_dishes == 0:
        return {"baseline": 1.0, "content": 0.0}
    elif num_liked_dishes <= 2:
        return {"baseline": 0.75, "content": 0.25}
    elif num_liked_dishes <= 4:
        return {"baseline": 0.5, "content": 0.5}
    else:
        return {"baseline": 0.3, "content": 0.7}


def combined_rank(
    dishes: list[dict],
    content_scorer: ContentScorer,
    user_ingredients: list[str],
    liked_dish_names: list[str] = None,
    user_type: str = None,
    requested_meal_type: str = None,
    max_prep_time: int = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Main entry point. Combines baseline + content scores into one ranked list.
    """
    liked_dish_names = liked_dish_names or []
    num_liked = len(liked_dish_names)
    weights = get_blend_weights(num_liked)

    # 1. Baseline scores for the WHOLE catalogue (not top_k yet — we need
    #    full scores to combine properly before truncating)
    baseline_results = rank_dishes(
        dishes,
        user_ingredients=user_ingredients,
        user_type=user_type,
        requested_meal_type=requested_meal_type,
        max_prep_time=max_prep_time,
        top_k=len(dishes),  # get all, truncate after combining
    )
    baseline_by_name = {d["name"]: d for d in baseline_results}

    # 2. Content scores — only meaningful once the user has liked something
    content_by_name = {}
    if num_liked > 0:
        try:
            content_results = content_scorer.score_against_profile(liked_dish_names)
            content_by_name = {d["name"]: d["content_score"] for d in content_results}
        except ValueError:
            # liked_dish_names didn't match anything in the catalogue — fall back
            # to pure baseline for this request
            weights = {"baseline": 1.0, "content": 0.0}

    # 3. Combine
    combined = []
    for name, dish in baseline_by_name.items():
        if name in liked_dish_names:
            continue  # don't recommend what they've already liked

        b_score = dish["baseline_score"]
        c_score = content_by_name.get(name, 0.0)  # 0 if no content signal yet

        final_score = (
            weights["baseline"] * b_score +
            weights["content"] * c_score
        )

        combined.append({
            **dish,
            "content_score": round(c_score, 4),
            "combined_score": round(final_score, 4),
            "_blend_weights": weights,
        })

    combined.sort(key=lambda d: d["combined_score"], reverse=True)
    return combined[:top_k]


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

    scorer = ContentScorer(dishes)
    liked = [dishes[0]["name"]]  # pretend user liked the first dish

    print(f"User liked: {liked}\n")

    for num_liked_label, liked_list in [("0 liked", []), ("1 liked", liked)]:
        print(f"--- {num_liked_label} ---")
        results = combined_rank(
            dishes,
            content_scorer=scorer,
            user_ingredients=["paneer", "tomato", "onion", "garlic", "ghee"],
            liked_dish_names=liked_list,
            user_type="bachelor",
            requested_meal_type="dinner",
            max_prep_time=40,
            top_k=5,
        )
        for i, d in enumerate(results, 1):
            print(f"{i}. {d['name']:30s} combined={d['combined_score']}  "
                  f"(baseline={d['baseline_score']}, content={d['content_score']}, "
                  f"weights={d['_blend_weights']})")
        print()

# src/hybrid_ranker.py

"""
Hybrid ranker — the Week 3 evolution of combined_ranker.py.

Blends THREE signals now instead of two:
  - baseline_score  (coverage + meal_type + prep_time + user_type — Week 2)
  - content_score   (similarity to liked dishes — Week 2)
  - cf_score         (collaborative filtering — Week 3)

Per the roadmap: "Weight CF higher when feedback history >= 5 interactions,
lower otherwise. This handles the cold-start/warm transition automatically."

So the blend now depends on TWO signals of "how much do we know about this user":
  1. num_liked_dishes      -> baseline vs content split (unchanged from Week 2)
  2. total_feedback_count  -> how much to trust CF on top of that

Transition stages:
  feedback_count == 0          -> pure baseline+content (Week 2 behavior, no CF)
  1 <= feedback_count < 5      -> small CF nudge (cold -> warm transition)
  feedback_count >= 5          -> CF becomes a major factor

CF only ever applies to dishes it has a score for (i.e. dishes a similar
user rated). Dishes with no CF signal get cf_score = 0, same pattern as
content_score in Week 2.
"""

from src.combined_ranker import get_blend_weights, combined_rank
from src.collaborative_filter import CollaborativeFilter
from src.content_scorer import ContentScorer
from src.feedback_store import FeedbackStore


def get_cf_weight(feedback_count: int) -> float:
    """
    Returns how much weight CF should get (0.0 - 1.0), to be carved out of
    the existing baseline+content blend. The remaining weight (1 - cf_weight)
    is split between baseline/content using the existing Week 2 logic.
    """
    if feedback_count == 0:
        return 0.0
    elif feedback_count < 5:
        return 0.15  # small nudge — CF has limited data, don't trust it much yet
    elif feedback_count < 15:
        return 0.4
    else:
        return 0.55  # plenty of history — CF is now a primary signal


def hybrid_rank(
    dishes: list[dict],
    content_scorer: ContentScorer,
    feedback_store: FeedbackStore,
    user_id: str,
    user_ingredients: list[str],
    user_type: str = None,
    requested_meal_type: str = None,
    max_prep_time: int = None,
    exclude_dish_names: list[str] = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Main entry point for Week 3+. Combines baseline, content, and CF scores.

    exclude_dish_names: dishes to hard-exclude from results regardless of
    score (used by Task 4 — rejection routing — to keep a rejected dish
    out of the candidate pool for the rest of the session).
    """
    exclude_dish_names = set(exclude_dish_names or [])

    liked_dish_names = feedback_store.get_liked_dishes(user_id)
    disliked_dish_names = feedback_store.get_disliked_dishes(user_id)
    feedback_count = feedback_store.feedback_count(user_id)

    # Never recommend something the user has already explicitly disliked
    exclude_dish_names |= set(disliked_dish_names)

    cf_weight = get_cf_weight(feedback_count)

    # 1. Get baseline+content blended scores for everything (Week 2 logic, untouched)
    base_combined = combined_rank(
        dishes,
        content_scorer=content_scorer,
        user_ingredients=user_ingredients,
        liked_dish_names=liked_dish_names,
        user_type=user_type,
        requested_meal_type=requested_meal_type,
        max_prep_time=max_prep_time,
        top_k=len(dishes),  # get all, we'll filter/truncate after CF blend
    )
    base_by_name = {d["name"]: d for d in base_combined}

    # 2. CF scores, only if we have enough data to bother
    cf_by_name = {}
    if cf_weight > 0:
        matrix = feedback_store.get_user_item_matrix()
        if not matrix.empty:
            cf = CollaborativeFilter(matrix)
            if cf.has_sufficient_data(user_id):
                cf_results = cf.recommend(user_id, top_k=len(dishes))
                cf_by_name = {r["dish_name"]: r["cf_score"] for r in cf_results}
            else:
                cf_weight = 0.0  # no neighbors yet -> fall back cleanly to Week 2 blend

    # 3. Final blend
    final = []
    for name, dish in base_by_name.items():
        if name in exclude_dish_names or name in liked_dish_names:
            continue

        existing_combined = dish["combined_score"]  # already baseline+content blended
        cf_score = cf_by_name.get(name, 0.0)

        # cf_score lives in roughly [-1, 1]; combined_score lives in [0, 1].
        # Normalise cf_score to [0, 1] so the blend is on comparable scales.
        cf_score_normalised = (cf_score + 1) / 2

        final_score = (
            (1 - cf_weight) * existing_combined +
            cf_weight * cf_score_normalised
        )

        final.append({
            **dish,
            "cf_score": round(cf_score, 4),
            "hybrid_score": round(final_score, 4),
            "_cf_weight": cf_weight,
            "_feedback_count": feedback_count,
        })

    final.sort(key=lambda d: d["hybrid_score"], reverse=True)
    return final[:top_k]


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
    store = FeedbackStore(db_path=Path("data/feedback_demo.db"))

    # Simulate a returning user with some feedback history
    test_user = "hybrid_demo_user"
    sample_names = [d["name"] for d in dishes[:6]]
    store.add_feedback(test_user, sample_names[0], 1)
    store.add_feedback(test_user, sample_names[1], -1)
    store.add_feedback("similar_user", sample_names[0], 1)
    store.add_feedback("similar_user", sample_names[1], -1)
    store.add_feedback("similar_user", sample_names[2], 1)  # this is the CF signal

    results = hybrid_rank(
        dishes,
        content_scorer=scorer,
        feedback_store=store,
        user_id=test_user,
        user_ingredients=["paneer", "tomato", "onion", "garlic"],
        user_type="bachelor",
        requested_meal_type="dinner",
        top_k=5,
    )

    print(f"Feedback count for {test_user}: {store.feedback_count(test_user)}")
    print(f"CF weight applied: {get_cf_weight(store.feedback_count(test_user))}\n")

    for i, d in enumerate(results, 1):
        print(f"{i}. {d['name']:30s} hybrid={d['hybrid_score']}  "
              f"(baseline+content={d['combined_score']}, cf={d['cf_score']}, "
              f"cf_weight={d['_cf_weight']})")

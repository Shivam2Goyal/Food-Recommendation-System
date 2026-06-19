# src/simulated_users.py

"""
Simulated user testing.

Per the roadmap: "Create 3-4 synthetic user personas (bachelor weekday,
family weekend, health-focused). Run 20-30 simulated sessions per persona.
Score against a manually-created 'ground truth' preference list."

This runs your ACTUAL hybrid_rank() pipeline (not a mock) against
synthetic personas, simulates a thumbs-up/down reaction using each
persona's ground-truth preferences, and feeds the results into
evaluation.py from Task 1.

Why ground truth instead of random reactions:
A persona's "ground truth" is a hand-written rule for what that kind of
person would actually like (e.g. health-focused -> likes low-prep-time +
'healthy'/'low-fat' tagged dishes, dislikes heavy desserts). This lets us
catch real systemic bias before real users ever see it — e.g. "the
bachelor persona keeps getting 90-minute mutton curries" would be an
obvious, fixable bug. Without ground truth we'd just be measuring noise.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from src.session import RecommendationSession
from src.content_scorer import ContentScorer
from src.feedback_store import FeedbackStore
from src.evaluation import SessionRecord, run_full_evaluation


@dataclass
class Persona:
    """
    A synthetic user archetype. `likes_rule` is a function that takes a
    dish dict and returns True if this persona would thumbs-up it — this
    IS the ground truth, written by us (Intern A), by hand, based on
    common sense about what that kind of person wants.
    """
    name: str
    user_type: str                  # matches your existing user_type field
    requested_meal_type: str | None
    max_prep_time: int | None
    sample_ingredients: list        # what this persona typically "has on hand"
    likes_rule: callable             # dish dict -> bool


def _is_quick_or_easy(dish: dict) -> bool:
    tags = set(t.lower() for t in dish.get("dietary_tags", []))
    return bool(tags & {"quick", "easy"}) or dish.get("prep_time_mins", 999) <= 25


def _is_healthy(dish: dict) -> bool:
    tags = set(t.lower() for t in dish.get("dietary_tags", []))
    return bool(tags & {"healthy", "low-fat", "high-protein", "gluten-free"})


def _is_comfort_traditional(dish: dict) -> bool:
    tags = set(t.lower() for t in dish.get("dietary_tags", []))
    return bool(tags & {"comfort", "traditional"})


PERSONAS = [
    Persona(
        name="bachelor_weekday",
        user_type="bachelor",
        requested_meal_type="dinner",
        max_prep_time=30,
        sample_ingredients=["onion", "tomato", "rice", "egg", "oil"],
        likes_rule=_is_quick_or_easy,
    ),
    Persona(
        name="family_weekend",
        user_type="family",
        requested_meal_type="dinner",
        max_prep_time=90,
        sample_ingredients=["paneer", "tomato", "onion", "garlic", "ghee", "milk", "rice"],
        likes_rule=_is_comfort_traditional,
    ),
    Persona(
        name="health_focused",
        user_type="health-focused",
        requested_meal_type=None,
        max_prep_time=45,
        sample_ingredients=["spinach", "tomato", "lentils", "yogurt", "cucumber"],
        likes_rule=_is_healthy,
    ),
    Persona(
        name="student_quick_snack",
        user_type="student",
        requested_meal_type="snack",
        max_prep_time=20,
        sample_ingredients=["bread", "potato", "onion", "oil"],
        likes_rule=_is_quick_or_easy,
    ),
]


def run_persona_sessions(
    persona: Persona,
    dishes: list[dict],
    content_scorer: ContentScorer,
    feedback_store: FeedbackStore,
    num_sessions: int = 25,
    seed: int = 42,
) -> list[SessionRecord]:
    """
    Simulates num_sessions independent sessions for one persona.
    Each session: get top recommendation, check ground-truth likes_rule,
    record accept/reject, feed that reaction back into FeedbackStore
    (so later sessions in this run can actually benefit from earlier
    "feedback" — mirroring what would happen with a real returning user).
    """
    rng = random.Random(seed)
    records = []

    for i in range(num_sessions):
        # Each "session" is a fresh RecommendationSession (no session-level
        # repeat exclusion carried across sessions -- only persistent
        # feedback carries over, same as a real user closing and reopening
        # the app), but the SAME user_id so feedback accumulates.
        user_id = f"sim_{persona.name}"
        session = RecommendationSession(user_id=user_id)

        # Slight variation per session so it's not the literal same
        # request 25 times in a row
        ingredients = persona.sample_ingredients.copy()
        rng.shuffle(ingredients)
        ingredients = ingredients[: rng.randint(max(2, len(ingredients) - 2), len(ingredients))]

        dish = session.get_next_recommendation(
            dishes,
            content_scorer=content_scorer,
            feedback_store=feedback_store,
            user_ingredients=ingredients,
            user_type=persona.user_type,
            requested_meal_type=persona.requested_meal_type,
            max_prep_time=persona.max_prep_time,
        )

        if dish is None:
            continue  # catalogue exhausted for this user (rare at this scale)

        accepted = persona.likes_rule(dish)

        if accepted:
            session.accept_current(dish["name"], feedback_store)
        else:
            session.reject_current(dish["name"], feedback_store)

        records.append(SessionRecord(
            user_id=user_id,
            recommended_dish=dish["name"],
            was_accepted=accepted,
        ))

    return records


def run_all_personas(
    dishes: list[dict],
    content_scorer: ContentScorer,
    feedback_store: FeedbackStore,
    num_sessions_per_persona: int = 25,
) -> dict:
    """
    Runs every persona and returns both per-persona and combined results,
    each already passed through evaluation.run_full_evaluation().
    """
    all_records = []
    per_persona_results = {}

    for persona in PERSONAS:
        records = run_persona_sessions(
            persona, dishes, content_scorer, feedback_store,
            num_sessions=num_sessions_per_persona,
        )
        all_records.extend(records)
        per_persona_results[persona.name] = run_full_evaluation(records)

    combined_results = run_full_evaluation(all_records)

    return {
        "per_persona": per_persona_results,
        "combined": combined_results,
    }


# ── Quick manual test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    clean_path = Path("data/clean/dishes.json")
    if not clean_path.exists():
        print("Run src/clean.py first to generate data/clean/dishes.json")
        exit()

    with open(clean_path) as f:
        dishes = json.load(f)

    scorer = ContentScorer(dishes)
    # Fresh DB just for this simulation run, so it doesn't pollute real demo data
    store = FeedbackStore(db_path=Path("data/feedback_simulation.db"))

    results = run_all_personas(dishes, scorer, store, num_sessions_per_persona=25)

    print("=== PER-PERSONA RESULTS (test split) ===\n")
    for persona_name, result in results["per_persona"].items():
        test_metrics = result["test"]
        print(f"{persona_name}:")
        print(f"  Precision@1:  {test_metrics['precision_at_1']}")
        print(f"  Rejection:    {test_metrics['rejection_rate']}")
        print(f"  Diversity:    {test_metrics['diversity']['overall_average']}")
        print()

    print("=== COMBINED (all personas, test split) ===")
    print(results["combined"]["test"])

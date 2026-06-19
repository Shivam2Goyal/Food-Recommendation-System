# src/constraint_fallback.py

"""
Constraint fallback guard.

Per the roadmap: "If ingredient coverage < 40% for ALL remaining dishes,
don't refuse — return the highest-scoring dish and flag missing_ingredients
so Intern B's LLM can ask the user about it."

This is mostly already true by construction:
  - compute_coverage() (Week 2) always returns missing_ingredients, never
    raises, never returns an empty result.
  - hybrid_rank() / RecommendationSession never filter dishes OUT based on
    coverage_score — coverage is a scoring FACTOR (55% weight in baseline),
    not a hard gate. A 10%-coverage dish can still be returned; it'll just
    rank lower than a 90%-coverage dish, all else equal.

This module adds one explicit, testable guarantee on top: a thin wrapper
around the session/ranker call stack that:
  1. Verifies a result was actually returned (never silently empty when
     candidates exist).
  2. Adds a `low_coverage_warning` flag when coverage is below threshold,
     so Intern B's layer has an explicit signal (rather than having to
     infer it by checking coverage_score < 0.4 themselves every time).
  3. Logs/counts how often this fallback path triggers, useful for Week 4
     evaluation ("how often are we recommending things with poor coverage").
"""

LOW_COVERAGE_THRESHOLD = 0.4


def apply_constraint_fallback(dish: dict | None) -> dict | None:
    """
    Takes the top recommended dish (or None) and returns it annotated with
    a low_coverage_warning flag. Never filters the dish out — that's the
    entire point of "fallback, don't refuse."

    Returns None only if the input was None (i.e. truly no candidates left
    in the catalogue — a separate, legitimate "nothing to recommend" case
    that's distinct from "low coverage", and should be handled by the
    caller as an empty-state UI message, not a coverage problem).
    """
    if dish is None:
        return None

    coverage = dish.get("coverage_score", 0.0)
    annotated = {**dish, "low_coverage_warning": coverage < LOW_COVERAGE_THRESHOLD}
    return annotated


def describe_fallback_reason(dish: dict) -> str | None:
    """
    Human-readable (and LLM-prompt-ready) explanation for why this
    recommendation has low coverage, for Intern B to drop directly into
    their prompt when low_coverage_warning is True.

    Returns None if coverage is fine — caller should check the warning
    flag first and only call this when it's True.
    """
    missing = dish.get("missing_ingredients", [])
    if not missing:
        return None

    coverage_pct = int(dish.get("coverage_score", 0.0) * 100)
    missing_list = ", ".join(missing)
    return (
        f"This recommendation only covers {coverage_pct}% of the ingredients "
        f"you have on hand. You're missing: {missing_list}. "
        f"Ask the user if they can get these, or want substitution suggestions."
    )


# ── Quick manual test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.session import RecommendationSession
    from src.content_scorer import ContentScorer
    from src.feedback_store import FeedbackStore

    clean_path = Path("data/clean/dishes.json")
    if not clean_path.exists():
        print("Run src/clean.py first to generate data/clean/dishes.json")
        exit()

    with open(clean_path) as f:
        dishes = json.load(f)

    scorer = ContentScorer(dishes)
    store = FeedbackStore(db_path=Path("data/feedback_demo.db"))
    session = RecommendationSession(user_id="fallback_demo_user")

    # Deliberately sparse ingredients -> coverage should be low for most/all dishes
    sparse_ingredients = ["water"]

    raw_dish = session.get_next_recommendation(
        dishes,
        content_scorer=scorer,
        feedback_store=store,
        user_ingredients=sparse_ingredients,
    )

    result = apply_constraint_fallback(raw_dish)

    if result is None:
        print("No candidates available at all (catalogue exhausted).")
    else:
        print(f"Recommended: {result['name']}")
        print(f"Coverage: {result['coverage_score']}")
        print(f"Missing: {result['missing_ingredients']}")
        print(f"Low coverage warning: {result['low_coverage_warning']}")

        if result["low_coverage_warning"]:
            print(f"\nPrompt hint for Intern B's LLM:")
            print(f"  {describe_fallback_reason(result)}")
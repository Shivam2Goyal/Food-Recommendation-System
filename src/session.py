# src/session.py

"""
Session-level rejection routing.

Two distinct exclusion mechanisms work together:

1. PERSISTENT dislikes (across all future sessions, forever):
   Handled already in hybrid_ranker.py via FeedbackStore.get_disliked_dishes().
   A thumbs-down is permanent signal — recorded in the DB.

2. SESSION-level "already shown" tracking (this file):
   Even dishes the user hasn't explicitly rejected shouldn't repeat within
   the same sitting. If we recommend "Dal Makhani" and the user hasn't
   reacted yet, then asks for "something else", we shouldn't show it again
   even though it's not a recorded dislike.

A session is intentionally lightweight and in-memory (no DB) since it only
needs to live as long as one user's active conversation. Intern B's layer
owns the actual session lifecycle (start/end) — this class just tracks
what's been shown and routes rejections for the recommender's side.
"""

from dataclasses import dataclass, field
from src.hybrid_ranker import hybrid_rank
from src.content_scorer import ContentScorer
from src.feedback_store import FeedbackStore


@dataclass
class RecommendationSession:
    """
    One active recommendation session for one user.

    shown_dish_names: every dish name returned to the user this session,
    regardless of whether they reacted. Always excluded from future
    recommend() calls in this same session.
    """
    user_id: str
    shown_dish_names: list = field(default_factory=list)
    rejected_this_session: list = field(default_factory=list)  # subset of shown, explicitly thumbs-downed

    def get_next_recommendation(
        self,
        dishes: list[dict],
        content_scorer: ContentScorer,
        feedback_store: FeedbackStore,
        user_ingredients: list[str],
        user_type: str = None,
        requested_meal_type: str = None,
        max_prep_time: int = None,
    ) -> dict | None:
        """
        Returns the single best next dish, excluding everything already
        shown this session AND everything persistently disliked.

        Returns None if no candidates remain (rare, but possible if the
        catalogue is small and the session has gone on a while).
        """
        results = hybrid_rank(
            dishes,
            content_scorer=content_scorer,
            feedback_store=feedback_store,
            user_id=self.user_id,
            user_ingredients=user_ingredients,
            user_type=user_type,
            requested_meal_type=requested_meal_type,
            max_prep_time=max_prep_time,
            exclude_dish_names=self.shown_dish_names,
            top_k=1,
        )

        if not results:
            return None

        top_dish = results[0]
        self.shown_dish_names.append(top_dish["name"])
        return top_dish

    def reject_current(
        self,
        dish_name: str,
        feedback_store: FeedbackStore,
    ):
        """
        Call this when the user thumbs-downs the dish they were just shown.
        Records the permanent dislike in FeedbackStore AND tracks it as
        rejected for this session (for transparency/debugging — it's
        already in shown_dish_names either way).
        """
        feedback_store.add_feedback(self.user_id, dish_name, -1)
        if dish_name not in self.rejected_this_session:
            self.rejected_this_session.append(dish_name)

    def accept_current(
        self,
        dish_name: str,
        feedback_store: FeedbackStore,
    ):
        """Call this when the user thumbs-ups the dish they were just shown."""
        feedback_store.add_feedback(self.user_id, dish_name, 1)

    def reset(self):
        """Start a fresh session for the same user (e.g. new day, new sitting)."""
        self.shown_dish_names = []
        self.rejected_this_session = []


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

    session = RecommendationSession(user_id="session_demo_user")

    print("Simulating a session: ask for dinner, reject twice, accept third.\n")

    for round_num in range(1, 4):
        dish = session.get_next_recommendation(
            dishes,
            content_scorer=scorer,
            feedback_store=store,
            user_ingredients=["paneer", "tomato", "onion", "garlic"],
            user_type="bachelor",
            requested_meal_type="dinner",
        )

        if dish is None:
            print(f"Round {round_num}: No more candidates available.")
            break

        print(f"Round {round_num}: Recommended '{dish['name']}' "
              f"(hybrid_score={dish['hybrid_score']})")

        if round_num < 3:
            print(f"  -> User rejects '{dish['name']}'")
            session.reject_current(dish["name"], store)
        else:
            print(f"  -> User accepts '{dish['name']}'")
            session.accept_current(dish["name"], store)

    print(f"\nDishes shown this session: {session.shown_dish_names}")
    print(f"Rejected this session: {session.rejected_this_session}")
    print(f"\nUser's permanent dislikes in DB: {store.get_disliked_dishes('session_demo_user')}")
    print(f"User's permanent likes in DB: {store.get_liked_dishes('session_demo_user')}")

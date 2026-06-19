# src/collaborative_filter.py

"""
Lightweight collaborative filtering (CF).

Approach: user-based CF with cosine similarity.
  1. Build the user-item ratings matrix (from FeedbackStore).
  2. Find users most similar to the target user (by rating pattern).
  3. Recommend dishes those similar users liked, that the target user
     hasn't rated yet — weighted by how similar each neighbor is.

This is intentionally the simpler of the two approaches the roadmap
mentions (user-based cosine vs SVD/matrix factorisation). It's easy to
reason about, works fine at small scale, and doesn't need a training
step — just recomputed on the fly from current feedback data.

Cold-start handling: if a user has too few ratings, or no similar users
exist yet, CF returns an empty list. The caller (hybrid_ranker, Task 3)
is responsible for falling back to baseline/content scoring in that case.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MIN_NEIGHBORS_REQUIRED = 1  # need at least 1 similar user with overlapping ratings


class CollaborativeFilter:
    def __init__(self, user_item_matrix: pd.DataFrame):
        """
        user_item_matrix: rows = user_id, columns = dish_name, values = rating
        (0 = no rating, 1 = thumbs up, -1 = thumbs down). Typically produced by
        FeedbackStore.get_user_item_matrix().
        """
        self.matrix = user_item_matrix

    def _user_similarity(self, target_user: str) -> pd.Series:
        """Cosine similarity between target_user and every other user."""
        if target_user not in self.matrix.index:
            return pd.Series(dtype=float)

        sims = cosine_similarity(self.matrix)
        sim_df = pd.DataFrame(sims, index=self.matrix.index, columns=self.matrix.index)
        user_sims = sim_df.loc[target_user].drop(target_user)
        return user_sims.sort_values(ascending=False)

    def recommend(self, target_user: str, top_k: int = 10) -> list[dict]:
        """
        Returns a list of {"dish_name": ..., "cf_score": ...} sorted by
        predicted preference, for dishes the target_user hasn't already rated.

        cf_score is a weighted average of neighbors' ratings for that dish,
        weighted by similarity to target_user. Range is roughly [-1, 1].

        Returns [] if the user doesn't exist in the matrix, has no neighbors
        with any meaningful similarity, or the matrix is empty/too small.
        """
        if self.matrix.empty or target_user not in self.matrix.index:
            return []

        user_sims = self._user_similarity(target_user)
        # Drop neighbors with zero or negative similarity — they add noise, not signal
        user_sims = user_sims[user_sims > 0]

        if len(user_sims) < MIN_NEIGHBORS_REQUIRED:
            return []

        target_ratings = self.matrix.loc[target_user]
        unrated_dishes = target_ratings[target_ratings == 0].index

        if len(unrated_dishes) == 0:
            return []  # user has rated everything in the catalogue (unlikely, but handle it)

        scores = {}
        for dish in unrated_dishes:
            neighbor_ratings = self.matrix.loc[user_sims.index, dish]
            # Only consider neighbors who actually rated this dish
            relevant = neighbor_ratings[neighbor_ratings != 0]
            if relevant.empty:
                continue
            relevant_sims = user_sims.loc[relevant.index]
            weighted_score = np.average(relevant.values, weights=relevant_sims.values)
            scores[dish] = weighted_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"dish_name": d, "cf_score": round(float(s), 4)} for d, s in ranked[:top_k]]

    def has_sufficient_data(self, target_user: str) -> bool:
        """
        Quick check used by the hybrid ranker (Task 3) to decide whether
        CF is even worth running for this user.
        """
        if self.matrix.empty or target_user not in self.matrix.index:
            return False
        user_sims = self._user_similarity(target_user)
        return bool((user_sims > 0).sum() >= MIN_NEIGHBORS_REQUIRED)


# ── Quick manual test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.feedback_store import FeedbackStore
    from pathlib import Path

    store = FeedbackStore(db_path=Path("data/feedback_demo.db"))

    # Simulate a slightly richer feedback history so CF has something to chew on
    store.add_feedback("user_3", "Dal Makhani", 1)
    store.add_feedback("user_3", "Shahi Paneer", 1)
    store.add_feedback("user_3", "Rasgulla", -1)

    matrix = store.get_user_item_matrix()
    print("User-item matrix:")
    print(matrix)
    print()

    cf = CollaborativeFilter(matrix)

    for user in matrix.index:
        print(f"--- Recommendations for {user} ---")
        if cf.has_sufficient_data(user):
            recs = cf.recommend(user, top_k=5)
            if recs:
                for r in recs:
                    print(f"  {r['dish_name']:20s} cf_score={r['cf_score']}")
            else:
                print("  No unrated dishes with neighbor signal.")
        else:
            print("  Insufficient data for CF — caller should fall back to baseline.")
        print()

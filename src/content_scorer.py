# src/content_scorer.py

"""
Content-based scoring.

Builds a feature vector per dish (ingredients + cuisine/region + flavor_profile
+ dietary_tags) and computes similarity between dishes using cosine similarity
over a bag-of-features representation.

Used for:
  - "More like this" recommendations once a user has liked/picked a dish
  - Diversifying or narrowing baseline results using content similarity

No external ML libraries beyond sklearn (already in requirements.txt).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def _dish_to_feature_string(dish: dict) -> str:
    """
    Flatten a dish's content fields into a single space-separated string
    so TF-IDF can vectorize it like a 'document'.

    Ingredients are weighted more heavily (repeated) since they matter most
    for similarity — two dishes sharing ingredients should be considered
    more similar than two dishes that just share a flavor_profile.
    """
    parts = []

    ingredients = dish.get("ingredients", [])
    # Repeat ingredients 3x to weight them higher in TF-IDF
    parts.extend(ingredients * 3)

    if dish.get("region"):
        parts.append(f"region_{dish['region'].lower().replace(' ', '_')}")
    if dish.get("state"):
        parts.append(f"state_{dish['state'].lower().replace(' ', '_')}")

    flavor = dish.get("flavor_profile") or dish.get("dietary_tags", [])
    if isinstance(flavor, str):
        parts.append(f"flavor_{flavor.lower()}")

    for tag in dish.get("dietary_tags", []):
        parts.append(f"tag_{tag.lower().replace(' ', '_')}")

    parts.append(f"meal_{dish.get('meal_type', 'unknown')}")

    return " ".join(parts)


class ContentScorer:
    """
    Fit once on the full dish catalogue, then query similarity cheaply.
    """

    def __init__(self, dishes: list[dict]):
        self.dishes = dishes
        self.name_to_idx = {d["name"]: i for i, d in enumerate(dishes)}

        corpus = [_dish_to_feature_string(d) for d in dishes]
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def similar_to(self, dish_name: str, top_k: int = 5) -> list[dict]:
        """
        Return top_k dishes most similar in content to the given dish_name.
        Excludes the dish itself.
        """
        if dish_name not in self.name_to_idx:
            raise ValueError(f"'{dish_name}' not found in catalogue")

        idx = self.name_to_idx[dish_name]
        target_vector = self.tfidf_matrix[idx]

        sims = cosine_similarity(target_vector, self.tfidf_matrix).flatten()

        # Rank all dishes by similarity, excluding itself
        ranked_idx = np.argsort(sims)[::-1]
        results = []
        for i in ranked_idx:
            if i == idx:
                continue
            results.append({
                **self.dishes[i],
                "content_similarity": round(float(sims[i]), 4),
            })
            if len(results) >= top_k:
                break
        return results

    def score_against_profile(self, liked_dish_names: list[str], candidate_dishes: list[dict] = None) -> list[dict]:
        """
        Given a list of dishes the user has liked, score all candidate dishes
        (or the full catalogue if none given) by average similarity to the
        liked dishes. This is the building block for a user "taste profile".
        """
        candidates = candidate_dishes if candidate_dishes is not None else self.dishes

        liked_indices = [self.name_to_idx[n] for n in liked_dish_names if n in self.name_to_idx]
        if not liked_indices:
            raise ValueError("None of the liked_dish_names were found in catalogue")

        liked_vectors = self.tfidf_matrix[liked_indices]

        results = []
        for dish in candidates:
            if dish["name"] in liked_dish_names:
                continue  # don't recommend what they already liked
            idx = self.name_to_idx.get(dish["name"])
            if idx is None:
                continue
            candidate_vector = self.tfidf_matrix[idx]
            sims = cosine_similarity(candidate_vector, liked_vectors).flatten()
            avg_sim = round(float(np.mean(sims)), 4)
            results.append({**dish, "content_score": avg_sim})

        results.sort(key=lambda d: d["content_score"], reverse=True)
        return results


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

    sample_dish = dishes[0]["name"]
    print(f"Dishes similar to '{sample_dish}':\n")

    for d in scorer.similar_to(sample_dish, top_k=5):
        print(f"  {d['name']:30s} similarity={d['content_similarity']}")

    print()
    print(f"Taste profile test (liked: '{sample_dish}'):\n")
    for d in scorer.score_against_profile([sample_dish])[:5]:
        print(f"  {d['name']:30s} content_score={d['content_score']}")

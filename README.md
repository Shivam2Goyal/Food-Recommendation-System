# Food Recommender

Recommender engine for the food recommendation app. Owns dataset, ingredient matching, ranking, feedback storage, and evaluation. Exposes a single HTTP endpoint for the LLM/product layer) to call.

## Architecture

```
data/raw/indian_food.csv        → source dataset
        ↓ src/clean.py
data/clean/dishes.json          → cleaned, normalised dish catalogue
        ↓
src/matcher.py                  → ingredient fuzzy-matching + coverage scoring
src/baseline_ranker.py          → cold-start rules-based ranking
src/content_scorer.py           → TF-IDF + cosine similarity between dishes
src/combined_ranker.py          → blends baseline + content (Week 2)
src/feedback_store.py           → SQLite thumbs up/down storage
src/collaborative_filter.py     → user-based CF (cosine similarity)
src/hybrid_ranker.py            → blends baseline + content + CF (Week 3)
src/session.py                  → session-level rejection routing
src/constraint_fallback.py      → low-coverage warning annotation
src/evaluation.py               → Precision@1, rejection rate, diversity metrics
src/simulated_users.py          → synthetic persona testing
        ↓
src/api.py                      → FastAPI endpoint: POST /recommend
```

## Dataset

**Source:** Indian Food 101 dataset (Kaggle) — 255 dishes after cleaning/dedup.

**Columns used:** `name`, `ingredients`, `diet`, `prep_time`, `cook_time`, `flavor_profile`, `course`, `state`, `region`.

**Known data quirks handled in `clean.py`:**
- `-1` used as a null sentinel, appearing as both int and string depending on column — normalised via `is_null_value()`.
- `course` values (e.g. "main course", "dessert") mapped to our standard `meal_type` (breakfast/lunch/dinner/snack) via `COURSE_TO_MEAL_TYPE`.
- Ingredient name variants (Hindi/English, plurals) normalised via `src/synonyms.py` (e.g. "tamatar"/"tomatoes" → "tomato").

**Heuristic tags (`infer_extra_tags` in `clean.py`):**
The source dataset only provides `diet` (veg/non-veg) and `flavor_profile` (sweet/spicy/savory/bitter/sour). Tags like `quick`, `easy`, `healthy`, `comfort`, and `traditional` — needed for the cold-start ranker's `user_type` affinity scoring — don't exist in the source data, so they're inferred from prep_time and ingredient-name keyword matching.

## How recommendation scoring works

1. **Ingredient coverage** (`matcher.py`) — fuzzy-matches user-stated ingredients against dish requirements (exact match → synonym → RapidFuzz fuzzy match → hand-curated substitutes like ghee/butter, paneer/tofu).
2. **Baseline score** (`baseline_ranker.py`) — weighted blend: coverage (55%) + meal_type match (20%) + prep_time fit (15%) + user_type affinity (10%). This is the cold-start fallback for every new user.
3. **Content score** (`content_scorer.py`) — TF-IDF over ingredients/region/flavor/tags, cosine similarity to dishes the user has liked.
4. **CF score** (`collaborative_filter.py`) — user-based collaborative filtering once enough feedback history exists.
5. **Final hybrid blend** (`hybrid_ranker.py`) — combines all three. Weighting shifts automatically as more signal becomes available:

   | Liked dishes | Baseline+content split |
   |---|---|
   | 0 | 100% baseline |
   | 1–2 | 75% baseline / 25% content |
   | 3–4 | 50% / 50% |
   | 5+ | 30% / 70% |

   | Feedback count | CF weight (carved out of the above) |
   |---|---|
   | 0 | 0% |
   | 1–4 | 15% |
   | 5–14 | 40% |
   | 15+ | 55% |

6. **Constraint fallback** (`constraint_fallback.py`) — low ingredient coverage (<40%) never blocks a recommendation. It's flagged via `low_coverage_warning` so the LLM layer can proactively ask the user about missing ingredients, rather than refusing to recommend anything.

## Running it

```bash
# Setup
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Clean the raw dataset
python -m src.clean

# Run the API
uvicorn src.api:app --reload --port 8000
# → http://127.0.0.1:8000/docs for interactive testing

# Run all tests
pytest -v

# Run evaluation
python -m src.simulated_users
```

## API contract

See `contract.json` for the full field-by-field spec. Summary:

```
POST /recommend
{
  "user_ingredients": ["paneer", "tomato", "onion"],
  "user_type": "bachelor",               // optional
  "requested_meal_type": "dinner",       // optional
  "max_prep_time": 30,                   // optional
  "liked_dish_names": [],                // optional
  "top_k": 10                            // optional, default 10
}

→ { "results": [ { dish_name, cuisine, region, state, meal_type,
                    flavor_profile, ingredients_required,
                    ingredients_available, missing_ingredients,
                    coverage_score, dietary_tags, prep_time_mins } ],
    "count": N }
```

For session-based rejection routing (no-repeat-in-session, persistent dislikes), use `src/session.py`'s `RecommendationSession` class directly rather than the stateless `/recommend` endpoint — see its docstring for usage.

## Evaluation results (Week 4)

Synthetic persona testing (`src/simulated_users.py`), 4 personas × 25 simulated sessions each, 80/20 train/test split:

| Metric | Before tag fix | After tag fix |
|---|---|---|
| Combined Precision@1 | 0.55 | 0.75–0.95* |
| Combined rejection rate | 0.33–0.45 | 0.05–0.25* |
| Diversity (repeat users) | 1.0 | 1.0 |

\* *Varies run-to-run because the test split is only ~20 sessions; small N makes percentages noisy. Directionally, the fix clearly helped — `family_weekend` and `health_focused` went from 0.0 precision (structurally impossible to succeed, since their required tags didn't exist in the data) to non-zero and usually high.*

**Root cause found during tuning:** the source dataset has no `comfort`/`traditional`/`healthy`/`quick`/`easy` tags. Two personas (`family_weekend`, `health_focused`) were scoring 0.0 precision not because the ranking algorithm was wrong, but because the ground-truth signal they needed didn't exist anywhere in the data. Fixed by adding a heuristic tag-inference step in `clean.py`.

## Known limitations

1. **Heuristic tags are imperfect.** `quick`/`easy`/`healthy`/`comfort`/`traditional` are inferred from ingredient-name keyword matching and prep_time, not from real metadata. This produces occasional false positives on dishes with incomplete or non-standard ingredient phrasing (e.g. a fried dish whose ingredient list doesn't explicitly say "fried" or "oil"). A production version should use a richer dataset with real nutritional/preparation metadata, or an LLM-based classifier instead of keyword matching.
2. **No diet/allergen hard filter.** The system currently has no hard exclusion for non-vegetarian dishes when a user is vegetarian, or for specific allergens — `diet` is currently just a soft-scored tag, not a filter. This should be added before any real user-facing launch; flagged to Intern B at SYNC 3.
3. **Small dataset (255 dishes).** Fine for an MVP demo, but CF and content-similarity will perform better with a larger, more diverse catalogue.
4. **CF is user-based cosine similarity**, not matrix factorisation/SVD. Simpler and sufficient at this scale; would need revisiting if the user base grows significantly.

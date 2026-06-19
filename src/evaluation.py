# src/evaluation.py

"""
Offline evaluation harness.

Per the roadmap: "Split your synthetic feedback data 80/20. Measure:
Precision@1 (did the user thumbs-up the single recommendation?),
rejection rate per session, and diversity (are repeat users getting
varied dishes over time?)."

This module is INTENTIONALLY decoupled from any particular dataset —
it works on a list of "session records" you generate (synthetically here
in Week 4, but the same shape would work on real feedback data later).

A session record looks like:
    {
        "user_id": "u1",
        "recommended_dish": "Dal Makhani",
        "was_accepted": True,   # thumbs up = True, thumbs down = False
    }

Metrics implemented:
  - precision_at_1: fraction of recommendations that got thumbs-up
  - rejection_rate: fraction of recommendations that got thumbs-down
    (note: precision_at_1 + rejection_rate == 1.0 by construction here,
    since every session record has a definite accept/reject. If you later
    add a "no reaction / skipped" state, the two will no longer sum to 1
    and that's fine — they're tracking different things.)
  - diversity: for repeat users (2+ sessions), how many UNIQUE dishes did
    they see relative to total sessions? 1.0 = never repeated, 0.0 = same
    dish every time.

Train/test split: an 80/20 split of (user_id, session) pairs. At this
scale we're not "training" anything (no learned parameters) — the split
exists so Task 3 (tune & iterate) can tune weights on the 80% "train" set
and report final metrics on the untouched 20% "test" set, avoiding the
trap of tuning directly on the numbers you're reporting.
"""

import random
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SessionRecord:
    user_id: str
    recommended_dish: str
    was_accepted: bool


def train_test_split(records: list[SessionRecord], test_fraction: float = 0.2, seed: int = 42) -> tuple[list, list]:
    """
    80/20 split. Shuffled deterministically (fixed seed) so re-running
    eval gives reproducible numbers — important when you're comparing
    "before tuning" vs "after tuning" in Task 3.
    """
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]


def precision_at_1(records: list[SessionRecord]) -> float:
    """Fraction of recommendations that the user thumbs-up'd."""
    if not records:
        return 0.0
    accepted = sum(1 for r in records if r.was_accepted)
    return round(accepted / len(records), 4)


def rejection_rate(records: list[SessionRecord]) -> float:
    """Fraction of recommendations that the user thumbs-down'd."""
    if not records:
        return 0.0
    rejected = sum(1 for r in records if not r.was_accepted)
    return round(rejected / len(records), 4)


def diversity_score(records: list[SessionRecord]) -> dict:
    """
    For users with 2+ sessions, measure how varied their recommendations
    were. Returns per-user diversity and an overall average.

    diversity for one user = unique_dishes_seen / total_sessions
      1.0 = every recommendation was a different dish (great)
      low = same dish recommended repeatedly (bad — likely a bug or an
            overly narrow content/CF signal dominating)
    """
    by_user = defaultdict(list)
    for r in records:
        by_user[r.user_id].append(r.recommended_dish)

    per_user_diversity = {}
    for user_id, dishes in by_user.items():
        if len(dishes) < 2:
            continue  # diversity is undefined/meaningless for single-session users
        unique_count = len(set(dishes))
        per_user_diversity[user_id] = round(unique_count / len(dishes), 4)

    if not per_user_diversity:
        overall_avg = None  # no repeat users in this dataset — can't compute
    else:
        overall_avg = round(sum(per_user_diversity.values()) / len(per_user_diversity), 4)

    return {
        "per_user": per_user_diversity,
        "overall_average": overall_avg,
        "repeat_users_count": len(per_user_diversity),
    }


def evaluate(records: list[SessionRecord]) -> dict:
    """Run all metrics on a set of records and return a summary dict."""
    return {
        "num_sessions": len(records),
        "precision_at_1": precision_at_1(records),
        "rejection_rate": rejection_rate(records),
        "diversity": diversity_score(records),
    }


def run_full_evaluation(records: list[SessionRecord], test_fraction: float = 0.2) -> dict:
    """
    Splits into train/test and reports metrics for both, plus the combined
    set. Train metrics are exploratory (use for Task 3 tuning); test
    metrics are what you report as the "real" numbers in Task 4/5.
    """
    train, test = train_test_split(records, test_fraction=test_fraction)
    return {
        "train": evaluate(train),
        "test": evaluate(test),
        "overall": evaluate(records),
    }


# ── Quick manual test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Tiny hand-built example so the metrics are easy to sanity-check by eye
    demo_records = [
        SessionRecord("u1", "Dal Makhani", True),
        SessionRecord("u1", "Shahi Paneer", True),
        SessionRecord("u1", "Rasgulla", False),
        SessionRecord("u2", "Poha", True),
        SessionRecord("u2", "Poha", True),       # same dish twice -> low diversity for u2
        SessionRecord("u3", "Mutton Rogan Josh", False),
    ]

    results = run_full_evaluation(demo_records, test_fraction=0.2)

    print("=== TRAIN ===")
    print(results["train"])
    print("\n=== TEST ===")
    print(results["test"])
    print("\n=== OVERALL ===")
    print(results["overall"])

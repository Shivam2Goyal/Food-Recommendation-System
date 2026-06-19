    # tests/test_evaluation.py

from src.evaluation import (
    SessionRecord,
    train_test_split,
    precision_at_1,
    rejection_rate,
    diversity_score,
    evaluate,
    run_full_evaluation,
)

SAMPLE_RECORDS = [
    SessionRecord("u1", "Dal Makhani", True),
    SessionRecord("u1", "Shahi Paneer", True),
    SessionRecord("u1", "Rasgulla", False),
    SessionRecord("u2", "Poha", True),
    SessionRecord("u2", "Poha", True),
    SessionRecord("u3", "Mutton Rogan Josh", False),
]

# ── train_test_split ───────────────────────────────────────────────────────────

def test_split_sizes_roughly_80_20():
    train, test = train_test_split(SAMPLE_RECORDS, test_fraction=0.2)
    assert len(train) + len(test) == len(SAMPLE_RECORDS)
    assert len(test) <= len(train)

def test_split_is_deterministic_with_same_seed():
    train1, test1 = train_test_split(SAMPLE_RECORDS, seed=42)
    train2, test2 = train_test_split(SAMPLE_RECORDS, seed=42)
    assert train1 == train2
    assert test1 == test2

def test_split_differs_with_different_seed():
    train1, _ = train_test_split(SAMPLE_RECORDS, seed=1)
    train2, _ = train_test_split(SAMPLE_RECORDS, seed=999)
    # Not guaranteed to always differ on tiny lists, but very likely here
    assert train1 != train2 or len(SAMPLE_RECORDS) < 3

def test_split_empty_list():
    train, test = train_test_split([])
    assert train == []
    assert test == []

# ── precision_at_1 ─────────────────────────────────────────────────────────────

def test_precision_at_1_all_accepted():
    records = [SessionRecord("u1", "A", True), SessionRecord("u2", "B", True)]
    assert precision_at_1(records) == 1.0

def test_precision_at_1_all_rejected():
    records = [SessionRecord("u1", "A", False), SessionRecord("u2", "B", False)]
    assert precision_at_1(records) == 0.0

def test_precision_at_1_mixed():
    records = [
        SessionRecord("u1", "A", True),
        SessionRecord("u2", "B", False),
        SessionRecord("u3", "C", True),
        SessionRecord("u4", "D", False),
    ]
    assert precision_at_1(records) == 0.5

def test_precision_at_1_empty_returns_zero():
    assert precision_at_1([]) == 0.0

# ── rejection_rate ─────────────────────────────────────────────────────────────

def test_rejection_rate_complements_precision():
    # Since every record is definite accept/reject, they must sum to 1.0
    p = precision_at_1(SAMPLE_RECORDS)
    r = rejection_rate(SAMPLE_RECORDS)
    assert abs((p + r) - 1.0) < 1e-9

def test_rejection_rate_empty_returns_zero():
    assert rejection_rate([]) == 0.0

# ── diversity_score ────────────────────────────────────────────────────────────

def test_diversity_excludes_single_session_users():
    records = [SessionRecord("u1", "A", True)]  # only 1 session
    result = diversity_score(records)
    assert "u1" not in result["per_user"]
    assert result["repeat_users_count"] == 0

def test_diversity_perfect_for_all_unique_dishes():
    records = [
        SessionRecord("u1", "A", True),
        SessionRecord("u1", "B", True),
        SessionRecord("u1", "C", False),
    ]
    result = diversity_score(records)
    assert result["per_user"]["u1"] == 1.0

def test_diversity_low_for_repeated_dish():
    records = [
        SessionRecord("u1", "A", True),
        SessionRecord("u1", "A", True),
    ]
    result = diversity_score(records)
    assert result["per_user"]["u1"] == 0.5  # 1 unique / 2 sessions

def test_diversity_overall_average_none_when_no_repeat_users():
    records = [SessionRecord("u1", "A", True), SessionRecord("u2", "B", True)]
    result = diversity_score(records)
    assert result["overall_average"] is None

def test_diversity_counts_repeat_users_correctly():
    result = diversity_score(SAMPLE_RECORDS)
    # u1 (3 sessions) and u2 (2 sessions) qualify; u3 (1 session) doesn't
    assert result["repeat_users_count"] == 2

# ── evaluate / run_full_evaluation ────────────────────────────────────────────

def test_evaluate_returns_all_keys():
    result = evaluate(SAMPLE_RECORDS)
    assert set(result.keys()) == {"num_sessions", "precision_at_1", "rejection_rate", "diversity"}

def test_evaluate_num_sessions_matches_input():
    result = evaluate(SAMPLE_RECORDS)
    assert result["num_sessions"] == len(SAMPLE_RECORDS)

def test_run_full_evaluation_has_train_test_overall():
    result = run_full_evaluation(SAMPLE_RECORDS)
    assert set(result.keys()) == {"train", "test", "overall"}

def test_run_full_evaluation_overall_matches_full_dataset():
    result = run_full_evaluation(SAMPLE_RECORDS)
    assert result["overall"]["num_sessions"] == len(SAMPLE_RECORDS)
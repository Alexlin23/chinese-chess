from AlphaZero.train.arena_stats import promotion_evidence


def test_all_draws_never_promote():
    score, lower, promote = promotion_evidence(0, 0, 400, 0.5)
    assert score == 0.5
    assert lower == 0
    assert not promote


def test_decisive_evidence_required():
    assert not promotion_evidence(2, 0, 98, 0.51)[2]
    assert promotion_evidence(80, 20, 0, 0.55)[2]

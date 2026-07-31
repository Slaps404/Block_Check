from session.matching_corpus import (
    CandidateRef,
    ScoredPair,
    TruePairRef,
    expand_same_work_order_candidates,
    make_pair_id,
    promote_near_misses,
)


def test_make_pair_id_is_stable_and_includes_endpoints():
    a = make_pair_id(1, "B1", "slide-aaa")
    b = make_pair_id(1, "B1", "slide-aaa")
    c = make_pair_id(1, "B1", "slide-bbb")
    assert a == b
    assert a != c
    assert "B1" in a and "slide-aaa" in a


def test_expand_candidates_excludes_true_pairs_and_crosses_blocks():
    trues = [
        TruePairRef(1, "WO1", "B1", "S1"),
        TruePairRef(1, "WO1", "B2", "S2"),
        TruePairRef(1, "WO1", "B3", "S3"),
    ]
    cands = expand_same_work_order_candidates(trues)
    assert CandidateRef(1, "WO1", "B1", "S1") not in cands
    assert CandidateRef(1, "WO1", "B1", "S2") in cands
    assert CandidateRef(1, "WO1", "B1", "S3") in cands
    assert CandidateRef(1, "WO1", "B2", "S1") in cands
    assert len(cands) == 6


def test_promote_near_misses_keeps_best_wrong_and_within_margin():
    rows = [
        ScoredPair("p-true", "B1", is_match=True, score=0.90),
        ScoredPair("p-best", "B1", is_match=False, score=0.80),
        ScoredPair("p-near", "B1", is_match=False, score=0.77),
        ScoredPair("p-far", "B1", is_match=False, score=0.50),
        ScoredPair("p-other", "B2", is_match=False, score=0.40),
    ]
    promoted = promote_near_misses(rows, margin=0.05)
    assert promoted == {"p-best", "p-near", "p-other"}

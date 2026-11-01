import pytest

from evaluation.metrics import MatchRecord, summarize_latencies, summarize_results


def test_results_perspective_and_forfeits():
    records = [MatchRecord("A", "B", 1), MatchRecord("B", "A", -1),
               MatchRecord("A", "B", 0), MatchRecord("B", "A", 1, "timeout")]
    result = summarize_results(records, "A")
    assert (result["wins"], result["draws"], result["losses"], result["total"]) == (2, 1, 1, 4)
    assert result["score"] == 0.625
    assert result["forfeits"]["timeout"] == 1
    assert summarize_results(records, "B")["forfeits"]["timeout"] == 0
    assert summarize_results(records, "không tham gia")["score"] is None


def test_score_course_example_and_latency():
    records = [MatchRecord("A", "B", v) for v in [1]*6 + [0]*2 + [-1]*2]
    assert summarize_results(records, "A")["score"] == 0.7
    assert summarize_latencies([1.0, 2.0, 3.0])["p95_s"] == pytest.approx(2.9)
    assert summarize_latencies([]) == {"count": 0, "mean_s": None, "p95_s": None}


@pytest.mark.parametrize("values", [[-1], [float("nan")], [float("inf")], [True]])
def test_invalid_latencies(values):
    with pytest.raises(ValueError):
        summarize_latencies(values)


@pytest.mark.parametrize("args", [("A", "A", 0), ("A", "B", 2),
                                 ("A", "B", 0, "timeout"), ("A", "B", 1, "sai"),
                                 ("A", "B", 1.0), (1, "B", 0)])
def test_invalid_records(args):
    with pytest.raises(ValueError):
        MatchRecord(*args)

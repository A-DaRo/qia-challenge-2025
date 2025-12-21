import math
import pytest

from caligo.scripts.peg_generator import DegreeDistribution


def test_degree_distribution_normalizes_and_converts():
    dd = DegreeDistribution(degrees=[2, 3], probabilities=[0.2, 0.8])
    assert all(isinstance(p, float) for p in dd.probabilities)
    assert math.isclose(sum(dd.probabilities), 1.0, rel_tol=1e-6)


@pytest.mark.parametrize("degrees,probs", [
    ([2], [0.5, 0.5]),
    ([0, 2], [0.5, 0.5]),
    ([2, 3], [1.2, -0.2]),
])
def test_degree_distribution_invalid_raises(degrees, probs):
    with pytest.raises(ValueError):
        DegreeDistribution(degrees=degrees, probabilities=probs)


def test_degree_distribution_zero_sum_raises():
    with pytest.raises(ValueError):
        DegreeDistribution(degrees=[2, 3], probabilities=[0.0, 0.0])

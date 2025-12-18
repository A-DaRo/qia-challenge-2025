import numpy as np
import pytest

from caligo.quantum.basis import BasisSelector
from caligo.types.exceptions import InvalidParameterError


def test_basis_selector_seeded_is_deterministic() -> None:
    seed = (123).to_bytes(8, "big")
    selector1 = BasisSelector(seed=seed)
    selector2 = BasisSelector(seed=seed)

    b1 = selector1.select_batch(n=128)
    b2 = selector2.select_batch(n=128)

    assert b1.dtype == np.uint8
    assert np.array_equal(b1, b2)


def test_basis_selector_generate_rejects_invalid_num_bases() -> None:
    selector = BasisSelector(seed=(0).to_bytes(8, "big"))

    with pytest.raises(InvalidParameterError):
        selector.select_batch(n=0)
    with pytest.raises(InvalidParameterError):
        selector.select_batch(n=-1)


def test_basis_selector_generate_values_are_bits() -> None:
    selector = BasisSelector(seed=(999).to_bytes(8, "big"))
    bases = selector.select_batch(n=1024)

    assert bases.dtype == np.uint8
    assert set(np.unique(bases)).issubset({0, 1})

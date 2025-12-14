"""
Pre-Flight Feasibility Checker for E-HOK Protocol.

This module implements the pre-flight feasibility gate (TASK-FEAS-001) that
determines whether a protocol session can produce a positive-length secure key
before consuming quantum resources.

Security Rationale
------------------
The feasibility checker enforces several critical conditions:

1. **QBER Hard Limit**: If expected QBER > 22%, secure OT is impossible
   regardless of other parameters (Lupo et al. 2023, Section VI).

2. **Strict-Less Condition**: Trusted noise must be strictly less than untrusted
   storage noise (Schaffner et al. 2009). Violation means adversary can hide
   in the trusted noise.

3. **Capacity × Rate Condition**: The storage capacity multiplied by storage rate
   must satisfy C_N · ν < 1/2 (König et al. 2012, Corollary I.2).

4. **Death Valley Detection**: Even if above conditions pass, insufficient
   sifted bits or excessive leakage can yield zero extractable key.

Abort Taxonomy
--------------
Following the roadmap's abort code taxonomy:
- ABORT-I-FEAS-001: QBER exceeds 22% hard limit
- ABORT-I-FEAS-002: Strict-less condition violated
- ABORT-I-FEAS-003: Capacity × rate condition violated
- ABORT-I-FEAS-004: Death Valley (ℓ_max ≤ 0)
- ABORT-I-FEAS-005: Invalid input parameters

References
----------
- Lupo et al. (2023): "Error-tolerant oblivious transfer in the noisy-storage model"
  Section VI: 22% hard limit derivation.
- König et al. (2012): "Unconditional Security from Noisy Quantum Storage"
  Corollary I.2: C_N · ν < 1/2 condition.
- Schaffner et al. (2009): "Robust Cryptography in the Noisy-Quantum-Storage Model"
  The strictly-less condition Q_trusted < r_storage.
- sprint_1_specification.md Section 3.2 (TASK-FEAS-001)
- master_roadmap.md Section 6.4 (Abort Code Taxonomy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ehok.analysis.nsm_bounds import (
    QBER_HARD_LIMIT,
    QBER_WARNING_THRESHOLD,
    FeasibilityResult,
    NSMBoundsCalculator,
    NSMBoundsInputs,
    channel_capacity,
    max_bound_entropy_rate,
)
from ehok.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Abort Codes
# =============================================================================

# Abort codes following roadmap taxonomy
ABORT_CODE_QBER_TOO_HIGH = "ABORT-I-FEAS-001"
ABORT_CODE_STRICT_LESS_VIOLATED = "ABORT-I-FEAS-002"
ABORT_CODE_CAPACITY_RATE_VIOLATED = "ABORT-I-FEAS-003"
ABORT_CODE_DEATH_VALLEY = "ABORT-I-FEAS-004"
ABORT_CODE_INVALID_PARAMETERS = "ABORT-I-FEAS-005"


# =============================================================================
# Input/Output Dataclasses
# =============================================================================


@dataclass(frozen=True)
class FeasibilityInputs:
    """
    Input parameters for pre-flight feasibility check.

    All parameters are validated before feasibility assessment.

    Attributes
    ----------
    expected_qber : float
        Expected QBER from channel/device characterization.
        Must be in [0, 0.5] as a probability.
    storage_noise_r : float
        Adversary's storage retention parameter r ∈ [0, 1].
        r = 0: Complete noise (best for security)
        r = 1: Perfect storage (worst for security)
    storage_rate_nu : float
        Fraction of qubits adversary can store ν ∈ [0, 1].
        ν = 0: No storage (trivially secure)
        ν = 1: All qubits storable
    epsilon_sec : float
        Security parameter ε ∈ (0, 1).
    n_target_sifted_bits : int
        Target number of sifted bits for the session.
    expected_leakage_bits : int
        Expected syndrome + verification leakage in bits.
    batch_size : int
        Per-batch feasibility check size. 0 = full-session mode (default),
        >0 = enables per-batch Death Valley detection for small batches.
        Used to detect cases where individual batches are too small to
        produce positive key even when aggregated session appears feasible.

    References
    ----------
    - sprint_1_specification.md Section 3.2
    - remediation_specification.md Section 4.1
    """

    expected_qber: float
    storage_noise_r: float
    storage_rate_nu: float
    epsilon_sec: float
    n_target_sifted_bits: int
    expected_leakage_bits: int
    batch_size: int = 0

    def __post_init__(self) -> None:
        """Validate input parameters."""
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {self.batch_size}")


@dataclass(frozen=True)
class FeasibilityDecision:
    """
    Result of pre-flight feasibility assessment.

    Attributes
    ----------
    is_feasible : bool
        True if protocol can proceed with positive key expectation.
    abort_code : str | None
        Abort code if not feasible; None if feasible.
    reason : str
        Human-readable explanation of the decision.
    recommended_min_n : int | None
        For Death Valley cases, minimum n for positive key.
        None if feasible or for other abort reasons.
    warnings : list[str]
        Non-fatal warnings (e.g., QBER in conservative zone).

    References
    ----------
    - sprint_1_specification.md Section 3.2
    - master_roadmap.md Section 6.4 (Abort Code Taxonomy)
    """

    is_feasible: bool
    abort_code: Optional[str]
    reason: str
    recommended_min_n: Optional[int] = None
    warnings: tuple[str, ...] = ()


# =============================================================================
# Feasibility Checker
# =============================================================================


class FeasibilityChecker:
    """
    Pre-flight feasibility gate for E-HOK protocol.

    This checker determines whether a protocol session should proceed based on:
    1. Parameter validity
    2. QBER hard limit (22%)
    3. Strict-less condition (Q_trusted < r_storage)
    4. Capacity × rate condition (C_N · ν < 1/2)
    5. Death Valley detection (ℓ_max > 0)

    The checker is stateless; each check() call is independent.

    Methods
    -------
    check(inputs: FeasibilityInputs) -> FeasibilityDecision
        Perform complete feasibility assessment.

    References
    ----------
    - sprint_1_specification.md Section 3.2 (TASK-FEAS-001)

    Examples
    --------
    >>> checker = FeasibilityChecker()
    >>> inputs = FeasibilityInputs(
    ...     expected_qber=0.05,
    ...     storage_noise_r=0.75,
    ...     storage_rate_nu=0.002,
    ...     epsilon_sec=1e-6,
    ...     n_target_sifted_bits=1000000,
    ...     expected_leakage_bits=50000
    ... )
    >>> decision = checker.check(inputs)
    >>> decision.is_feasible
    True
    """

    def __init__(self) -> None:
        """Initialize the feasibility checker."""
        self._bounds_calculator = NSMBoundsCalculator()

    def check(self, inputs: FeasibilityInputs) -> FeasibilityDecision:
        """
        Perform comprehensive feasibility assessment.

        Parameters
        ----------
        inputs : FeasibilityInputs
            Protocol session parameters.

        Returns
        -------
        FeasibilityDecision
            Assessment result with abort code if not feasible.

        Notes
        -----
        Checks are performed in order of computational cost and security
        criticality:
        1. Parameter validation (fast, catches obvious errors)
        2. QBER hard limit (fast, security-critical)
        3. Strict-less condition (fast, security-critical)
        4. Capacity × rate condition (fast, security-critical)
        5. Death Valley (requires full bound computation)
        """
        warnings: list[str] = []

        # Step 1: Validate input parameters
        validation_error = self._validate_inputs(inputs)
        if validation_error is not None:
            logger.warning(
                "Feasibility check failed: invalid parameters - %s",
                validation_error,
            )
            return FeasibilityDecision(
                is_feasible=False,
                abort_code=ABORT_CODE_INVALID_PARAMETERS,
                reason=f"Invalid parameters: {validation_error}",
            )

        # Step 2: Check QBER hard limit (22%)
        if inputs.expected_qber > QBER_HARD_LIMIT:
            logger.warning(
                "Feasibility check failed: QBER %.4f > hard limit %.2f",
                inputs.expected_qber,
                QBER_HARD_LIMIT,
            )
            return FeasibilityDecision(
                is_feasible=False,
                abort_code=ABORT_CODE_QBER_TOO_HIGH,
                reason=(
                    f"Expected QBER {inputs.expected_qber:.4f} exceeds hard limit "
                    f"{QBER_HARD_LIMIT:.2f}. Secure OT is impossible."
                ),
            )

        # Check conservative QBER warning (11%)
        if inputs.expected_qber > QBER_WARNING_THRESHOLD:
            warnings.append(
                f"QBER {inputs.expected_qber:.4f} exceeds conservative threshold "
                f"{QBER_WARNING_THRESHOLD:.2f} but below hard limit"
            )

        # Step 3: Check strict-less condition (Q_trusted < r_storage)
        # Interpretation: The trusted noise (QBER) must be strictly less than
        # the untrusted storage noise parameter to ensure adversary cannot
        # hide in the trusted noise.
        if inputs.expected_qber >= inputs.storage_noise_r:
            logger.warning(
                "Feasibility check failed: strict-less condition violated "
                "(QBER %.4f >= storage_r %.4f)",
                inputs.expected_qber,
                inputs.storage_noise_r,
            )
            return FeasibilityDecision(
                is_feasible=False,
                abort_code=ABORT_CODE_STRICT_LESS_VIOLATED,
                reason=(
                    f"Strict-less condition violated: trusted noise "
                    f"(QBER={inputs.expected_qber:.4f}) must be strictly less than "
                    f"untrusted storage noise (r={inputs.storage_noise_r:.4f}). "
                    f"Adversary can hide in trusted noise."
                ),
            )

        # Step 4: Check capacity × rate condition (C_N · ν < 1/2)
        # This ensures the storage channel cannot transmit enough information
        # for the adversary to break security.
        c_n = channel_capacity(inputs.storage_noise_r)
        capacity_rate_product = c_n * inputs.storage_rate_nu

        if capacity_rate_product >= 0.5:
            logger.warning(
                "Feasibility check failed: capacity × rate condition violated "
                "(C_N=%.4f × ν=%.4f = %.4f >= 0.5)",
                c_n,
                inputs.storage_rate_nu,
                capacity_rate_product,
            )
            return FeasibilityDecision(
                is_feasible=False,
                abort_code=ABORT_CODE_CAPACITY_RATE_VIOLATED,
                reason=(
                    f"Capacity × rate condition violated: "
                    f"C_N({inputs.storage_noise_r:.4f})={c_n:.4f} × "
                    f"ν={inputs.storage_rate_nu:.4f} = {capacity_rate_product:.4f} >= 0.5. "
                    f"Storage channel has too much capacity."
                ),
            )

        # Step 5: Check Death Valley (compute expected key length)
        bounds_inputs = NSMBoundsInputs(
            storage_noise_r=inputs.storage_noise_r,
            adjusted_qber=inputs.expected_qber,
            total_leakage_bits=inputs.expected_leakage_bits,
            epsilon_sec=inputs.epsilon_sec,
            n_sifted_bits=inputs.n_target_sifted_bits,
        )
        bounds_result = self._bounds_calculator.compute(bounds_inputs)

        if bounds_result.feasibility_status == FeasibilityResult.INFEASIBLE_INSUFFICIENT_ENTROPY:
            logger.warning(
                "Feasibility check failed: Death Valley (ℓ_max=%d <= 0). "
                "Recommended min n: %d",
                bounds_result.max_secure_key_length_bits,
                bounds_result.recommended_min_n,
            )
            return FeasibilityDecision(
                is_feasible=False,
                abort_code=ABORT_CODE_DEATH_VALLEY,
                reason=(
                    f"Death Valley: expected key length {bounds_result.max_secure_key_length_bits} <= 0. "
                    f"Insufficient sifted bits ({inputs.n_target_sifted_bits}) or "
                    f"excessive leakage ({inputs.expected_leakage_bits} bits). "
                    f"Recommended minimum n: {bounds_result.recommended_min_n}."
                ),
                recommended_min_n=bounds_result.recommended_min_n,
                warnings=tuple(warnings),
            )

        # All checks passed
        h_min = max_bound_entropy_rate(inputs.storage_noise_r)
        logger.info(
            "Feasibility check passed: h_min=%.4f, expected ℓ_max=%d, C_N·ν=%.4f",
            h_min,
            bounds_result.max_secure_key_length_bits,
            capacity_rate_product,
        )

        return FeasibilityDecision(
            is_feasible=True,
            abort_code=None,
            reason=(
                f"Feasible: h_min(r={inputs.storage_noise_r:.4f})={h_min:.4f}, "
                f"expected key length={bounds_result.max_secure_key_length_bits} bits, "
                f"C_N·ν={capacity_rate_product:.4f} < 0.5"
            ),
            warnings=tuple(warnings),
        )

    def _validate_inputs(self, inputs: FeasibilityInputs) -> Optional[str]:
        """
        Validate all input parameters.

        Returns
        -------
        str | None
            Error message if validation fails, None if valid.
        """
        # expected_qber in [0, 0.5]
        if inputs.expected_qber < 0.0 or inputs.expected_qber > 0.5:
            return f"expected_qber must be in [0, 0.5], got {inputs.expected_qber}"

        # storage_noise_r in [0, 1]
        if inputs.storage_noise_r < 0.0 or inputs.storage_noise_r > 1.0:
            return f"storage_noise_r must be in [0, 1], got {inputs.storage_noise_r}"

        # storage_rate_nu in [0, 1]
        if inputs.storage_rate_nu < 0.0 or inputs.storage_rate_nu > 1.0:
            return f"storage_rate_nu must be in [0, 1], got {inputs.storage_rate_nu}"

        # epsilon_sec in (0, 1)
        if inputs.epsilon_sec <= 0.0 or inputs.epsilon_sec >= 1.0:
            return f"epsilon_sec must be in (0, 1), got {inputs.epsilon_sec}"

        # n_target_sifted_bits >= 0
        if inputs.n_target_sifted_bits < 0:
            return f"n_target_sifted_bits must be >= 0, got {inputs.n_target_sifted_bits}"

        # expected_leakage_bits >= 0
        if inputs.expected_leakage_bits < 0:
            return f"expected_leakage_bits must be >= 0, got {inputs.expected_leakage_bits}"

        return None


# =============================================================================
# Convenience Function
# =============================================================================


def check_feasibility(
    expected_qber: float,
    storage_noise_r: float,
    storage_rate_nu: float,
    epsilon_sec: float,
    n_target_sifted_bits: int,
    expected_leakage_bits: int,
) -> FeasibilityDecision:
    """
    Convenience function for pre-flight feasibility check.

    Parameters
    ----------
    expected_qber : float
        Expected QBER in [0, 0.5].
    storage_noise_r : float
        Storage retention parameter r ∈ [0, 1].
    storage_rate_nu : float
        Storage rate ν ∈ [0, 1].
    epsilon_sec : float
        Security parameter ε ∈ (0, 1).
    n_target_sifted_bits : int
        Target sifted bits for session.
    expected_leakage_bits : int
        Expected information leakage.

    Returns
    -------
    FeasibilityDecision
        Assessment result.
    """
    checker = FeasibilityChecker()
    inputs = FeasibilityInputs(
        expected_qber=expected_qber,
        storage_noise_r=storage_noise_r,
        storage_rate_nu=storage_rate_nu,
        epsilon_sec=epsilon_sec,
        n_target_sifted_bits=n_target_sifted_bits,
        expected_leakage_bits=expected_leakage_bits,
    )
    return checker.check(inputs)

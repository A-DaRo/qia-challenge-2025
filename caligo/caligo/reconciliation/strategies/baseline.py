"""
Baseline Reconciliation Strategy (Phase 2).

Implements the Elkouss et al. (2010) rate-compatible reconciliation protocol.

Per Theoretical Report v2 §3 and Implementation Report v2 §5.2:
- Single-shot syndrome transmission
- QBER-based rate selection: R = 1 - f(p*) × h(p*)
- Frame construction via g(x, σ, π) function
- Fixed syndrome leakage: |Σ| = (1 - R_0) × n

References
----------
[1] Elkouss et al. (2010), "Rate Compatible Protocol for Information Reconciliation"
"""

from __future__ import annotations

from typing import Any, Dict, Generator, TYPE_CHECKING

import hashlib
import numpy as np

from caligo.reconciliation.strategies import (
    BlockResult,
    ReconciliationContext,
    ReconciliationStrategy,
)
from caligo.utils.logging import get_logger

if TYPE_CHECKING:
    from caligo.reconciliation.leakage_tracker import LeakageTracker
    from caligo.reconciliation.matrix_manager import MotherCodeManager
    from caligo.reconciliation.strategies.codec import LDPCCodec

logger = get_logger(__name__)


def compute_hash(payload: np.ndarray, seed: int) -> int:
    """
    Compute verification hash for payload.
    
    Parameters
    ----------
    payload : np.ndarray
        Payload bits (uint8).
    seed : int
        Block ID for hash salting.
        
    Returns
    -------
    int
        64-bit hash value.
    """
    hasher = hashlib.sha256()
    hasher.update(payload.tobytes())
    hasher.update(seed.to_bytes(8, 'little'))
    return int.from_bytes(hasher.digest()[:8], 'little')


def build_three_state_llr(
    received_bits: np.ndarray,
    qber: float,
    puncture_mask: np.ndarray,
    shorten_mask: np.ndarray | None = None,
    shorten_values: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build three-state LLR array for decoder initialization.
    
    Per Theoretical Report v2 §3.3:
    - Payload bits: LLR based on QBER (± α)
    - Punctured bits: LLR = 0 (erasure)
    - Shortened bits: LLR = ±∞ (known values)
    
    Parameters
    ----------
    received_bits : np.ndarray
        Received bit array (full frame).
    qber : float
        Channel QBER estimate.
    puncture_mask : np.ndarray
        Boolean mask for punctured positions.
    shorten_mask : np.ndarray, optional
        Boolean mask for shortened positions.
    shorten_values : np.ndarray, optional
        Known values at shortened positions.
        
    Returns
    -------
    np.ndarray
        LLR array (float64).
    """
    n = len(received_bits)
    llr = np.zeros(n, dtype=np.float64)
    
    # Compute LLR magnitude from QBER
    # LLR = log((1-p)/p) where p = QBER
    if qber <= 0 or qber >= 0.5:
        alpha = 2.0  # Conservative fallback
    else:
        alpha = np.log((1 - qber) / qber)
    
    # Three-state initialization
    for i in range(n):
        if puncture_mask[i]:
            # Punctured: erasure (LLR = 0)
            llr[i] = 0.0
        elif shorten_mask is not None and shorten_mask[i]:
            # Shortened: known value (LLR = ±∞)
            llr[i] = +100.0 if shorten_values[i] == 0 else -100.0
        else:
            # Payload: soft information from channel
            llr[i] = +alpha if received_bits[i] == 0 else -alpha
    
    return llr


class BaselineStrategy(ReconciliationStrategy):
    """
    Elkouss et al. (2010) rate-compatible reconciliation.
    
    Single-shot syndrome transmission with QBER-based rate selection.
    
    Per Theoretical Report v2 §3:
    - Requires explicit QBER estimation via sampling (t disclosed bits)
    - Rate selection: R = 1 - f(p*) × h(p*)
    - Frame construction via g(x, σ, π) function
    - Fixed syndrome leakage: |Σ| = (1 - R_0) × n
    
    The QBER estimate must be obtained from sifting/qber.py BEFORE
    calling this strategy. The estimate is passed via context.qber_measured.
    
    Parameters
    ----------
    mother_code : MotherCodeManager
        Singleton mother code manager.
    codec : LDPCCodec
        Numba kernel facade.
    leakage_tracker : LeakageTracker
        Leakage accounting tracker.
    """
    
    def __init__(
        self,
        mother_code: "MotherCodeManager",
        codec: "LDPCCodec",
        leakage_tracker: "LeakageTracker",
    ) -> None:
        self._mother_code = mother_code
        self._codec = codec
        self._leakage_tracker = leakage_tracker
    
    @property
    def requires_qber_estimation(self) -> bool:
        """Baseline requires measured QBER from test bit exchange."""
        return True
    
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Alice computes syndrome and sends to Bob.
        
        Per Theoretical Report v2 §3.2-3.3:
        1. Select rate based on QBER: R = 1 - f(p*) × h(p*)
        2. Compute (s, p) from (R, δ) using Eq. (7)
        3. Construct frame x+ = g(x, σ, π)
        4. Compute syndrome s = H × x+ mod 2
        
        Parameters
        ----------
        payload : np.ndarray
            Alice's payload bits.
        ctx : ReconciliationContext
            Session context.
        block_id : int
            Block identifier.
            
        Yields
        ------
        Dict[str, Any]
            Syndrome message to Bob.
            
        Returns
        -------
        BlockResult
            Reconciliation result.
        """
        # 1. Get measured QBER from context
        qber_estimate = ctx.qber_for_baseline
        
        # 2. Select effective rate based on QBER
        effective_rate = self._select_rate(qber_estimate, ctx.f_crit)
        
        logger.debug(
            f"Block {block_id}: QBER={qber_estimate:.4f} → R_eff={effective_rate:.3f}"
        )
        
        # 3. Get hybrid puncture pattern for this rate
        pattern = self._mother_code.get_pattern(effective_rate)
        
        # 4. Construct frame x+ = g(x, σ, π)
        frame = self._construct_frame(payload, pattern, ctx.frame_size)
        
        # 5. Compute syndrome using Numba bit-packed SpMV kernel
        syndrome = self._codec.encode(frame, pattern)
        
        # 6. Compute verification hash
        hash_value = compute_hash(payload, seed=block_id)
        
        # 7. Record leakage: |Σ| + |Hash|
        self._leakage_tracker.record_block(
            block_id=block_id,
            syndrome_bits=len(syndrome),
            hash_bits=ctx.hash_bits,
        )
        
        logger.debug(
            f"Block {block_id}: Syndrome={len(syndrome)} bits, Hash={ctx.hash_bits} bits"
        )
        
        # 8. Send syndrome message to Bob
        response = yield {
            "kind": "baseline",
            "block_id": block_id,
            "syndrome": syndrome.tolist(),
            "pattern_id": effective_rate,
            "payload_length": len(payload),
            "hash_value": hash_value,
            "qber_channel": qber_estimate,
        }
        
        # 9. Check Bob's response
        verified = response.get("verified", False)
        converged = response.get("converged", False)
        
        return BlockResult(
            corrected_payload=payload,  # Alice already has correct bits
            verified=verified,
            converged=converged,
            iterations_used=0,
            syndrome_leakage=len(syndrome),
            revealed_leakage=0,
            hash_leakage=ctx.hash_bits,
            retry_count=1,
            effective_rate=effective_rate,
        )
    
    def bob_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Bob receives syndrome and decodes using Virtual Graph kernel.
        
        Per Theoretical Report v2 §3.3:
        - Construct frame y+ = g(y, σ, π)
        - Initialize LLRs with three-state builder
        - Decode using BP against syndrome s
        
        Parameters
        ----------
        payload : np.ndarray
            Bob's received payload bits.
        ctx : ReconciliationContext
            Session context.
        block_id : int
            Block identifier.
            
        Yields
        ------
        Dict[str, Any]
            Verification response to Alice.
            
        Returns
        -------
        BlockResult
            Reconciliation result.
        """
        # 1. Receive syndrome message from Alice
        msg = yield {}  # Initial yield to receive first message
        
        if msg.get("kind") != "baseline":
            raise ValueError(f"Expected baseline, got {msg.get('kind')}")
        
        syndrome = np.array(msg["syndrome"], dtype=np.uint8)
        pattern_id = msg["pattern_id"]
        payload_length = msg["payload_length"]
        expected_hash = msg["hash_value"]
        qber_channel = msg["qber_channel"]
        
        logger.debug(
            f"Block {block_id}: Received syndrome ({len(syndrome)} bits) for R_eff={pattern_id:.3f}"
        )
        
        # 2. Get hybrid puncture pattern
        pattern = self._mother_code.get_pattern(pattern_id)
        
        # 3. Construct frame and build LLRs
        frame = self._construct_frame(payload, pattern, ctx.frame_size)
        
        # Build puncture mask
        puncture_mask = (pattern == 1)
        
        llr = build_three_state_llr(
            received_bits=frame,
            qber=qber_channel,
            puncture_mask=puncture_mask,
            shorten_mask=None,
            shorten_values=None,
        )
        
        # 4. Decode using Virtual Graph kernel
        result = self._codec.decode_baseline(
            syndrome, llr, pattern, max_iterations=ctx.max_iterations
        )
        
        # 5. Extract corrected payload
        corrected_payload = result.corrected_bits[:payload_length]
        
        # 6. Verify hash
        computed_hash = compute_hash(corrected_payload, seed=block_id)
        verified = (computed_hash == expected_hash)
        
        logger.debug(
            f"Block {block_id}: Converged={result.converged}, Verified={verified}, Iterations={result.iterations}"
        )
        
        # 7. Send response to Alice
        yield {"verified": verified, "converged": result.converged}
        
        return BlockResult(
            corrected_payload=corrected_payload,
            verified=verified,
            converged=result.converged,
            iterations_used=result.iterations,
            syndrome_leakage=len(syndrome),
            revealed_leakage=0,
            hash_leakage=ctx.hash_bits,
            retry_count=1,
            effective_rate=pattern_id,
        )
    
    def _select_rate(self, qber: float, f_crit: float) -> float:
        """
        Select effective rate using reconciliation efficiency model.
        
        Per Theoretical Report v2 §3.2:
        R = 1 - f(p*) × h(p*)
        
        where h(·) is binary entropy and f(·) ≥ 1 is efficiency.
        
        Parameters
        ----------
        qber : float
            Measured QBER.
        f_crit : float
            Target efficiency.
            
        Returns
        -------
        float
            Selected effective rate.
        """
        from caligo.utils.math import binary_entropy
        
        if qber <= 0 or qber >= 0.5:
            return 0.5  # Fallback to mother rate
        
        h_qber = binary_entropy(qber)
        target_rate = 1.0 - f_crit * h_qber
        
        # Clamp to achievable range
        # NOTE: The decoder struggles with heavy puncturing (>15%).
        # We cap at r_max=0.60 to ensure reliable convergence.
        # This sacrifices efficiency for robustness - acceptable tradeoff
        # until decoder improvements allow higher rates.
        r_min = 0.51
        r_max = 0.60  # Conservative cap for decoder stability
        
        return max(r_min, min(target_rate, r_max))
    
    def _construct_frame(
        self, payload: np.ndarray, pattern: np.ndarray, frame_size: int
    ) -> np.ndarray:
        """
        Construct LDPC frame from payload.
        
        For baseline reconciliation with sifted keys:
        - The payload IS the frame (already frame_size bits after sifting)
        - The pattern is used ONLY for LLR initialization (erasures)
        - No embedding/puncturing at frame construction time
        
        Per Theoretical Report v2 §3.2:
        - Alice's sifted key block → syndrome computation
        - Bob's sifted key block → decoder with erasures at pattern positions
        
        Parameters
        ----------
        payload : np.ndarray
            Sifted key bits (should be frame_size already).
        pattern : np.ndarray
            Puncturing pattern (1=erasure for Bob's LLR, not used here).
        frame_size : int
            Target frame size.
            
        Returns
        -------
        np.ndarray
            Frame for syndrome computation (padded if payload < frame_size).
        """
        frame = np.zeros(frame_size, dtype=np.uint8)
        
        # Copy payload into frame (pad with zeros if shorter than frame_size)
        payload_length = min(len(payload), frame_size)
        frame[:payload_length] = payload[:payload_length]
        
        return frame

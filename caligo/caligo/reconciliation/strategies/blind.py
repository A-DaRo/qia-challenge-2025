"""
Blind Reconciliation Strategy (Phase 2).

Implements the Martinez-Mateo et al. (2012) blind reconciliation protocol
using iterative puncturing/shortening for QBER-free rate discovery.

Per Theoretical Report v2 §4 and Implementation Report v2 §5.3:
- Syndrome computed and transmitted ONCE (Theorem 4.1)
- Leakage: leak_Blind = (1-R_0)×n + h + Σ Δ_i (Corollary 4.1)
- Revelation order fixed by hybrid pattern (security requirement)
- Optional heuristic QBER from compute_qber_erven for NSM-gated optimization
- Hot-Start decoder: messages persist across iterations
- Freeze optimization: revealed bits skip tanh/arctanh

References
----------
[1] Martinez-Mateo et al. (2012), "Blind Reconciliation"
[2] Theoretical Report v2 §4 (Blind Protocol Theory)
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any, Dict, Generator, Optional, TYPE_CHECKING

import numpy as np

from caligo.reconciliation.strategies import (
    BlockResult,
    BlindDecoderState,
    ReconciliationContext,
    ReconciliationStrategy,
)
from caligo.utils.logging import get_logger

if TYPE_CHECKING:
    from caligo.reconciliation.leakage_tracker import LeakageTracker
    from caligo.reconciliation.matrix_manager import MotherCodeManager
    from caligo.reconciliation.strategies.codec import LDPCCodec

logger = get_logger(__name__)


def compute_hash(payload: np.ndarray, seed: int, session_salt: bytes = b"") -> int:
    """
    Compute verification hash for payload only (not padding).
    
    Per Martinez-Mateo §4.1: Hash covers the secret key X, NOT
    the random padding bits used in frame construction.
    
    Parameters
    ----------
    payload : np.ndarray
        Payload bits (uint8) - the actual secret key material.
    seed : int
        Block ID for deterministic differentiation.
    session_salt : bytes, optional
        Session-specific salt for IT-security. If empty, uses
        deterministic mode (for testing/backward compatibility).
        
    Returns
    -------
    int
        64-bit hash value.
        
    Notes
    -----
    When session_salt is provided, uses HMAC-SHA256 for IT-security.
    Otherwise falls back to plain SHA-256 (deterministic, for tests).
    """
    seed_bytes = seed.to_bytes(8, 'little')
    
    if session_salt:
        # IT-secure: HMAC-SHA256 with session salt
        key = session_salt + seed_bytes
        digest = hmac.new(key, payload.tobytes(), hashlib.sha256).digest()
    else:
        # Deterministic mode for testing
        hasher = hashlib.sha256()
        hasher.update(payload.tobytes())
        hasher.update(seed_bytes)
        digest = hasher.digest()
    
    return int.from_bytes(digest[:8], 'little')


class BlindStrategy(ReconciliationStrategy):
    """
    Martinez-Mateo et al. (2012) blind reconciliation.
    
    Iterative protocol without QBER pre-estimation. Bob maintains
    decoder state across iterations; Alice progressively reveals
    punctured values (shortening them).
    
    Per Theoretical Report v2 §4:
    - Syndrome computed and transmitted ONCE (Theorem 4.1)
    - Leakage: leak_Blind = (1-R_0)×n + h + Σ Δ_i (Corollary 4.1)
    - Revelation order fixed by hybrid pattern (security requirement)
    - Optional heuristic QBER from compute_qber_erven for gating
    
    Architecture Notes
    ------------------
    - Uses "Hot-Start" kernel: messages persist across iterations
    - Uses "Freeze" optimization: revealed bits skip tanh/arctanh
    
    Parameters
    ----------
    mother_code : MotherCodeManager
        Singleton mother code manager.
    codec : LDPCCodec
        Numba kernel facade.
    leakage_tracker : LeakageTracker
        Leakage accounting tracker.
    max_blind_iterations : int
        Maximum reveal iterations (t parameter).
    modulation_fraction : float
        Modulation budget δ = (p+s)/n.
    """
    
    def __init__(
        self,
        mother_code: "MotherCodeManager",
        codec: "LDPCCodec",
        leakage_tracker: "LeakageTracker",
        max_blind_iterations: int = 3,
        modulation_fraction: float = 0.1,
    ) -> None:
        self._mother_code = mother_code
        self._codec = codec
        self._leakage_tracker = leakage_tracker
        self._max_iterations = max_blind_iterations
        self._delta = modulation_fraction
    
    @property
    def requires_qber_estimation(self) -> bool:
        """
        Blind does NOT require QBER pre-estimation.
        
        Per Theoretical Report v2 §4: The blind protocol iteratively
        discovers the effective rate without prior QBER knowledge.
        An optional heuristic may be used for NSM-gated optimization.
        
        Returns
        -------
        bool
            Always False for Blind.
        """
        return False
    
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Alice sends syndrome once, then iteratively reveals punctured values.
        
        Per Theoretical Report v2 §4.3:
        - Iteration 1: p=d, s=0 (all modulation bits punctured)
        - Iteration i≥2: Reveal Δ bits, update p←p-Δ, s←s+Δ
        - Termination: Success OR p=0
        
        NSM-Gated Variant (from §4.3):
        - If qber_heuristic provided (from compute_qber_erven), use for:
          a) Permissive starting-rate cap (small s_1 > 0)
          b) Restrictive iteration budget (t=3 default, t>3 for high QBER)
        
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
            Syndrome/reveal messages to Bob.
            
        Returns
        -------
        BlockResult
            Reconciliation result.
        """
        # 1. Compute modulation parameters
        d = int(self._delta * ctx.frame_size)  # Total modulation bits
        
        # 2. NSM-gated starting adjustment (optional)
        qber_estimate = ctx.qber_heuristic
        initial_shortened = 0
        if qber_estimate is not None and qber_estimate > 0.05:
            initial_shortened = self._compute_initial_shortening(
                qber_estimate, ctx.f_crit, d
            )
        
        # 3. Get revelation order from hybrid pattern (deterministic)
        puncture_indices = self._mother_code.get_modulation_indices(d)
        
        # 4. Compute step size: Δ = d/t
        delta_step = max(1, d // self._max_iterations)
        
        # 5. Construct frame per Martinez-Mateo §4.1:
        #    - Payload X (m bits) placed in positions [0, m)
        #    - Random padding R (d bits) placed at puncture_indices
        #    - Frame X̃ = [X, R] where R is INDEPENDENT of X
        #
        # CRITICAL: Punctured positions contain RANDOM bits, NOT payload.
        # When we reveal during blind iterations, we reveal R values,
        # which have no correlation with the secret key X.
        frame = np.zeros(ctx.frame_size, dtype=np.uint8)
        
        # Determine actual capacity and used payload
        # Since d bits are reserved for padding, we can only embed (frame_size - d)
        capacity = ctx.frame_size - d
        used_payload_len = min(len(payload), capacity)
        used_payload = payload[:used_payload_len]
        
        # Place payload in first m positions (excluding puncture positions)
        # Create mask for non-punctured positions
        puncture_set = set(puncture_indices.tolist())
        payload_idx = 0
        for i in range(ctx.frame_size):
            if i not in puncture_set and payload_idx < used_payload_len:
                frame[i] = used_payload[payload_idx]
                payload_idx += 1
        
        # Generate deterministic random padding for punctured positions
        # Seed ensures Alice and Bob can independently verify frame layout
        padding_seed = block_id ^ 0xB11DABC  # Deterministic per-block
        padding_rng = np.random.default_rng(padding_seed & 0xFFFFFFFF)
        padding_values = padding_rng.integers(0, 2, size=len(puncture_indices), dtype=np.uint8)
        frame[puncture_indices] = padding_values
        
        # 6. Compute syndrome ONCE (Theorem 4.1: syndrome reuse)
        # Use mother pattern (no puncturing for syndrome computation)
        mother_pattern = np.zeros(ctx.frame_size, dtype=np.uint8)
        syndrome = self._codec.encode(frame, mother_pattern)
        
        # Hash covers ONLY the used payload, NOT the padding
        # This is critical: revealed padding has no entropy correlation with key
        hash_value = compute_hash(used_payload, seed=block_id)
        
        # 7. Record initial syndrome leakage
        # Note: len(syndrome) is already in bits (codec returns uint8 bit array)
        self._leakage_tracker.record_block(
            block_id=block_id,
            syndrome_bits=len(syndrome),
            hash_bits=ctx.hash_bits,
            n_shortened=initial_shortened,
            frame_size=ctx.frame_size,
        )
        
        # 8. Prepare initial revealed indices/values
        if initial_shortened > 0:
            revealed_indices = puncture_indices[:initial_shortened].tolist()
            revealed_values = padding_values[:initial_shortened].tolist()
        else:
            revealed_indices = []
            revealed_values = []
        
        # 9. Send initial syndrome + any initial shortening
        response = yield {
            "kind": "blind",
            "block_id": block_id,
            "syndrome": syndrome.tolist(),
            "puncture_indices": puncture_indices.tolist(),
            "payload_length": used_payload_len,
            "hash_value": hash_value,
            "qber_prior": ctx.qber_for_blind_gating,
            "iteration": 1,
            "revealed_indices": revealed_indices,
            "revealed_values": revealed_values,
        }
        
        # 10. Iterative reveal loop
        # Per Theoretical Report v2 §4.3: Continue revealing until verified
        # OR max iterations reached. The decoder may converge to wrong codewords
        # with pure erasures - we need more revealed bits to constrain the solution.
        iteration = 1
        total_revealed = initial_shortened
        
        # Handle initial verification response
        verified = response.get("verified", False)
        converged = response.get("converged", False)
        
        # 6. Send initial response
        # response = yield {"verified": verified, "converged": result.converged} 
        # (This is handled in the surrounding loop structure usually, but here 
        # we are essentially in the body of alice_reconcile_block generator)
        
        # Loop if not verified and Alice/Bob not done
        while not verified and iteration < self._max_iterations:
            
            iteration += 1
            
            # Select next batch of punctured bits to reveal
            reveal_start = initial_shortened + (iteration - 1) * delta_step
            reveal_end = min(initial_shortened + iteration * delta_step, d)
            reveal_indices = puncture_indices[reveal_start:reveal_end]
            reveal_values = padding_values[reveal_start:reveal_end]
            
            total_revealed += len(reveal_indices)
            
            # Record additional leakage (Δ_i bits)
            self._leakage_tracker.record_reveal(
                block_id=block_id,
                iteration=iteration,
                revealed_bits=len(reveal_indices),
            )
            
            # Send reveal message (NO new syndrome!)
            response = yield {
                "kind": "blind_reveal",
                "block_id": block_id,
                "iteration": iteration,
                "revealed_indices": reveal_indices.tolist(),
                "revealed_values": reveal_values.tolist(),
            }
            
            verified = response.get("verified", False)
            converged = response.get("converged", False)
        # Get QBER estimated by Bob (based on correction diff)
        estimated_qber = response.get("qber", 0.11)
        
        return BlockResult(
            corrected_payload=used_payload,
            verified=verified,
            converged=converged,
            iterations_used=0,
            syndrome_leakage=len(syndrome),  # Already in bits (uint8 array)
            revealed_leakage=total_revealed,
            hash_leakage=ctx.hash_bits,
            retry_count=iteration,
            effective_rate=self._compute_effective_rate(d, total_revealed),
            estimated_qber=estimated_qber,
        )

    def bob_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Bob decodes with Hot-Start persistence across reveal iterations.
        
        Per Theoretical Report v2 §4:
        - Iteration 1: Full decode attempt with p=d, s=0
        - Iteration i: Update LLRs with revealed bits, Hot-Start decode
        - Hot-Start: Reuse edge messages from previous iteration
        - Freeze: Revealed bits have LLR=±∞, skip tanh/arctanh
        
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
        # 1. Receive initial syndrome message
        msg = yield {}
        
        if msg.get("kind") != "blind":
            raise ValueError(f"Expected blind, got {msg.get('kind')}")
        
        syndrome = np.array(msg["syndrome"], dtype=np.uint8)
        puncture_indices = np.array(msg["puncture_indices"], dtype=np.int64)
        payload_length = msg["payload_length"]
        expected_hash = msg["hash_value"]
        qber_prior = msg.get("qber_prior", 0.05)
        revealed_indices = np.array(msg.get("revealed_indices", []), dtype=np.int64)
        revealed_values = np.array(msg.get("revealed_values", []), dtype=np.uint8)
        
        logger.debug(
            f"Block {block_id}: Blind initial syndrome ({len(syndrome)} bytes), "
            f"{len(puncture_indices)} punctured, {len(revealed_indices)} pre-revealed"
        )
        
        # 2. Initialize decoder state
        state = self._initialize_decoder_state(
            payload=payload,
            puncture_indices=puncture_indices,
            revealed_indices=revealed_indices,
            revealed_values=revealed_values,
            syndrome=syndrome,
            qber=qber_prior,
            frame_size=ctx.frame_size,
        )
        
        # 3. First decode attempt
        result = self._codec.decode_blind(
            syndrome,
            state.llr,
            state.messages,
            state.frozen_mask,
            max_iterations=ctx.max_iterations,
        )
        
        # 4. Update state with decoder output
        state = BlindDecoderState(
            llr=result.llr if hasattr(result, 'llr') else state.llr,
            messages=result.messages,
            puncture_indices=state.puncture_indices,
            shortened_indices=state.shortened_indices,
            shortened_values=state.shortened_values,
            frozen_mask=state.frozen_mask,
            iteration=1,
            syndrome=syndrome,
        )
        
        # 5. Extract corrected payload from non-punctured positions
        # Per Martinez-Mateo frame layout: payload at [0..m) excluding punctured
        corrected_payload = self._extract_payload_from_frame(
            result.corrected_bits, puncture_indices, payload_length
        )
        computed_hash = compute_hash(corrected_payload, seed=block_id)
        verified = (computed_hash == expected_hash)
        
        # Estimate QBER by comparing frame vs corrected (on payload part roughly)
        # Note: Bob doesn't know original payload, but if verified=True, we assume corrected is clean.
        # We can just report how many bits changed in the payload section.
        # But for 'QBER' we typically mean error rate relative to raw.
        # So we compare Bob's RAW payload vs Corrected payload.
        # We should use the payload_length truncation applied in init_decoder_state.
        
        # We only compare the bits that were actually part of the frame
        raw_payload_segment = payload[:payload_length]
        
        if len(raw_payload_segment) == len(corrected_payload):
            # Safe comparison
            diff_bits = np.count_nonzero(raw_payload_segment != corrected_payload)
            estimated_qber = diff_bits / len(raw_payload_segment) if len(raw_payload_segment) > 0 else 0.0
        else:
            logger.warning(f"Payload length mismatch for QBER est: {len(raw_payload_segment)} vs {len(corrected_payload)}")
            estimated_qber = 0.5

        total_revealed = len(revealed_indices)
        iteration = 1
        
        # 6. Send initial response
        response = yield {"verified": verified, "converged": result.converged, "qber": estimated_qber}
        
        # 7. Handle reveal iterations
        while response.get("kind") == "blind_reveal":
            iteration += 1
            
            new_revealed_indices = np.array(response["revealed_indices"], dtype=np.int64)
            new_revealed_values = np.array(response["revealed_values"], dtype=np.uint8)
            
            total_revealed += len(new_revealed_indices)
            
            # Update state with newly revealed bits
            state = self._update_state_with_reveals(
                state, new_revealed_indices, new_revealed_values
            )
            
            # Hot-Start decode with updated LLRs (decode_blind supports Hot-Start)
            result = self._codec.decode_blind(
                syndrome,
                state.llr,
                state.messages,
                state.frozen_mask,
                max_iterations=ctx.max_iterations,
            )
            
            # Update messages for next iteration
            state = BlindDecoderState(
                llr=result.llr if hasattr(result, 'llr') else state.llr,
                messages=result.messages,
                puncture_indices=state.puncture_indices,
                shortened_indices=np.concatenate([state.shortened_indices, new_revealed_indices]),
                shortened_values=np.concatenate([state.shortened_values, new_revealed_values]),
                frozen_mask=state.frozen_mask,
                iteration=iteration,
                syndrome=syndrome,
            )
            
            # Verify again - extract payload from non-punctured positions
            corrected_payload = self._extract_payload_from_frame(
                result.corrected_bits, state.puncture_indices, payload_length
            )
            computed_hash = compute_hash(corrected_payload, seed=block_id)
            verified = (computed_hash == expected_hash)
            
            # Re-estimate QBER if verified changed (or even if not)
            if verified:
                raw_payload_segment = payload[:payload_length]
                diff_bits = np.count_nonzero(raw_payload_segment != corrected_payload)
                estimated_qber = diff_bits / len(raw_payload_segment) if len(raw_payload_segment) > 0 else 0.0

            # Send response
            response = yield {"verified": verified, "converged": result.converged, "qber": estimated_qber}
        
        return BlockResult(
            corrected_payload=corrected_payload,
            verified=verified,
            converged=result.converged,
            iterations_used=result.iterations,
            syndrome_leakage=len(syndrome),  # Already in bits (uint8 array)
            revealed_leakage=total_revealed,
            hash_leakage=ctx.hash_bits,
            retry_count=iteration,
            effective_rate=self._compute_effective_rate(len(puncture_indices), total_revealed),
            estimated_qber=estimated_qber,
        )
    
    def _initialize_decoder_state(
        self,
        payload: np.ndarray,
        puncture_indices: np.ndarray,
        revealed_indices: np.ndarray,
        revealed_values: np.ndarray,
        syndrome: np.ndarray,
        qber: float,
        frame_size: int,
    ) -> BlindDecoderState:
        """
        Initialize BlindDecoderState for first decode attempt.
        
        Parameters
        ----------
        payload : np.ndarray
            Bob's received payload bits.
        puncture_indices : np.ndarray
            All punctured positions.
        revealed_indices : np.ndarray
            Initially revealed (shortened) positions.
        revealed_values : np.ndarray
            Values at initially revealed positions.
        syndrome : np.ndarray
            Alice's syndrome.
        qber : float
            Prior QBER estimate.
        frame_size : int
            LDPC frame size.
            
        Returns
        -------
        BlindDecoderState
            Initialized decoder state.
        """
        # Build frame with Bob's payload at non-punctured positions
        # (matching Alice's frame layout per Martinez-Mateo §4.1)
        frame = np.zeros(frame_size, dtype=np.uint8)
        puncture_set = set(puncture_indices.tolist())
        payload_len = min(len(payload), frame_size - len(puncture_indices))
        
        # Place payload at non-punctured positions
        payload_idx = 0
        for i in range(frame_size):
            if i not in puncture_set and payload_idx < payload_len:
                frame[i] = payload[payload_idx]
                payload_idx += 1
        
        # Compute channel LLR magnitude
        qber_clamped = np.clip(qber, 1e-6, 0.5 - 1e-6)
        alpha = np.log((1 - qber_clamped) / qber_clamped)
        
        # Initialize LLR array
        llr = np.zeros(frame_size, dtype=np.float64)
        
        # Non-punctured positions: channel LLR from Bob's bits
        revealed_set = set(revealed_indices.tolist())
        for i in range(frame_size):
            if i not in puncture_set:
                llr[i] = +alpha if frame[i] == 0 else -alpha
        
        # Create frozen mask
        frozen_mask = np.zeros(frame_size, dtype=np.bool_)
        
        # Punctured positions: erasure (LLR=0) unless revealed
        for idx in puncture_indices:
            if idx not in revealed_set:
                llr[idx] = 0.0
        
        # Revealed positions: frozen with known values (LLR=±∞)
        for idx, val in zip(revealed_indices, revealed_values):
            llr[idx] = +100.0 if val == 0 else -100.0
            frozen_mask[idx] = True
        
        # Initialize edge messages to zeros (no prior)
        # Messages array is 1D with shape (2 * n_edges,) containing both c2v and v2c messages
        n_edges = self._mother_code.compiled_topology.n_edges
        messages = np.zeros(2 * n_edges, dtype=np.float64)
        
        return BlindDecoderState(
            llr=llr,
            messages=messages,
            puncture_indices=puncture_indices,
            shortened_indices=revealed_indices.copy(),
            shortened_values=revealed_values.copy(),
            frozen_mask=frozen_mask,
            iteration=0,
            syndrome=syndrome,
        )
    
    def _update_state_with_reveals(
        self,
        state: BlindDecoderState,
        revealed_indices: np.ndarray,
        revealed_values: np.ndarray,
    ) -> BlindDecoderState:
        """
        Update decoder state with newly revealed bits.
        
        Updates LLR to ±∞ for revealed positions and marks them frozen.
        Preserves edge messages for Hot-Start continuation.
        
        Parameters
        ----------
        state : BlindDecoderState
            Current decoder state.
        revealed_indices : np.ndarray
            Newly revealed positions.
        revealed_values : np.ndarray
            Values at newly revealed positions.
            
        Returns
        -------
        BlindDecoderState
            Updated state with new reveals.
        """
        llr = state.llr.copy()
        frozen_mask = state.frozen_mask.copy()
        
        # Update LLR and frozen mask for revealed positions
        for idx, val in zip(revealed_indices, revealed_values):
            llr[idx] = +100.0 if val == 0 else -100.0
            frozen_mask[idx] = True
        
        return BlindDecoderState(
            llr=llr,
            messages=state.messages,  # Preserve for Hot-Start
            puncture_indices=state.puncture_indices,
            shortened_indices=state.shortened_indices,
            shortened_values=state.shortened_values,
            frozen_mask=frozen_mask,
            iteration=state.iteration,
            syndrome=state.syndrome,
        )
    
    def _extract_payload_from_frame(
        self,
        frame: np.ndarray,
        puncture_indices: np.ndarray,
        payload_length: int,
    ) -> np.ndarray:
        """
        Extract payload bits from frame (excluding punctured positions).
        
        Per Martinez-Mateo §4.1, the frame layout is:
        - Payload bits at non-punctured positions
        - Random padding at punctured positions
        
        This method extracts only the payload portion.
        
        Parameters
        ----------
        frame : np.ndarray
            Full decoded frame.
        puncture_indices : np.ndarray
            Positions containing padding (to skip).
        payload_length : int
            Expected payload length.
            
        Returns
        -------
        np.ndarray
            Extracted payload bits.
        """
        puncture_set = set(puncture_indices.tolist())
        payload = np.zeros(payload_length, dtype=np.uint8)
        
        payload_idx = 0
        for i in range(len(frame)):
            if i not in puncture_set and payload_idx < payload_length:
                payload[payload_idx] = frame[i]
                payload_idx += 1
        
        return payload
    
    def _construct_frame_with_padding(
        self,
        payload: np.ndarray,
        puncture_indices: np.ndarray,
        frame_size: int,
    ) -> np.ndarray:
        """
        Construct frame with payload and random padding at punctured positions.
        
        Parameters
        ----------
        payload : np.ndarray
            Payload bits.
        puncture_indices : np.ndarray
            Positions to fill with padding.
        frame_size : int
            Target frame size.
            
        Returns
        -------
        np.ndarray
            Constructed frame.
        """
        frame = np.zeros(frame_size, dtype=np.uint8)
        
        # Fill payload
        payload_len = min(len(payload), frame_size)
        frame[:payload_len] = payload[:payload_len]
        
        # Fill punctured positions with random padding (deterministic)
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        frame[puncture_indices] = rng.integers(0, 2, size=len(puncture_indices), dtype=np.uint8)
        
        return frame
    
    def _compute_initial_shortening(
        self, qber: float, f_crit: float, d: int
    ) -> int:
        """
        Compute permissive initial shortening s_1 for NSM-gated variant.
        
        Per Theoretical Report v2 §4.3: choose s_1 as smallest value
        that brings initial effective rate below conservative cap.
        
        Parameters
        ----------
        qber : float
            Heuristic QBER estimate.
        f_crit : float
            Target efficiency.
        d : int
            Total modulation bits.
            
        Returns
        -------
        int
            Initial shortening count.
        """
        from caligo.utils.math import binary_entropy
        
        h_qber = binary_entropy(qber)
        target_rate = 1.0 - f_crit * h_qber
        
        # Conservative: start closer to target rate
        return min(int(d * 0.1), d // self._max_iterations)
    
    def _compute_effective_rate(self, d: int, shortened: int) -> float:
        """
        Compute effective rate given shortening.
        
        R_eff increases as we shorten more bits.
        
        Parameters
        ----------
        d : int
            Total modulation bits.
        shortened : int
            Number of shortened bits.
            
        Returns
        -------
        float
            Effective code rate.
        """
        # R_eff ≈ (R_0) / (1 - s/n) for small modulation
        # Simplified approximation
        if shortened >= d:
            return 0.5  # Mother rate
        return 0.5 + 0.5 * (shortened / d) * 0.4  # Linear interpolation to ~0.7

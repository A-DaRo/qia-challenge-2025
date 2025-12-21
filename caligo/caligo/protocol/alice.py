"""Alice role program for Caligo Phase E."""

from __future__ import annotations

from typing import Any, Dict, Generator, Tuple

import numpy as np
from bitarray import bitarray

from caligo.amplification import (
    NSMEntropyCalculator,
    OTOutputFormatter,
    SecureKeyLengthCalculator,
    ToeplitzHasher,
)
from caligo.connection.envelope import MessageType
from caligo.protocol.base import CaligoProgram, ProtocolParameters
from caligo.reconciliation import constants as recon_constants
from caligo.reconciliation.factory import (
    ReconciliationConfig,
    ReconciliationType,
    create_strategy,
)
from caligo.reconciliation.leakage_tracker import LeakageTracker, compute_safety_cap
from caligo.reconciliation.strategies import ReconciliationContext
from caligo.sifting.commitment import SHA256Commitment
from caligo.sifting.qber import QBEREstimator
from caligo.sifting.sifter import Sifter
from caligo.types.exceptions import EntropyDepletedError, SecurityError
from caligo.types.keys import AliceObliviousKey
from caligo.utils.bitarray_utils import bitarray_from_numpy, bitarray_to_numpy
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class AliceProgram(CaligoProgram):
    """Alice's (sender) program."""

    PEER = "Bob"
    ROLE = "alice"

    def __init__(self, params: ProtocolParameters) -> None:
        super().__init__(params=params)
        self._commitment = SHA256Commitment()
        self._sifter = Sifter()

        # Adjust security parameter for small-scale simulations to avoid
        # false positives from finite-size penalties.
        epsilon_sec = 1e-10
        if params.num_pairs < 10000:
            epsilon_sec = 1e-1
            logger.warning(
                f"Using relaxed epsilon_sec={epsilon_sec} for small simulation "
                f"(n={params.num_pairs})"
            )

        self._qber_estimator = QBEREstimator(epsilon_sec=epsilon_sec)

    def _run_protocol(self, context) -> Generator[Any, None, Dict[str, Any]]:
        assert self._ordered_socket is not None

        # ------------------------------------------------------------------
        # Phase I: Quantum (implemented minimally; results are exchanged via
        # classical messages for now).
        # ------------------------------------------------------------------
        alice_outcomes, alice_bases = yield from self._phase1_quantum(context)

        # ------------------------------------------------------------------
        # Phase II: Sifting (+ optional QBER) using ordered messaging.
        # ------------------------------------------------------------------
        sifting_payload, timing_ok = yield from self._phase2_sifting(
            alice_outcomes=alice_outcomes,
            alice_bases=alice_bases,
        )

        # ------------------------------------------------------------------
        # Phase III: Reconciliation (message-based: Alice sends syndromes).
        # ------------------------------------------------------------------
        reconciled_bits, total_syndrome_bits, verified_positions = yield from self._phase3_reconcile(
            alice_bits=sifting_payload["alice_bits"],
            qber_observed=float(sifting_payload["qber_estimate"]),
            qber_adjusted=float(sifting_payload["qber_adjusted"]),
        )

        # Filter partition indices to only include verified positions.
        # The reconciled_bits correspond to verified_positions in the original key.
        verified_set = set(verified_positions.tolist())
        original_matching = np.array(sifting_payload["matching_indices"], dtype=np.int64)
        original_i0 = np.array(sifting_payload["i0_indices"], dtype=np.int64)
        original_i1 = np.array(sifting_payload["i1_indices"], dtype=np.int64)

        # Filter to keep only verified positions
        verified_matching = np.array([i for i in original_matching if i in verified_set], dtype=np.int64)
        verified_i0 = np.array([i for i in original_i0 if i in verified_set], dtype=np.int64)
        verified_i1 = np.array([i for i in original_i1 if i in verified_set], dtype=np.int64)

        # ------------------------------------------------------------------
        # Phase IV: Amplification (Alice computes S0,S1 and sends seeds).
        # ------------------------------------------------------------------
        alice_key, key_length, entropy_consumed, entropy_rate, seeds = self._phase4_amplify(
            reconciled_bits=reconciled_bits,
            total_syndrome_bits=total_syndrome_bits,
            matching_indices=verified_matching,
            i0_indices=verified_i0,
            i1_indices=verified_i1,
        )

        yield from self._ordered_socket.send(
            MessageType.TOEPLITZ_SEED,
            {
                "key_length": int(key_length),
                "seed_0": seeds[0].hex(),
                "seed_1": seeds[1].hex(),
            },
        )

        return {
            "role": self.ROLE,
            "aborted": False,
            "alice_key": alice_key,
            "qber": float(sifting_payload["qber_adjusted"]),
            "key_length": int(key_length),
            "entropy_consumed": float(entropy_consumed),
            "entropy_rate": float(entropy_rate),
            "timing_compliant": bool(timing_ok),
            "total_rounds": int(self.params.num_pairs),
        }

    def _phase1_quantum(
        self, context
    ) -> Generator[Any, None, Tuple[np.ndarray, np.ndarray]]:
        """Generate and measure EPR pairs (Alice side)."""

        if self.params.precomputed_epr is not None:
            n = int(self.params.num_pairs)
            outcomes = np.asarray(self.params.precomputed_epr.alice_outcomes, dtype=np.uint8)
            bases = np.asarray(self.params.precomputed_epr.alice_bases, dtype=np.uint8)
            if len(outcomes) != n or len(bases) != n:
                raise ValueError(
                    "precomputed_epr length mismatch: "
                    f"expected n={n}, got outcomes={len(outcomes)} bases={len(bases)}"
                )
            self._timing_barrier.mark_quantum_complete()
            return outcomes, bases

        from caligo.quantum import BasisSelector, MeasurementExecutor

        epr_socket = context.epr_sockets[self.PEER]
        basis_selector = BasisSelector()
        meas = MeasurementExecutor()

        n = int(self.params.num_pairs)
        bases = basis_selector.select_batch(n)
        outcomes = np.zeros(n, dtype=np.uint8)

        # Use the same pattern as SquidASM's QKD example:
        # generate and measure one EPR pair per round to preserve ordering.
        for round_id in range(n):
            q = epr_socket.create_keep(1)[0]
            outcomes[round_id] = yield from meas.measure_qubit(
                qubit=q,
                basis=int(bases[round_id]),
                round_id=round_id,
                context=context,
            )

        # Mark quantum phase completion as the reference for Δt.
        self._timing_barrier.mark_quantum_complete()

        return outcomes.astype(np.uint8), bases.astype(np.uint8)

    def _phase2_sifting(
        self, alice_outcomes: np.ndarray, alice_bases: np.ndarray
    ) -> Generator[Any, None, Tuple[Dict[str, Any], bool]]:
        assert self._ordered_socket is not None

        # 1) Receive Bob's commitment.
        commit_msg = yield from self._ordered_socket.recv(MessageType.DETECTION_COMMITMENT)
        commitment = bytes.fromhex(str(commit_msg["commitment"]))

        # 2) Enforce Δt before revealing bases.
        yield from self._timing_barrier.wait_delta_t()
        timing_ok = self._timing_barrier.timing_compliant

        # 3) Reveal Alice bases.
        yield from self._ordered_socket.send(
            MessageType.BASIS_REVEAL,
            {"bases": alice_bases.tobytes().hex()},
        )

        # 4) Receive Bob opening.
        opening_msg = yield from self._ordered_socket.recv(MessageType.COMMITMENT_OPENING)
        nonce = bytes.fromhex(str(opening_msg["nonce"]))
        bob_outcomes = np.frombuffer(
            bytes.fromhex(str(opening_msg["outcomes"])), dtype=np.uint8
        )
        bob_bases = np.frombuffer(
            bytes.fromhex(str(opening_msg["bases"])), dtype=np.uint8
        )

        # Verify commitment binds Bob's measurement record.
        data_bytes = np.concatenate([bob_outcomes, bob_bases]).astype(np.uint8).tobytes()
        self._commitment.verify(commitment=commitment, nonce=nonce, data=data_bytes)

        logger.error(f"DEBUG: Alice outcomes (first 20): {alice_outcomes[:20]}")
        logger.error(f"DEBUG: Alice bases (first 20): {alice_bases[:20]}")
        logger.error(f"DEBUG: Bob outcomes (first 20): {bob_outcomes[:20]}")
        logger.error(f"DEBUG: Bob bases (first 20): {bob_bases[:20]}")

        # Compute sifted keys.
        alice_sift, bob_sift = self._sifter.compute_sifted_key(
            alice_bases=alice_bases,
            alice_outcomes=alice_outcomes,
            bob_bases=bob_bases,
            bob_outcomes=bob_outcomes,
        )

        # Optional test subset + QBER estimation.
        # Baseline reconciliation requires QBER; blind does not.
        key_indices = alice_sift.matching_indices
        test_indices = np.array([], dtype=np.int64)
        if self.params.reconciliation.requires_qber_estimation:
            # Use larger test fraction for small simulations to reduce statistical fluctuation.
            test_fraction = 0.1
            if self.params.num_pairs < 10000:
                test_fraction = 0.3

            test_indices, key_indices = self._sifter.select_test_subset(
                matching_indices=alice_sift.matching_indices,
                test_fraction=test_fraction,
                min_test_size=1,
            )

            yield from self._ordered_socket.send(
                MessageType.INDEX_LISTS,
                {"test_indices": test_indices.astype(int).tolist()},
            )

            test_outcomes_msg = yield from self._ordered_socket.recv(MessageType.TEST_OUTCOMES)
            bob_test_bits = np.frombuffer(
                bytes.fromhex(str(test_outcomes_msg["test_bits"])), dtype=np.uint8
            )
            alice_test_bits = alice_outcomes[test_indices]

            qber = self._qber_estimator.estimate(
                alice_test_bits=alice_test_bits,
                bob_test_bits=bob_test_bits,
                key_size=len(key_indices),
            )
            qber_estimate = float(qber.observed_qber)
            qber_adjusted = float(qber.adjusted_qber)
            finite_size_penalty = float(qber.mu_penalty)
            test_set_size = int(qber.num_test_bits)
        else:
            # Blind reconciliation: provide a prior from NSM parameters.
            qber_estimate = float(self.params.nsm_params.qber_conditional)
            qber_adjusted = qber_estimate
            finite_size_penalty = 0.0
            test_set_size = 0

        # Filter sifted keys down to key_indices only.
        # Build an index map from original index -> sifted position.
        idx_map = {orig: i for i, orig in enumerate(alice_sift.matching_indices)}
        key_positions = [idx_map[idx] for idx in key_indices]
        alice_key_bits = bitarray([alice_sift.sifted_bits[p] for p in key_positions])
        bob_key_bits = bitarray([bob_sift.sifted_bits[p] for p in key_positions])

        # Also filter I0/I1 partitions down to key indices.
        i0_indices = np.intersect1d(alice_sift.i0_indices, key_indices)
        i1_indices = np.intersect1d(alice_sift.i1_indices, key_indices)

        return (
            {
                "alice_bits": alice_key_bits,
                "matching_indices": key_indices.astype(int).tolist(),
                "i0_indices": i0_indices.astype(int).tolist(),
                "i1_indices": i1_indices.astype(int).tolist(),
                "test_set_indices": test_indices.astype(int).tolist(),
                "qber_estimate": float(qber_estimate),
                "qber_adjusted": float(qber_adjusted),
                "finite_size_penalty": float(finite_size_penalty),
                "test_set_size": int(test_set_size),
            },
            timing_ok,
        )

    def _phase3_reconcile(
        self, alice_bits: bitarray, qber_observed: float, qber_adjusted: float
    ) -> Generator[Any, None, Tuple[bitarray, int, np.ndarray]]:
        """
        Execute reconciliation (Phase III) via Strategy Pattern.

        Per Implementation Report v2 §10.2.6:
        - Delegates to ReconciliationStrategy (Baseline or Blind)
        - Role class kept thin (< 10 lines per protocol)
        - YAML-driven runtime switching

        Parameters
        ----------
        alice_bits : bitarray
            Alice's sifted key bits.
        qber_observed : float
            Measured QBER from test bits (for decoder LLR).
        qber_adjusted : float
            Conservative QBER with finite-size penalty (for rate selection).

        Yields
        ------
        Messages to/from ordered socket.

        Returns
        -------
        Tuple[bitarray, int, np.ndarray]
            Reconciled key bits, total syndrome bits leaked, and verified bit indices.
        """
        assert self._ordered_socket is not None

        alice_arr = bitarray_to_numpy(alice_bits)
        frame_size = int(self.params.reconciliation.frame_size)

        # 1. Build ReconciliationConfig from protocol parameters
        config = ReconciliationConfig(
            reconciliation_type=self.params.reconciliation.reconciliation_type,
            frame_size=frame_size,
            max_iterations=int(self.params.reconciliation.max_iterations),
            max_blind_rounds=int(self.params.reconciliation.max_blind_rounds),
        )

        # 2. Create dependencies (injected into strategy)
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        from caligo.reconciliation.strategies.codec import LDPCCodec

        mother_code = MotherCodeManager.from_config()
        codec = LDPCCodec(mother_code)
        safety_cap = compute_safety_cap(len(alice_arr), qber_adjusted)
        leakage_tracker = LeakageTracker(safety_cap=safety_cap, abort_on_exceed=True)

        # 3. Create strategy via factory (P3.1)
        strategy = create_strategy(config, mother_code, codec, leakage_tracker)

        # 4. Build ReconciliationContext with QBER flow (P3.5/P3.6)
        ctx = ReconciliationContext(
            session_id=id(self),
            frame_size=frame_size,
            mother_rate=0.5,
            max_iterations=config.max_iterations,
            hash_bits=64,
            f_crit=1.16,  # Per Theoretical Report v2 §2.3
            qber_measured=qber_adjusted if strategy.requires_qber_estimation else None,
            qber_heuristic=qber_observed if not strategy.requires_qber_estimation else None,
            modulation_delta=0.44,  # Per Theoretical Report v2 Remark 2.1
        )

        # 5. Partition key into blocks
        from caligo.reconciliation.orchestrator import partition_key
        alice_blocks = partition_key(alice_arr, frame_size)

        # 6. Reconcile each block via strategy generator
        total_leakage = 0
        reconciled_parts: list[np.ndarray] = []
        verified_indices: list[int] = []

        for block_id, alice_block in enumerate(alice_blocks):
            # Drive the strategy generator via network socket
            result = yield from self._drive_alice_strategy(
                strategy, alice_block, ctx, block_id
            )

            if result.verified:
                reconciled_parts.append(result.corrected_payload)
                start = block_id * frame_size
                verified_indices.extend(range(start, start + len(result.corrected_payload)))
                total_leakage += result.total_leakage
            else:
                logger.warning(f"Block {block_id} failed verification - excluding from final key")

        if len(reconciled_parts) == 0:
            raise SecurityError("All reconciliation blocks failed verification")

        reconciled_np = np.concatenate(reconciled_parts) if reconciled_parts else np.array([], dtype=np.uint8)
        return bitarray_from_numpy(reconciled_np), int(total_leakage), np.array(verified_indices, dtype=np.int64)

    def _drive_alice_strategy(
        self,
        strategy,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Any, None, Any]:
        """
        Drive Alice's strategy generator via network socket.

        Bridges the strategy's generator interface with the ordered socket
        for message exchange with Bob.

        Parameters
        ----------
        strategy : ReconciliationStrategy
            Active strategy instance (Baseline or Blind).
        payload : np.ndarray
            Alice's payload bits for this block.
        ctx : ReconciliationContext
            Session context.
        block_id : int
            Block identifier.

        Yields
        ------
        Messages to/from ordered socket.

        Returns
        -------
        BlockResult
            Reconciliation result for this block.
        """
        gen = strategy.alice_reconcile_block(payload, ctx, block_id)
        
        # First call returns the outgoing message
        try:
            outgoing = next(gen)
        except StopIteration as e:
            return e.value

        # Loop: send message, receive response, send to generator
        while True:
            # Send outgoing message
            yield from self._ordered_socket.send(MessageType.SYNDROME, outgoing)
            
            # Receive response
            resp = yield from self._ordered_socket.recv(MessageType.SYNDROME_RESPONSE)
            
            if int(resp.get("block_id", -1)) != int(block_id):
                raise SecurityError("Reconciliation block_id mismatch")
            
            # Send response to generator
            try:
                outgoing = gen.send(resp)
            except StopIteration as e:
                return e.value

    def _phase4_amplify(
        self,
        reconciled_bits: bitarray,
        total_syndrome_bits: int,
        matching_indices: np.ndarray,
        i0_indices: np.ndarray,
        i1_indices: np.ndarray,
    ) -> Tuple[AliceObliviousKey, int, float, float, Tuple[bytes, bytes]]:
        reconciled_arr = bitarray_to_numpy(reconciled_bits)

        entropy_calc = NSMEntropyCalculator(
            storage_noise_r=float(self.params.nsm_params.storage_noise_r)
        )
        key_length_calc = SecureKeyLengthCalculator(entropy_calculator=entropy_calc)
        detailed = key_length_calc.compute_detailed(
            reconciled_length=int(len(reconciled_arr)),
            syndrome_leakage=int(total_syndrome_bits),
        )

        if detailed.final_length <= 0:
            raise EntropyDepletedError("Key extraction not viable (Death Valley)")

        # Partition reconciled key.
        reconciled_ba = bitarray_from_numpy(reconciled_arr)
        key_i0_ba, key_i1_ba = self._sifter.extract_partition_keys(
            sifted_bits=reconciled_ba,
            i0_indices=i0_indices,
            i1_indices=i1_indices,
            matching_indices=matching_indices,
        )

        key_i0_np = bitarray_to_numpy(key_i0_ba)
        key_i1_np = bitarray_to_numpy(key_i1_ba)

        # Seeds for Toeplitz; must be shared with Bob.
        # Toeplitz needs (n + m - 1) random bits.
        seed_0_bytes = ToeplitzHasher.generate_seed(
            num_bits=int(len(key_i0_np) + detailed.final_length - 1)
        )
        seed_1_bytes = ToeplitzHasher.generate_seed(
            num_bits=int(len(key_i1_np) + detailed.final_length - 1)
        )

        formatter = OTOutputFormatter(
            key_length=int(detailed.final_length),
            seed_0=seed_0_bytes,
            seed_1=seed_1_bytes,
        )

        alice_out = formatter.compute_alice_keys(key_i0=key_i0_np, key_i1=key_i1_np)

        alice_key = AliceObliviousKey(
            s0=bitarray_from_numpy(alice_out.key_0),
            s1=bitarray_from_numpy(alice_out.key_1),
            key_length=int(detailed.final_length),
            security_parameter=1e-10,
            entropy_consumed=float(detailed.entropy_consumed),
        )

        entropy_rate = float(detailed.final_length) / float(len(reconciled_arr))

        return (
            alice_key,
            int(detailed.final_length),
            float(detailed.entropy_consumed),
            float(entropy_rate),
            (seed_0_bytes, seed_1_bytes),
        )

"""Bob role program for Caligo Phase E."""

from __future__ import annotations

from typing import Any, Dict, Generator, Tuple

import numpy as np

from caligo.amplification import OTOutputFormatter
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
from caligo.types.exceptions import SecurityError, SynchronizationError
from caligo.sifting.commitment import SHA256Commitment
from caligo.types.keys import BobObliviousKey
from caligo.utils.bitarray_utils import bitarray_from_numpy, bitarray_to_numpy
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class BobProgram(CaligoProgram):
    """Bob's (receiver) program."""

    PEER = "Alice"
    ROLE = "bob"

    def __init__(self, params: ProtocolParameters, choice_bit: int = 0) -> None:
        super().__init__(params=params)
        if choice_bit not in (0, 1):
            raise ValueError("choice_bit must be 0 or 1")
        self._choice_bit = int(choice_bit)
        self._commitment = SHA256Commitment()

    @property
    def choice_bit(self) -> int:
        return self._choice_bit

    def _run_protocol(self, context) -> Generator[Any, None, Dict[str, Any]]:
        assert self._ordered_socket is not None

        bob_outcomes, bob_bases, nonce, commitment = yield from self._phase1_quantum_and_commit(
            context
        )

        # Send commitment (ordered).
        yield from self._ordered_socket.send(
            MessageType.DETECTION_COMMITMENT,
            {"commitment": commitment.hex()},
        )

        # Receive Alice bases.
        bases_msg = yield from self._ordered_socket.recv(MessageType.BASIS_REVEAL)
        _alice_bases = np.frombuffer(
            bytes.fromhex(str(bases_msg["bases"])), dtype=np.uint8
        )

        # Open commitment by sending nonce and data.
        yield from self._ordered_socket.send(
            MessageType.COMMITMENT_OPENING,
            {
                "nonce": nonce.hex(),
                "outcomes": bob_outcomes.astype(np.uint8).tobytes().hex(),
                "bases": bob_bases.astype(np.uint8).tobytes().hex(),
            },
        )

        # Optional Phase II test exchange.
        # For blind reconciliation, Alice will skip INDEX_LISTS/TEST_OUTCOMES.
        test_indices: np.ndarray
        if self.params.reconciliation.requires_qber_estimation:
            idx_msg = yield from self._ordered_socket.recv(MessageType.INDEX_LISTS)
            test_indices = np.array(idx_msg["test_indices"], dtype=np.int64)
            test_bits = bob_outcomes[test_indices].astype(np.uint8)

            yield from self._ordered_socket.send(
                MessageType.TEST_OUTCOMES,
                {"test_bits": test_bits.tobytes().hex()},
            )

        # Phase III: Reconciliation via Strategy Pattern
        # Per Implementation Report v2 §10.2.6:
        # - Delegates to ReconciliationStrategy (Baseline or Blind)
        # - Role class kept thin (< 10 lines per protocol)

        # Determine matching indices from bases only.
        matching_mask = _alice_bases == bob_bases
        matching_indices = np.where(matching_mask)[0].astype(np.int64)

        # Partition by basis value at matching positions.
        matching_bases = _alice_bases[matching_indices]
        i0_mask = matching_bases == 0
        i1_mask = matching_bases == 1
        i0_all = matching_indices[i0_mask]
        i1_all = matching_indices[i1_mask]

        # For baseline, Alice removes test indices; mimic that removal locally.
        if self.params.reconciliation.requires_qber_estimation:
            test_set = set(test_indices.tolist())
            key_indices = np.array([i for i in matching_indices.tolist() if i not in test_set], dtype=np.int64)
        else:
            key_indices = matching_indices

        # Build key bitarray from Bob outcomes at key indices.
        bob_key_np = bob_outcomes[key_indices].astype(np.uint8)

        # Partitions (I0/I1) restricted to key indices.
        i0_indices = np.intersect1d(i0_all, key_indices)
        i1_indices = np.intersect1d(i1_all, key_indices)

        # Execute Phase III reconciliation via strategy delegation
        reconciled_np, verified_positions = yield from self._phase3_reconcile(bob_key_np)

        # Filter partition indices to only include verified positions.
        # The reconciled_np corresponds to verified_positions in the original key.
        verified_set = set(verified_positions)
        verified_key_indices = np.array([i for i in key_indices.tolist() if i in verified_set], dtype=np.int64)
        verified_i0 = np.array([i for i in i0_indices.tolist() if i in verified_set], dtype=np.int64)
        verified_i1 = np.array([i for i in i1_indices.tolist() if i in verified_set], dtype=np.int64)

        # Receive Toeplitz seeds.
        seed_msg = yield from self._ordered_socket.recv(MessageType.TOEPLITZ_SEED)
        key_length = int(seed_msg["key_length"])
        seed_0 = bytes.fromhex(str(seed_msg["seed_0"]))
        seed_1 = bytes.fromhex(str(seed_msg["seed_1"]))

        # Partition reconciled key using the same sifting mapping.
        from caligo.sifting.sifter import Sifter

        sifter = Sifter()
        reconciled_ba = bitarray_from_numpy(reconciled_np)
        key_i0_ba, key_i1_ba = sifter.extract_partition_keys(
            sifted_bits=reconciled_ba,
            i0_indices=verified_i0,
            i1_indices=verified_i1,
            matching_indices=verified_key_indices,
        )

        formatter = OTOutputFormatter(
            key_length=key_length,
            seed_0=seed_0,
            seed_1=seed_1,
        )

        bob_out = formatter.compute_bob_key(
            bob_key_i0=bitarray_to_numpy(key_i0_ba),
            bob_key_i1=bitarray_to_numpy(key_i1_ba),
            choice_bit=self._choice_bit,
        )

        bob_key = BobObliviousKey(
            sc=bitarray_from_numpy(bob_out.key_c),
            choice_bit=int(bob_out.choice_bit),
            key_length=int(bob_out.key_length),
            security_parameter=1e-10,
        )

        return {
            "role": self.ROLE,
            "aborted": False,
            "bob_key": bob_key,
            "choice_bit": int(self._choice_bit),
        }

    def _phase3_reconcile(
        self, bob_key_np: np.ndarray
    ) -> Generator[Any, None, Tuple[np.ndarray, list]]:
        """
        Execute reconciliation (Phase III) via Strategy Pattern.

        Per Implementation Report v2 §10.2.6:
        - Delegates to ReconciliationStrategy (Baseline or Blind)
        - Role class kept thin (< 10 lines per protocol)
        - YAML-driven runtime switching

        Parameters
        ----------
        bob_key_np : np.ndarray
            Bob's sifted key bits.

        Yields
        ------
        Messages to/from ordered socket.

        Returns
        -------
        Tuple[np.ndarray, list]
            Reconciled key bits and list of verified positions.
        """
        assert self._ordered_socket is not None

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
        # Bob doesn't know QBER, use conservative default for safety cap
        safety_cap = compute_safety_cap(len(bob_key_np), 0.05)
        leakage_tracker = LeakageTracker(safety_cap=safety_cap, abort_on_exceed=False)

        # 3. Create strategy via factory
        strategy = create_strategy(config, mother_code, codec, leakage_tracker)

        # 4. Build ReconciliationContext (Bob doesn't have QBER initially)
        ctx = ReconciliationContext(
            session_id=id(self),
            frame_size=frame_size,
            mother_rate=0.5,
            max_iterations=config.max_iterations,
            hash_bits=64,
            f_crit=1.16,
            qber_measured=None,  # Bob doesn't know QBER
            qber_heuristic=None,
            modulation_delta=0.44,
        )

        # 5. Partition key into blocks
        from caligo.reconciliation.orchestrator import partition_key
        bob_blocks = partition_key(bob_key_np, frame_size)

        # 6. Reconcile each block via strategy generator
        reconciled_parts: list[np.ndarray] = []
        verified_positions: list[int] = []

        for block_id, bob_block in enumerate(bob_blocks):
            # Drive the strategy generator via network socket
            result = yield from self._drive_bob_strategy(
                strategy, bob_block, ctx, block_id
            )

            if result.verified:
                reconciled_parts.append(result.corrected_payload)
                start = block_id * frame_size
                verified_positions.extend(range(start, start + len(result.corrected_payload)))
            else:
                logger.warning(f"Block {block_id} failed verification - excluding from final key")

        if len(reconciled_parts) == 0:
            raise SecurityError("All reconciliation blocks failed verification")

        reconciled_np = np.concatenate(reconciled_parts) if reconciled_parts else np.array([], dtype=np.uint8)
        return reconciled_np, verified_positions

    def _drive_bob_strategy(
        self,
        strategy,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Any, None, Any]:
        """
        Drive Bob's strategy generator via network socket.

        Bridges the strategy's generator interface with the ordered socket
        for message exchange with Alice.

        Parameters
        ----------
        strategy : ReconciliationStrategy
            Active strategy instance (Baseline or Blind).
        payload : np.ndarray
            Bob's payload bits for this block.
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
        gen = strategy.bob_reconcile_block(payload, ctx, block_id)
        
        # First call yields an empty dict, expecting incoming message
        try:
            _ = next(gen)  # Initial yield
        except StopIteration as e:
            return e.value

        # Receive first message from Alice
        msg = yield from self._ordered_socket.recv(MessageType.SYNDROME)
        
        if int(msg.get("block_id", -1)) != int(block_id):
            raise SecurityError("Reconciliation block_id mismatch")
        
        # Loop: send message to generator, get response, send to Alice
        while True:
            try:
                response = gen.send(msg)
            except StopIteration as e:
                return e.value
            
            # Send response to Alice
            yield from self._ordered_socket.send(MessageType.SYNDROME_RESPONSE, {
                "block_id": int(block_id),
                **response,
            })
            
            # Check if this was a terminal response (verified or final iteration)
            if response.get("verified", False) or response.get("kind") != "blind_reveal":
                # Try to get final result
                try:
                    next(gen)
                except StopIteration as e:
                    return e.value
            
            # For blind protocol, may receive another message
            msg = yield from self._ordered_socket.recv(MessageType.SYNDROME)

    def _phase1_quantum_and_commit(
        self, context
    ) -> Generator[Any, None, Tuple[np.ndarray, np.ndarray, bytes, bytes]]:
        """Receive and measure EPR pairs (Bob side), then commit."""

        if self.params.precomputed_epr is not None:
            n = int(self.params.num_pairs)
            outcomes = np.asarray(self.params.precomputed_epr.bob_outcomes, dtype=np.uint8)
            bases = np.asarray(self.params.precomputed_epr.bob_bases, dtype=np.uint8)
            if len(outcomes) != n or len(bases) != n:
                raise ValueError(
                    "precomputed_epr length mismatch: "
                    f"expected n={n}, got outcomes={len(outcomes)} bases={len(bases)}"
                )

            data_bytes = np.concatenate([outcomes, bases]).astype(np.uint8).tobytes()
            commit_res = self._commitment.commit(data_bytes)

            return (
                outcomes.astype(np.uint8),
                bases.astype(np.uint8),
                commit_res.nonce,
                commit_res.commitment,
            )

        from caligo.quantum import BasisSelector, MeasurementExecutor

        epr_socket = context.epr_sockets[self.PEER]
        basis_selector = BasisSelector()
        meas = MeasurementExecutor()

        n = int(self.params.num_pairs)
        bases = basis_selector.select_batch(n)
        outcomes = np.zeros(n, dtype=np.uint8)

        # Use the same pattern as SquidASM's QKD example:
        # receive and measure one EPR pair per round to preserve ordering.
        for round_id in range(n):
            q = epr_socket.recv_keep(1)[0]
            outcomes[round_id] = yield from meas.measure_qubit(
                qubit=q,
                basis=int(bases[round_id]),
                round_id=round_id,
                context=context,
            )

        data_bytes = np.concatenate([outcomes, bases]).astype(np.uint8).tobytes()
        commit_res = self._commitment.commit(data_bytes)

        return (
            outcomes.astype(np.uint8),
            bases.astype(np.uint8),
            commit_res.nonce,
            commit_res.commitment,
        )

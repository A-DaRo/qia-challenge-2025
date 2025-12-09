"""Alice baseline E-HOK role using the extensible orchestration layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Dict, Any, Tuple

import numpy as np
from pydynaa import EventExpression

from ehok.core.config import ProtocolConfig
from ehok.core.constants import TARGET_EPSILON_SEC
from ehok.core.data_structures import LDPCBlockResult, ObliviousKey
from ehok.core.exceptions import CommitmentVerificationError, MatrixSynchronizationError, ReconciliationFailedError
from ehok.protocols.base import EHOKRole
from ehok.implementations import factories
from ehok.quantum.runner import QuantumPhaseResult
from ehok.utils.logging import get_logger

logger = get_logger("protocols.alice")


class AliceBaselineEHOK(EHOKRole):
	"""Baseline Alice role with pluggable strategies."""

	PEER_NAME = "bob"
	ROLE = "alice"

	def __init__(self, config: ProtocolConfig | None = None, **kwargs):
		super().__init__(config, **kwargs)

	# ------------------------------------------------------------------
	# Main template implementation
	# ------------------------------------------------------------------
	def _execute_remaining_phases(
		self, quantum_result: QuantumPhaseResult
	) -> Generator[EventExpression, None, Dict[str, Any]]:
		outcomes_alice = quantum_result.outcomes
		bases_alice = quantum_result.bases

		commitment = yield from self._phase2_receive_commitment()
		(
			I_0,
			I_1,
			test_set,
			key_set,
			qber,
			outcomes_bob,
			bases_bob,
		) = yield from self._phase3_sifting_sampling(
			outcomes_alice, bases_alice, commitment
		)

		alice_key = outcomes_alice[key_set]
		self._build_reconciliator()
		block_result, corrected_key = yield from self._phase4_reconciliation(alice_key)

		self._build_privacy_amplifier()
		oblivious_key = yield from self._phase5_privacy_amplification(
			corrected_key, block_result
		)

		return self._result_success(
			oblivious_key=oblivious_key,
			qber=qber,
			raw_count=len(outcomes_alice),
			sifted_count=len(I_0),
			test_count=len(test_set),
			final_count=oblivious_key.final_length,
			role=self.ROLE,
			measurement_records=quantum_result.measurement_records,
		)

	# ------------------------------------------------------------------
	# Phase helpers
	# ------------------------------------------------------------------
	def _phase2_receive_commitment(self) -> Generator[EventExpression, None, bytes]:
		logger.info("=== PHASE 2: Commitment ===")
		commitment_msg = yield from self.context.csockets[self.PEER_NAME].recv()
		commitment = bytes.fromhex(commitment_msg)
		logger.info("Received commitment from Bob: %s...", commitment.hex()[:16])
		return commitment

	def _phase3_sifting_sampling(
		self,
		outcomes_alice: np.ndarray,
		bases_alice: np.ndarray,
		commitment: bytes,
	) -> Generator[EventExpression, None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]]:
		logger.info("=== PHASE 3: Sifting & Sampling ===")

		bases_msg = bases_alice.tobytes().hex()
		self.context.csockets[self.PEER_NAME].send(bases_msg)

		bob_data_msg = yield from self.context.csockets[self.PEER_NAME].recv()
		bob_data = np.frombuffer(bytes.fromhex(bob_data_msg), dtype=np.uint8)

		n = len(outcomes_alice)
		outcomes_bob = bob_data[:n]
		bases_bob = bob_data[n : 2 * n]
		decommitment_salt = bob_data[2 * n :].tobytes()

		I_0, I_1 = self.sifting_manager.identify_matching_bases(bases_alice, bases_bob)
		test_set, key_set = self.sampling_strategy.select_sets(
			I_0,
			fraction=self.config.security.test_set_fraction,
			min_size=self.config.security.min_test_set_size,
			seed=self.config.sampling_seed,
		)

		full_bob_data = np.concatenate([outcomes_bob, bases_bob])
		if not self.commitment_scheme.verify(commitment, full_bob_data, decommitment_salt):
			raise CommitmentVerificationError("Bob's commitment verification failed")

		qber = self.sifting_manager.estimate_qber(outcomes_alice, outcomes_bob, test_set)
		self.sifting_manager.check_qber_abort(
			qber, threshold=self.config.security.qber_threshold
		)

		return I_0, I_1, test_set, key_set, qber, outcomes_bob, bases_bob

	def _phase4_reconciliation(
		self, alice_key: np.ndarray
	) -> Generator[EventExpression, None, Tuple[LDPCBlockResult, np.ndarray]]:
		logger.info("=== PHASE 4: Information Reconciliation ===")

		# Matrix pool synchronization
		remote_checksum = yield from self.context.csockets[self.PEER_NAME].recv()
		local_checksum = self.reconciliator.matrix_manager.checksum  # type: ignore[attr-defined]
		if remote_checksum != local_checksum:
			raise MatrixSynchronizationError(local_checksum, remote_checksum)
		self.context.csockets[self.PEER_NAME].send(local_checksum)

		# Receive Bob's reconciliation parameters
		bob_msg = yield from self.context.csockets[self.PEER_NAME].recv()
		payload = json.loads(bob_msg)
		rate = float(payload["rate"])
		n_short = int(payload["n_short"])
		seed = int(payload["seed"])
		payload_len = int(payload["payload_len"])
		syndrome = np.frombuffer(bytes.fromhex(payload["syndrome"]), dtype=np.uint8)
		bob_hash = bytes.fromhex(payload["hash"])

		alice_block = alice_key[:payload_len]
		corrected_block, converged, error_count = self.reconciliator.reconcile_block(
			alice_block, syndrome, rate, n_short, seed
		)

		alice_hash = self.reconciliator.hash_verifier.compute_hash(corrected_block, seed)  # type: ignore[attr-defined]
		verified = alice_hash == bob_hash and converged
		result = LDPCBlockResult(
			verified=verified,
			error_count=error_count,
			block_length=len(alice_block),
			syndrome_length=len(syndrome),
			hash_bits=self.reconciliator.hash_verifier.hash_length_bits,  # type: ignore[attr-defined]
		)

		ack = {"verified": verified, "error_count": error_count}
		self.context.csockets[self.PEER_NAME].send(json.dumps(ack))
		if not verified:
			raise ReconciliationFailedError("LDPC reconciliation failed verification")
		logger.info("Reconciliation successful (errors corrected=%d)", error_count)
		return result, corrected_block

	def _phase5_privacy_amplification(
		self, corrected_key: np.ndarray, block_result: LDPCBlockResult
	) -> Generator[EventExpression, None, ObliviousKey]:
		logger.info("=== PHASE 5: Privacy Amplification ===")
		leakage = self.reconciliator.estimate_leakage_block(  # type: ignore[attr-defined]
			block_result.syndrome_length, block_result.hash_bits
		)
		if block_result.block_length == 0:
			qber = 0.0 if block_result.verified else 0.5
		else:
			qber = block_result.error_count / block_result.block_length if block_result.verified else 0.5

		final_length = self.privacy_amplifier.compute_final_length(
			len(corrected_key), qber, leakage, self.config.privacy_amplification.target_epsilon
		)

		seed = self.privacy_amplifier.generate_hash_seed(len(corrected_key), final_length)
		self.context.csockets[self.PEER_NAME].send(seed.tobytes().hex())

		final_key = self.privacy_amplifier.compress(corrected_key, seed)
		knowledge_mask = np.zeros_like(final_key)
		oblivious_key = ObliviousKey(
			key_value=final_key,
			knowledge_mask=knowledge_mask,
			security_param=TARGET_EPSILON_SEC,
			qber=qber,
			final_length=final_length,
		)

		yield from self.context.connection.flush()
		return oblivious_key

	# ------------------------------------------------------------------
	# Utilities
	# ------------------------------------------------------------------
	def _build_reconciliator(self) -> None:
		if self.reconciliator is not None:
			return
		self.reconciliator = factories.build_reconciliator(self.config)


# Backwards compatibility aliases
AliceEHOKProgram = AliceBaselineEHOK
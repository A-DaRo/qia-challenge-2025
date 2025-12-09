"""Alice baseline E-HOK role using the extensible orchestration layer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Generator, Dict, Any, Tuple

import numpy as np
import scipy.sparse as sp
from pydynaa import EventExpression

from ehok.core.config import ProtocolConfig
from ehok.core.constants import TARGET_EPSILON_SEC
from ehok.core.data_structures import ObliviousKey
from ehok.core.exceptions import CommitmentVerificationError, ReconciliationFailedError
from ehok.protocols.base import EHOKRole
from ehok.quantum.runner import QuantumPhaseResult
from ehok.utils.logging import get_logger

logger = get_logger("protocols.alice")


class AliceBaselineEHOK(EHOKRole):
	"""Baseline Alice role with pluggable strategies."""

	PEER_NAME = "bob"
	ROLE = "alice"

	def __init__(self, config: ProtocolConfig | None = None):
		super().__init__(config)

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
		parity_check = self._load_ldpc_matrix(len(alice_key))
		self._build_reconciliator(parity_check)
		yield from self._phase4_reconciliation(alice_key)

		self._build_privacy_amplifier()
		oblivious_key = yield from self._phase5_privacy_amplification(
			alice_key, qber
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
	) -> Generator[EventExpression, None, None]:
		logger.info("=== PHASE 4: Information Reconciliation ===")

		if len(alice_key) > self.reconciliator.n:
			logger.warning(
				"Truncating key from %d to %d to match LDPC matrix",
				len(alice_key),
				self.reconciliator.n,
			)
			alice_key = alice_key[: self.reconciliator.n]

		syndrome = self.reconciliator.compute_syndrome(alice_key)
		self.context.csockets[self.PEER_NAME].send(syndrome.tobytes().hex())

		bob_hash_msg = yield from self.context.csockets[self.PEER_NAME].recv()
		bob_hash = bob_hash_msg
		alice_hash = hashlib.sha256(alice_key.tobytes()).hexdigest()

		if bob_hash != alice_hash:
			raise ReconciliationFailedError(
				"Key hash mismatch after reconciliation. "
				f"Alice: {alice_hash[:16]}..., Bob: {bob_hash[:16]}..."
			)
		logger.info("Reconciliation successful")

	def _phase5_privacy_amplification(
		self, alice_key: np.ndarray, qber: float
	) -> Generator[EventExpression, None, ObliviousKey]:
		logger.info("=== PHASE 5: Privacy Amplification ===")
		syndrome_length = self.reconciliator.m
		leakage = self.reconciliator.estimate_leakage(syndrome_length, qber)

		final_length = self.privacy_amplifier.compute_final_length(
			len(alice_key), qber, leakage, self.config.privacy_amplification.target_epsilon
		)

		seed = self.privacy_amplifier.generate_hash_seed(len(alice_key), final_length)
		self.context.csockets[self.PEER_NAME].send(seed.tobytes().hex())

		final_key = self.privacy_amplifier.compress(alice_key, seed)
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
	def _load_ldpc_matrix(self, n: int) -> sp.spmatrix:
		ldpc_dir = Path(__file__).parent.parent / "configs" / "ldpc_matrices"
		available_sizes = [1000, 2000, 4500, 5000]
		closest = min(available_sizes, key=lambda x: abs(x - n))
		matrix_file = ldpc_dir / f"ldpc_{closest}_rate05.npz"
		if not matrix_file.exists():
			raise FileNotFoundError(
				f"LDPC matrix not found: {matrix_file}. "
				"Run ehok/configs/generate_ldpc.py to generate matrices."
			)

		H = sp.load_npz(matrix_file)
		if n < H.shape[1]:
			H = H[:, :n]
			m_new = int(n * self.config.reconciliation.code_rate)
			H = H[:m_new, :]
		return H


# Backwards compatibility aliases
AliceEHOKProgram = AliceBaselineEHOK
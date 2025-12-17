"""Alice baseline E-HOK role using the extensible orchestration layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Dict, Any, Tuple, Union, Optional

import numpy as np
import netsquid as ns
from pydynaa import EventExpression

from ehok.core.config import ProtocolConfig
from ehok.core.constants import TARGET_EPSILON_SEC
from ehok.core.data_structures import LDPCBlockResult, ObliviousKey
from ehok.core.oblivious_formatter import (
    AliceObliviousKey,
    ProtocolMetrics,
    ObliviousKeyFormatter,
)
from ehok.core.timing import TimingEnforcer
from ehok.core.exceptions import CommitmentVerificationError, MatrixSynchronizationError, ReconciliationFailedError
from ehok.protocols.base import EHOKRole
from ehok.protocols.ordered_messaging import OrderedProtocolSocket, MessageType
from ehok.protocols.leakage_manager import LeakageSafetyManager, BlockReconciliationReport, ABORT_CODE_LEAKAGE_CAP_EXCEEDED
from ehok.implementations import factories
from ehok.implementations.privacy_amplification.finite_key import (
    FiniteKeyParams,
    compute_final_length_finite_key,
)
from ehok.implementations.privacy_amplification.nsm_privacy_amplifier import (
    NSMPrivacyAmplificationParams,
    compute_nsm_key_length,
)
from ehok.analysis.nsm_bounds import FeasibilityResult
from ehok.quantum.runner import QuantumPhaseResult
from ehok.utils.logging import get_logger

logger = get_logger("protocols.alice")


class AliceBaselineEHOK(EHOKRole):
	"""
	Baseline Alice role with pluggable strategies.

	Supports dependency injection for:
	- OrderedProtocolSocket: Enforces commit-then-reveal ordering
	- TimingEnforcer: Enforces NSM timing barrier Δt before basis reveal
	- LeakageSafetyManager: Tracks reconciliation leakage budget
	"""

	PEER_NAME = "bob"
	ROLE = "alice"

	def __init__(
		self,
		config: ProtocolConfig | None = None,
		ordered_socket: Optional[OrderedProtocolSocket] = None,
		timing_enforcer: Optional[TimingEnforcer] = None,
		leakage_manager: Optional[LeakageSafetyManager] = None,
		**kwargs
	):
		super().__init__(
			config=config,
			ordered_socket=ordered_socket,
			timing_enforcer=timing_enforcer,
			leakage_manager=leakage_manager,
			**kwargs
		)

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

		# If no secure key could be extracted, abort gracefully
		if oblivious_key.final_length == 0:
			abort_reason = "PRIVACY_AMPLIFICATION_NO_SECURE_KEY"
			return self._result_abort(
				abort_reason=abort_reason,
				qber=qber,
				raw_count=len(outcomes_alice),
				sifted_count=len(I_0),
				test_count=len(test_set),
				role=self.ROLE,
				measurement_records=quantum_result.measurement_records,
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

		# SECURITY: Start NSM timing barrier upon receiving Bob's commitment
		# Alice must wait Δt before revealing her bases (commit-then-reveal)
		if self._timing_enforcer is not None:
			sim_time = int(ns.sim_time())
			self._timing_enforcer.mark_commit_received(sim_time_ns=sim_time)
			logger.debug(
				"Timing barrier started at t=%d ns after commitment received",
				sim_time
			)

		return commitment

	def _phase3_sifting_sampling(
		self,
		outcomes_alice: np.ndarray,
		bases_alice: np.ndarray,
		commitment: bytes,
	) -> Generator[EventExpression, None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]]:
		logger.info("=== PHASE 3: Sifting & Sampling ===")

		# SECURITY: Enforce NSM timing barrier before basis reveal
		# Alice must wait Δt after Bob's commitment acknowledgment
		if self._timing_enforcer is not None:
			sim_time = int(ns.sim_time())
			self._timing_enforcer.mark_basis_reveal_attempt(sim_time_ns=sim_time)
			logger.debug(
				"Timing barrier verified at t=%d ns before basis reveal",
				sim_time
			)

		# Send basis information (commit-then-reveal: bases revealed AFTER Δt)
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
		# Store measured QBER for reconciliator initialization
		self.measured_qber = qber

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
		# Accept block if the verification hash matches even when the BP decoder
		# did not report 'converged'. The hash ensures correctness even if the
		# decoder used an alternative decision that didn't set the flag.
		verified = alice_hash == bob_hash
		result = LDPCBlockResult(
			verified=verified,
			error_count=error_count,
			block_length=len(alice_block),
			syndrome_length=len(syndrome),
			hash_bits=self.reconciliator.hash_verifier.hash_length_bits,  # type: ignore[attr-defined]
		)

		# Wire leakage tracking for security accounting
		if self._leakage_manager is not None:
			report = BlockReconciliationReport(
				block_index=0,
				syndrome_bits=len(syndrome) * 8,
				hash_bits=self.reconciliator.hash_verifier.hash_length_bits,  # type: ignore[attr-defined]
				decode_converged=converged,
				hash_verified=verified,
				iterations=0,  # BP iterations not exposed in current interface
			)
			self._leakage_manager.account_block(report)
			if self._leakage_manager.is_cap_exceeded:
				logger.error(
					"Leakage cap exceeded: total=%d bits, cap=%d bits",
					self._leakage_manager.total_leakage_bits,
					self._leakage_manager.max_leakage_bits
				)
				raise RuntimeError(ABORT_CODE_LEAKAGE_CAP_EXCEEDED)

		ack = {"verified": verified, "error_count": error_count}
		self.context.csockets[self.PEER_NAME].send(json.dumps(ack))
		if not verified:
			raise ReconciliationFailedError("LDPC reconciliation failed verification")
		logger.info("Reconciliation successful (errors corrected=%d)", error_count)
		return result, corrected_block

	def _phase5_privacy_amplification(
		self, corrected_key: np.ndarray, block_result: LDPCBlockResult
	) -> Generator[EventExpression, None, ObliviousKey]:
		"""
		Execute Phase V: Privacy Amplification using NSM bounds.

		This method computes the secure key length using the NSM Max Bound:
		    ℓ ≤ n · h_min(r) - |Σ| - 2·log₂(1/ε_sec)

		Where h_min(r) = max { Γ[1 - log₂(1 + 3r²)], 1 - r }

		Returns
		-------
		ObliviousKey
			Contains final_length, key_value, and metadata.
			The knowledge_mask is set to zeros for Alice (she knows her entire key).
		"""
		logger.info("=== PHASE 5: Privacy Amplification ===")

		# Compute leakage from reconciliation (wiretap cost |Σ|)
		leakage = self.reconciliator.estimate_leakage_block(  # type: ignore[attr-defined]
			block_result.syndrome_length, block_result.hash_bits
		)

		# Compute QBER from reconciliation results
		if block_result.block_length == 0:
			qber = 0.0 if block_result.verified else 0.5
		else:
			qber = block_result.error_count / block_result.block_length if block_result.verified else 0.5

		pa_config = self.config.privacy_amplification

		# Get storage noise parameter from NSM config
		storage_noise_r = getattr(self.config, 'nsm', None)
		if storage_noise_r is not None:
			storage_noise_r = self.config.nsm.storage_noise_r
		else:
			# Default to Erven et al. value if nsm config not present
			storage_noise_r = 0.75

		# Handle empty reconciled key case
		if len(corrected_key) == 0:
			final_length = 0
			entropy_bound_used = "none"
		else:
			# Use NSM formula for key length calculation
			nsm_params = NSMPrivacyAmplificationParams(
				reconciled_key_length=len(corrected_key),
				storage_noise_r=storage_noise_r,
				syndrome_leakage_bits=block_result.syndrome_length,
				hash_leakage_bits=block_result.hash_bits,
				epsilon_sec=pa_config.target_epsilon_sec,
				adjusted_qber=qber,
			)
			nsm_result = compute_nsm_key_length(nsm_params)
			final_length = nsm_result.secure_key_length
			entropy_bound_used = nsm_result.entropy_bound_used

			logger.debug(
				"NSM PA: n=%d, r=%.3f, h_min=%.4f, leakage=%d, ε=%.2e → ℓ=%d [%s]",
				len(corrected_key),
				storage_noise_r,
				nsm_result.min_entropy_rate,
				nsm_params.total_leakage,
				pa_config.target_epsilon_sec,
				final_length,
				entropy_bound_used,
			)

			# Log death valley warning if needed
			if nsm_result.feasibility != FeasibilityResult.FEASIBLE:
				logger.warning(
					"Death Valley: feasibility=%s, extractable=%.1f, consumed=%.1f",
					nsm_result.feasibility.name,
					nsm_result.extractable_entropy,
					nsm_result.entropy_consumed,
				)

		# Generate Toeplitz seed and compress
		if final_length <= 0:
			# No secure key: send empty seed
			seed = np.array([], dtype=np.uint8)
			self.context.csockets[self.PEER_NAME].send(seed.tobytes().hex())
			final_key = np.zeros(0, dtype=np.uint8)
		else:
			seed = self.privacy_amplifier.generate_hash_seed(len(corrected_key), final_length)
			self.context.csockets[self.PEER_NAME].send(seed.tobytes().hex())
			final_key = self.privacy_amplifier.compress(corrected_key, seed)

		# Create ObliviousKey output
		# Alice knows all bits (knowledge_mask = 0)
		knowledge_mask = np.zeros_like(final_key)
		oblivious_key = ObliviousKey(
			key_value=final_key,
			knowledge_mask=knowledge_mask,
			security_param=pa_config.target_epsilon_sec,
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
		# Update reconciliator with measured QBER from Phase 3 if available
		if hasattr(self, 'measured_qber'):
			self.reconciliator.current_qber_est = self.measured_qber


# Protocol aliases
AliceEHOKProgram = AliceBaselineEHOK
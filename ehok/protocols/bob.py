"""
Bob's protocol for E-HOK Baseline.

Complete implementation of all 5 phases:
1. Quantum Generation
2. Commitment (send)
3. Sifting & Sampling
4. Information Reconciliation
5. Privacy Amplification
"""

import hashlib
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Generator, Dict, Any

from pydynaa import EventExpression
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.sim.stack.common import LogManager
import netsquid as ns

from ..core.constants import TOTAL_EPR_PAIRS, BATCH_SIZE, TARGET_EPSILON_SEC
from ..core.data_structures import ObliviousKey
from ..core.sifting import SiftingManager
from ..core.exceptions import ReconciliationFailedError
from ..quantum.batching_manager import BatchingManager, EPRGenerator
from ..quantum.measurement import MeasurementBuffer
from ..implementations.commitment.sha256_commitment import SHA256Commitment
from ..implementations.reconciliation.ldpc_reconciliator import LDPCReconciliator
from ..implementations.privacy_amplification.toeplitz_amplifier import ToeplitzAmplifier

logger = LogManager.get_stack_logger("ehok.protocols.bob")



class BobEHOKProgram(Program):
    """
    Bob's program for complete E-HOK baseline protocol.
    
    Bob plays the role of the receiver who:
    - Receives and measures EPR pairs
    - Commits to his outcomes BEFORE learning Alice's bases
    - Reveals his data for sifting verification
    - Decodes syndrome for reconciliation
    - Receives privacy amplification seed
    - Outputs ObliviousKey with partial knowledge
    """
    PEER_NAME = "alice"

    def __init__(self, total_pairs: int = TOTAL_EPR_PAIRS):
        """
        Initialize Bob's program.
        
        Parameters
        ----------
        total_pairs : int
            Total number of EPR pairs to generate.
        """
        self.total_pairs = total_pairs
        self.measurement_buffer = MeasurementBuffer()

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="bob_ehok",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=5,
        )

    def run(self, context: ProgramContext) -> Generator[EventExpression, None, Dict[str, Any]]:
        """
        Execute Bob's complete E-HOK protocol.
        
        Yields
        ------
        EventExpression
            NetSquid events for simulation scheduling.
        
        Returns
        -------
        dict
            Protocol results including ObliviousKey and statistics.
        """
        logger.info("=== Bob E-HOK Protocol Started ===")
        self.context = context
        
        # Initialize components
        self.epr_socket = context.epr_sockets[self.PEER_NAME]
        self.csocket = context.csockets[self.PEER_NAME]
        
        self.batching_manager = BatchingManager(self.total_pairs, BATCH_SIZE)
        self.epr_generator = EPRGenerator(self.epr_socket, "bob")
        
        self.commitment_scheme = SHA256Commitment()
        self.sifting = SiftingManager()

        # ===== PHASE 1: Quantum Generation =====
        yield from self._phase1_quantum_generation()
        
        # Extract arrays for classical processing
        outcomes_bob = self.measurement_buffer.get_outcomes()
        bases_bob = self.measurement_buffer.get_bases()
        
        # ===== PHASE 2: Commitment (Send to Alice) =====
        decommitment_salt = yield from self._phase2_send_commitment(
            outcomes_bob, bases_bob
        )
        
        # ===== PHASE 3: Sifting & Sampling =====
        I_0, I_1, key_set = yield from self._phase3_sifting_sampling(
            outcomes_bob, bases_bob, decommitment_salt
        )
        
        # ===== PHASE 4: Information Reconciliation =====
        bob_key = outcomes_bob[key_set]
        qber = yield from self._phase4_reconciliation(bob_key)
        
        # ===== PHASE 5: Privacy Amplification =====
        oblivious_key = yield from self._phase5_privacy_amplification(
            bob_key, qber, I_1, len(outcomes_bob)
        )
        
        logger.info("=== Bob E-HOK Protocol Complete ===")
        logger.info(f"Final key length: {oblivious_key.final_length} bits")
        logger.info(f"QBER: {qber*100:.2f}%")
        logger.info(f"Security parameter: {oblivious_key.security_param}")
        logger.info(f"Unknown bits: {np.sum(oblivious_key.knowledge_mask)}")
        
        return {
            "success": True,
            "oblivious_key": oblivious_key,
            "qber": qber,
            "raw_count": len(outcomes_bob),
            "sifted_count": len(I_0),
            "final_count": oblivious_key.final_length,
            "role": "bob"
        }

    def _phase1_quantum_generation(self) -> Generator[EventExpression, None, None]:
        """
        Execute Phase 1: Receive and measure EPR pairs in batches.
        
        Yields
        ------
        EventExpression
            Flush events for quantum operations.
        """
        logger.info("=== PHASE 1: Quantum Generation ===")
        
        batch_sizes = self.batching_manager.compute_batch_sizes()
        
        for i, batch_size in enumerate(batch_sizes):
            logger.debug(f"Processing batch {i+1}/{len(batch_sizes)} (size={batch_size})")
            
            # 1. Receive EPR pairs (returns futures)
            sim_time = ns.sim_time()
            qubits = self.epr_generator.generate_batch_bob(batch_size, sim_time)
            
            # 2. Measure locally (returns futures)
            outcome_futures, bases = self.epr_generator.measure_batch(qubits, sim_time)
            
            # 3. Flush to execute on quantum hardware
            yield from self.context.connection.flush()
            
            # 4. Extract results
            batch_result = self.epr_generator.extract_batch_results(
                outcome_futures, bases, sim_time
            )
            batch_result.batch_index = i
            
            # 5. Store in measurement buffer
            self.measurement_buffer.add_batch(
                batch_result.outcomes,
                batch_result.bases,
                batch_result.timestamps
            )
        
        logger.info(f"Generated {len(self.measurement_buffer)} EPR pairs")
    
    def _phase2_send_commitment(
        self,
        outcomes_bob: np.ndarray,
        bases_bob: np.ndarray
    ) -> Generator[EventExpression, None, bytes]:
        """
        Execute Phase 2: Commit to measurement results.
        
        SECURITY CRITICAL: Bob must commit to his results BEFORE learning
        Alice's bases. This ensures Bob cannot adapt his claimed results.
        
        Parameters
        ----------
        outcomes_bob : np.ndarray
            Bob's measurement outcomes.
        bases_bob : np.ndarray
            Bob's measurement bases.
        
        Yields
        ------
        EventExpression
            Classical socket send event.
        
        Returns
        -------
        bytes
            Decommitment salt for later opening.
        """
        logger.info("=== PHASE 2: Commitment ===")
        
        # 1. Prepare data to commit (outcomes || bases)
        data = np.concatenate([outcomes_bob, bases_bob])
        
        # 2. Generate commitment
        commitment, decommitment_salt = self.commitment_scheme.commit(data)
        
        # 3. Send commitment to Alice
        commitment_msg = commitment.hex()
        self.csocket.send(commitment_msg)
        
        logger.info(f"Sent commitment to Alice: {commitment.hex()[:16]}...")
        
        # Yield to ensure message is sent before proceeding
        yield from self.context.connection.flush()
        
        return decommitment_salt
    
    def _phase3_sifting_sampling(
        self,
        outcomes_bob: np.ndarray,
        bases_bob: np.ndarray,
        decommitment_salt: bytes
    ) -> Generator[EventExpression, None, tuple]:
        """
        Execute Phase 3: Sifting & Sampling.
        
        Parameters
        ----------
        outcomes_bob : np.ndarray
            Bob's measurement outcomes.
        bases_bob : np.ndarray
            Bob's measurement bases.
        decommitment_salt : bytes
            Salt for commitment verification.
        
        Yields
        ------
        EventExpression
            Classical communication events.
        
        Returns
        -------
        tuple
            (I_0, I_1, key_set)
        """
        logger.info("=== PHASE 3: Sifting & Sampling ===")
        
        # 1. Receive Alice's bases
        bases_alice_msg = yield from self.csocket.recv()
        bases_alice = np.frombuffer(bytes.fromhex(bases_alice_msg), dtype=np.uint8)
        
        logger.debug(f"Received Alice's bases: {len(bases_alice)} bases")
        
        # 2. Identify matching bases (sifting)
        I_0, I_1 = self.sifting.identify_matching_bases(bases_alice, bases_bob)
        
        # 3. Select test set (must use same random selection as Alice)
        # NOTE: Both parties use SiftingManager with default seed (None)
        # This means they will get the same random selection if I_0 is the same
        test_set, key_set = self.sifting.select_test_set(I_0)
        
        # 4. Send decommitment: full data + salt
        # Alice needs full data to verify SHA256 commitment
        full_data = np.concatenate([outcomes_bob, bases_bob, 
                                    np.frombuffer(decommitment_salt, dtype=np.uint8)])
        bob_data_msg = full_data.tobytes().hex()
        self.csocket.send(bob_data_msg)
        
        logger.debug(f"Sent decommitment data to Alice")
        
        return I_0, I_1, key_set
    
    def _phase4_reconciliation(
        self,
        bob_key: np.ndarray
    ) -> Generator[EventExpression, None, float]:
        """
        Execute Phase 4: Information Reconciliation.
        
        Bob receives syndrome from Alice and decodes his key.
        
        Parameters
        ----------
        bob_key : np.ndarray
            Bob's sifted key (after removing test set).
        
        Yields
        ------
        EventExpression
            Classical communication events.
        
        Returns
        -------
        float
            Estimated QBER.
        """
        logger.info("=== PHASE 4: Information Reconciliation ===")
        
        # 1. Receive syndrome from Alice
        syndrome_msg = yield from self.csocket.recv()
        syndrome = np.frombuffer(bytes.fromhex(syndrome_msg), dtype=np.uint8)
        
        logger.info(f"Received syndrome: {len(syndrome)} bits, weight={np.sum(syndrome)}")
        
        # 2. Load LDPC matrix (same as Alice's)
        sifted_length = len(bob_key)
        H = self._load_ldpc_matrix(sifted_length)
        self.reconciliator = LDPCReconciliator(H)
        
        # 3. Handle key length mismatch
        if sifted_length > H.shape[1]:
            logger.warning(
                f"Truncating key from {sifted_length} to {H.shape[1]} to match LDPC matrix"
            )
            bob_key = bob_key[:H.shape[1]]
        
        # 4. Decode using belief propagation
        logger.debug(f"Bob key length: {len(bob_key)}, LDPC matrix shape: {H.shape}")
        logger.debug(f"Bob key hash BEFORE decoding: {hashlib.sha256(bob_key.tobytes()).hexdigest()[:16]}...")
        bob_key_corrected = self.reconciliator.reconcile(bob_key, syndrome)
        
        # 5. Estimate QBER from error correction
        errors = np.sum(bob_key != bob_key_corrected)
        qber = errors / len(bob_key) if len(bob_key) > 0 else 0.0
        logger.info(f"Corrected {errors} errors, estimated QBER: {qber*100:.2f}%")
        
        # 6. Send hash of corrected key for verification
        bob_hash = hashlib.sha256(bob_key_corrected.tobytes()).hexdigest()
        self.csocket.send(bob_hash)
        
        # 7. Update Bob's key with corrected version
        bob_key[:] = bob_key_corrected
        
        logger.info("Reconciliation complete")
        
        return qber
    
    def _phase5_privacy_amplification(
        self,
        bob_key: np.ndarray,
        qber: float,
        I_1: np.ndarray,
        total_length: int
    ) -> Generator[EventExpression, None, ObliviousKey]:
        """
        Execute Phase 5: Privacy Amplification.
        
        Parameters
        ----------
        bob_key : np.ndarray
            Reconciled sifted key.
        qber : float
            Measured QBER.
        I_1 : np.ndarray
            Indices of mismatched bases (for knowledge mask).
        total_length : int
            Total raw key length before sifting.
        
        Yields
        ------
        EventExpression
            Classical communication events.
        
        Returns
        -------
        ObliviousKey
            Final oblivious key with metadata.
        """
        logger.info("=== PHASE 5: Privacy Amplification ===")
        
        # 1. Receive Toeplitz seed from Alice
        seed_msg = yield from self.csocket.recv()
        seed = np.frombuffer(bytes.fromhex(seed_msg), dtype=np.uint8)
        
        logger.debug(f"Received Toeplitz seed: {len(seed)} bits")
        
        # 2. Compress key using same seed
        amplifier = ToeplitzAmplifier()
        final_key = amplifier.compress(bob_key, seed)
        
        final_length = len(final_key)
        logger.info(f"Compressed key: {len(bob_key)} → {final_length} bits")
        
        # 3. Construct knowledge mask
        # Bob doesn't know bits at mismatched basis positions (I_1)
        # But knowledge_mask is for the FINAL key, not the raw key
        # For simplicity in baseline: mark proportion of final key as unknown
        # This is a simplified model; in reality, the oblivious property
        # is more nuanced (related to which raw bits contributed to final bits)
        
        # Conservative approach: mark fraction corresponding to I_1 as unknown
        fraction_unknown = len(I_1) / total_length
        num_unknown = int(final_length * fraction_unknown)
        
        knowledge_mask = np.zeros(final_length, dtype=np.uint8)
        if num_unknown > 0:
            # Mark first num_unknown bits as unknown (arbitrary choice)
            knowledge_mask[:num_unknown] = 1
        
        logger.debug(f"Knowledge mask: {num_unknown}/{final_length} bits unknown")
        
        # 4. Construct ObliviousKey
        oblivious_key = ObliviousKey(
            key_value=final_key,
            knowledge_mask=knowledge_mask,
            security_param=TARGET_EPSILON_SEC,
            qber=qber,
            final_length=final_length
        )
        
        logger.info(f"Privacy amplification complete: {final_length}-bit key")
        
        return oblivious_key
    
    def _load_ldpc_matrix(self, n: int) -> sp.spmatrix:
        """
        Load appropriate LDPC matrix for key size.
        
        Parameters
        ----------
        n : int
            Sifted key length.
        
        Returns
        -------
        H : scipy.sparse matrix
            Parity check matrix.
        """
        ldpc_dir = Path(__file__).parent.parent / "configs" / "ldpc_matrices"
        
        # Find closest available size
        available_sizes = [1000, 2000, 4500, 5000]
        closest = min(available_sizes, key=lambda x: abs(x - n))
        
        matrix_file = ldpc_dir / f"ldpc_{closest}_rate05.npz"
        
        if not matrix_file.exists():
            raise FileNotFoundError(
                f"LDPC matrix not found: {matrix_file}. "
                f"Run ehok/configs/generate_ldpc.py to generate matrices."
            )
        
        H = sp.load_npz(matrix_file)
        logger.info(f"Loaded LDPC matrix: {H.shape} from {matrix_file.name}")
        
        # Adjust matrix to match key length exactly
        if n != H.shape[1]:
            if n < H.shape[1]:
                # Truncate columns
                H = H[:, :n]
                # Adjust row count to maintain rate ~0.5
                m_new = int(n * 0.5)
                H = H[:m_new, :]
                logger.debug(f"Truncated matrix to {H.shape}")
            else:
                # Key is longer than matrix - need to extend
                # For simplicity, use the largest available matrix and truncate key
                # or pad matrix (padding is complex, so we truncate key)
                logger.warning(
                    f"Key length {n} exceeds largest matrix {H.shape[1]}. "
                    f"Using matrix size {H.shape[1]}."
                )
                # Don't modify matrix, caller will need to truncate key
        
        return H

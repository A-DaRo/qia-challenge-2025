"""
Alice's protocol for E-HOK Baseline.

Complete implementation of all 5 phases:
1. Quantum Generation
2. Commitment (receive)
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
from ..core.exceptions import CommitmentVerificationError, ReconciliationFailedError
from ..quantum.batching_manager import BatchingManager, EPRGenerator
from ..quantum.measurement import MeasurementBuffer
from ..implementations.commitment.sha256_commitment import SHA256Commitment
from ..implementations.reconciliation.ldpc_reconciliator import LDPCReconciliator
from ..implementations.privacy_amplification.toeplitz_amplifier import ToeplitzAmplifier

logger = LogManager.get_stack_logger("ehok.protocols.alice")


class AliceEHOKProgram(Program):
    """
    Alice's program for complete E-HOK baseline protocol.
    
    Alice plays the role of the sender who:
    - Creates and measures EPR pairs
    - Receives Bob's commitment before revealing bases
    - Performs sifting and verifies Bob's honesty
    - Computes syndrome for reconciliation
    - Generates privacy amplification seed
    - Outputs ObliviousKey with full knowledge
    """
    PEER_NAME = "bob"

    def __init__(self, total_pairs: int = TOTAL_EPR_PAIRS):
        """
        Initialize Alice's program.
        
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
            name="alice_ehok",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=5,
        )

    def run(self, context: ProgramContext) -> Generator[EventExpression, None, Dict[str, Any]]:
        """
        Execute Alice's complete E-HOK protocol.
        
        Yields
        ------
        EventExpression
            NetSquid events for simulation scheduling.
        
        Returns
        -------
        dict
            Protocol results including ObliviousKey and statistics.
        """
        logger.info("=== Alice E-HOK Protocol Started ===")
        self.context = context
        
        # Initialize components
        self.epr_socket = context.epr_sockets[self.PEER_NAME]
        self.csocket = context.csockets[self.PEER_NAME]
        
        self.batching_manager = BatchingManager(self.total_pairs, BATCH_SIZE)
        self.epr_generator = EPRGenerator(self.epr_socket, "alice")
        
        self.commitment_scheme = SHA256Commitment()
        self.sifting = SiftingManager()

        # ===== PHASE 1: Quantum Generation =====
        yield from self._phase1_quantum_generation()
        
        # Extract arrays for classical processing
        outcomes_alice = self.measurement_buffer.get_outcomes()
        bases_alice = self.measurement_buffer.get_bases()
        
        # ===== PHASE 2: Commitment (Receive from Bob) =====
        commitment = yield from self._phase2_receive_commitment()
        
        # ===== PHASE 3: Sifting & Sampling =====
        I_0, I_1, test_set, key_set, qber = yield from self._phase3_sifting_sampling(
            outcomes_alice, bases_alice, commitment
        )
        
        # ===== PHASE 4: Information Reconciliation =====
        alice_key = outcomes_alice[key_set]
        yield from self._phase4_reconciliation(alice_key)
        
        # ===== PHASE 5: Privacy Amplification =====
        oblivious_key = yield from self._phase5_privacy_amplification(
            alice_key, qber, I_1
        )
        
        logger.info("=== Alice E-HOK Protocol Complete ===")
        logger.info(f"Final key length: {oblivious_key.final_length} bits")
        logger.info(f"QBER: {qber*100:.2f}%")
        logger.info(f"Security parameter: {oblivious_key.security_param}")
        
        return {
            "success": True,
            "oblivious_key": oblivious_key,
            "qber": qber,
            "raw_count": len(outcomes_alice),
            "sifted_count": len(I_0),
            "final_count": oblivious_key.final_length,
            "role": "alice"
        }

    def _phase1_quantum_generation(self) -> Generator[EventExpression, None, None]:
        """
        Execute Phase 1: Generate and measure EPR pairs in batches.
        
        Yields
        ------
        EventExpression
            Flush events for quantum operations.
        """
        logger.info("=== PHASE 1: Quantum Generation ===")
        
        batch_sizes = self.batching_manager.compute_batch_sizes()
        
        for i, batch_size in enumerate(batch_sizes):
            logger.debug(f"Processing batch {i+1}/{len(batch_sizes)} (size={batch_size})")
            
            # 1. Create EPR pairs (returns futures)
            sim_time = ns.sim_time()
            qubits = self.epr_generator.generate_batch_alice(batch_size, sim_time)
            
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
    
    def _phase2_receive_commitment(self) -> Generator[EventExpression, None, bytes]:
        """
        Execute Phase 2: Receive commitment from Bob.
        
        SECURITY CRITICAL: Alice must receive Bob's commitment BEFORE revealing
        her bases. This ordering ensures Bob cannot adapt his bases to Alice's.
        
        Yields
        ------
        EventExpression
            Classical socket receive event.
        
        Returns
        -------
        bytes
            Bob's commitment hash.
        """
        logger.info("=== PHASE 2: Commitment ===")
        
        # Block until Bob's commitment arrives
        commitment_msg = yield from self.csocket.recv()
        commitment = bytes.fromhex(commitment_msg)
        
        logger.info(f"Received commitment from Bob: {commitment.hex()[:16]}...")
        
        return commitment
    
    def _phase3_sifting_sampling(
        self, 
        outcomes_alice: np.ndarray,
        bases_alice: np.ndarray,
        commitment: bytes
    ) -> Generator[EventExpression, None, tuple]:
        """
        Execute Phase 3: Sifting & Sampling with verification.
        
        Parameters
        ----------
        outcomes_alice : np.ndarray
            Alice's measurement outcomes.
        bases_alice : np.ndarray
            Alice's measurement bases.
        commitment : bytes
            Bob's commitment hash.
        
        Yields
        ------
        EventExpression
            Classical communication events.
        
        Returns
        -------
        tuple
            (I_0, I_1, test_set, key_set, qber)
        """
        logger.info("=== PHASE 3: Sifting & Sampling ===")
        
        # 1. Send bases to Bob
        bases_msg = bases_alice.tobytes().hex()
        self.csocket.send(bases_msg)
        logger.debug(f"Sent bases to Bob: {len(bases_alice)} bases")
        
        # 2. Receive Bob's full data (outcomes + bases)
        bob_data_msg = yield from self.csocket.recv()
        bob_data = np.frombuffer(bytes.fromhex(bob_data_msg), dtype=np.uint8)
        
        # Split into outcomes, bases, and salt
        n = len(outcomes_alice)
        outcomes_bob = bob_data[:n]
        bases_bob = bob_data[n:2*n]
        decommitment_salt = bob_data[2*n:].tobytes()
        
        logger.debug(f"Received Bob's data: {len(outcomes_bob)} outcomes, {len(bases_bob)} bases")
        
        # 3. Identify matching bases (sifting)
        I_0, I_1 = self.sifting.identify_matching_bases(bases_alice, bases_bob)
        
        # 4. Select test set from sifted key
        test_set, key_set = self.sifting.select_test_set(I_0)
        
        # 5. Verify commitment on the ENTIRE data
        # (SHA256 scheme requires full data for verification)
        full_bob_data = np.concatenate([outcomes_bob, bases_bob])
        if not self.commitment_scheme.verify(commitment, full_bob_data, decommitment_salt):
            raise CommitmentVerificationError("Bob's commitment verification failed")
        
        logger.info("Commitment verified successfully")
        
        # 6. Estimate QBER on test set
        qber = self.sifting.estimate_qber(outcomes_alice, outcomes_bob, test_set)
        
        # 7. Check abort condition
        self.sifting.check_qber_abort(qber)
        
        return I_0, I_1, test_set, key_set, qber
    
    def _phase4_reconciliation(
        self, 
        alice_key: np.ndarray
    ) -> Generator[EventExpression, None, None]:
        """
        Execute Phase 4: Information Reconciliation.
        
        Alice computes syndrome and sends to Bob. Bob decodes and confirms.
        
        Parameters
        ----------
        alice_key : np.ndarray
            Alice's sifted key (after removing test set).
        
        Yields
        ------
        EventExpression
            Classical communication events.
        """
        logger.info("=== PHASE 4: Information Reconciliation ===")
        
        # 1. Load LDPC matrix
        sifted_length = len(alice_key)
        H = self._load_ldpc_matrix(sifted_length)
        self.reconciliator = LDPCReconciliator(H)
        
        # 2. Handle key length mismatch
        if sifted_length > H.shape[1]:
            logger.warning(
                f"Truncating key from {sifted_length} to {H.shape[1]} to match LDPC matrix"
            )
            alice_key = alice_key[:H.shape[1]]
        
        # 3. Compute syndrome
        syndrome = self.reconciliator.compute_syndrome(alice_key)
        logger.info(f"Syndrome computed: {len(syndrome)} bits, weight={np.sum(syndrome)}")
        logger.debug(f"Alice key length: {len(alice_key)}, LDPC matrix shape: {H.shape}")
        logger.debug(f"Alice key hash BEFORE reconciliation: {hashlib.sha256(alice_key.tobytes()).hexdigest()[:16]}...")
        
        # 4. Send syndrome to Bob
        syndrome_msg = syndrome.tobytes().hex()
        self.csocket.send(syndrome_msg)
        
        # 5. Wait for Bob's reconciled key hash confirmation
        bob_hash_msg = yield from self.csocket.recv()
        bob_hash = bob_hash_msg
        
        # 6. Compute Alice's key hash
        alice_hash = hashlib.sha256(alice_key.tobytes()).hexdigest()
        
        # 7. Verify match
        if bob_hash != alice_hash:
            raise ReconciliationFailedError(
                f"Key hash mismatch after reconciliation. "
                f"Alice: {alice_hash[:16]}..., Bob: {bob_hash[:16]}..."
            )
        
        logger.info("Reconciliation successful: keys match")
    
    def _phase5_privacy_amplification(
        self,
        alice_key: np.ndarray,
        qber: float,
        I_1: np.ndarray
    ) -> Generator[EventExpression, None, ObliviousKey]:
        """
        Execute Phase 5: Privacy Amplification.
        
        Parameters
        ----------
        alice_key : np.ndarray
            Reconciled sifted key.
        qber : float
            Measured QBER.
        I_1 : np.ndarray
            Indices of mismatched bases (for knowledge mask).
        
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
        
        # 1. Estimate information leakage
        syndrome_length = self.reconciliator.m  # Number of syndrome bits sent
        amplifier = ToeplitzAmplifier()
        leakage = self.reconciliator.estimate_leakage(syndrome_length, qber)
        
        # 2. Compute final key length
        final_length = amplifier.compute_final_length(
            len(alice_key), qber, leakage
        )
        
        logger.info(f"Compressing key: {len(alice_key)} → {final_length} bits")
        
        # 3. Generate Toeplitz seed
        seed = amplifier.generate_hash_seed(len(alice_key), final_length)
        
        # 4. Send seed to Bob
        seed_msg = seed.tobytes().hex()
        self.csocket.send(seed_msg)
        
        # 5. Compress key
        final_key = amplifier.compress(alice_key, seed)
        
        # 6. Construct ObliviousKey
        # Alice knows everything (knowledge_mask = all zeros)
        knowledge_mask = np.zeros_like(final_key)
        
        oblivious_key = ObliviousKey(
            key_value=final_key,
            knowledge_mask=knowledge_mask,
            security_param=TARGET_EPSILON_SEC,
            qber=qber,
            final_length=final_length
        )
        
        logger.info(f"Privacy amplification complete: {final_length}-bit key")
        
        # Yield to allow final message to be sent
        yield from self.context.connection.flush()
        
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


"""Bob role program for Caligo Phase E."""

from __future__ import annotations

from typing import Any, Dict, Generator, Tuple

import numpy as np

from caligo.amplification import OTOutputFormatter
from caligo.connection.envelope import MessageType
from caligo.protocol.base import CaligoProgram, ProtocolParameters
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

        # Receive test indices and respond with test outcomes.
        idx_msg = yield from self._ordered_socket.recv(MessageType.INDEX_LISTS)
        test_indices = np.array(idx_msg["test_indices"], dtype=np.int64)
        test_bits = bob_outcomes[test_indices].astype(np.uint8)

        yield from self._ordered_socket.send(
            MessageType.TEST_OUTCOMES,
            {"test_bits": test_bits.tobytes().hex()},
        )

        # Receive reconciled key material + partition metadata.
        recon_msg = yield from self._ordered_socket.recv(MessageType.SYNDROME_RESPONSE)
        reconciled_np = np.frombuffer(
            bytes.fromhex(str(recon_msg["reconciled"])), dtype=np.uint8
        )
        matching_indices = np.array(recon_msg["matching_indices"], dtype=np.int64)
        i0_indices = np.array(recon_msg["i0_indices"], dtype=np.int64)
        i1_indices = np.array(recon_msg["i1_indices"], dtype=np.int64)

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
            i0_indices=i0_indices,
            i1_indices=i1_indices,
            matching_indices=matching_indices,
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

    def _phase1_quantum_and_commit(
        self, context
    ) -> Generator[Any, None, Tuple[np.ndarray, np.ndarray, bytes, bytes]]:
        """Receive and measure EPR pairs (Bob side), then commit."""

        from caligo.quantum import BasisSelector, MeasurementExecutor

        epr_socket = context.epr_sockets[self.PEER]
        basis_selector = BasisSelector()
        meas = MeasurementExecutor()

        n = int(self.params.num_pairs)
        bases = basis_selector.select_batch(n)
        outcomes = np.zeros(n, dtype=np.uint8)

        # NetQASM limits recv_keep batch size by max_qubits.
        batch_size = max(1, int(self.params.num_qubits))
        for start in range(0, n, batch_size):
            count = min(batch_size, n - start)
            qubits = epr_socket.recv_keep(number=count)
            for j, q in enumerate(qubits):
                round_id = start + j
                outcomes[round_id] = yield from meas.measure_qubit(
                    qubit=q,
                    basis=int(bases[round_id]),
                    round_id=round_id,
                    context=context,
                )

        data_bytes = np.concatenate([outcomes, bases]).astype(np.uint8).tobytes()
        commit_res = self._commitment.commit(data_bytes)

        return outcomes.astype(np.uint8), bases.astype(np.uint8), commit_res.nonce, commit_res.commitment

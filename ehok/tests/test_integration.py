"""
Integration tests for Phase 6: Protocol sequencing and synchronization checks.

These tests verify that the protocol enforces correct classical ordering
and that quantum measurement results require a `connection.flush()` before
being resolved.
"""

import time
from typing import Dict, Any

import pytest
import numpy as np

from squidasm.run.stack.run import run
from squidasm.run.stack.config import StackNetworkConfig, StackConfig, LinkConfig
from squidasm.run.stack.config import DepolariseLinkConfig

from ehok.protocols.base import EHOKRole
from ehok.core.config import ProtocolConfig
from ehok.core.data_structures import ObliviousKey
from ehok.utils.logging import get_logger

logger = get_logger("test.integration")


def _make_perfect_network_config() -> StackNetworkConfig:
    alice_cfg = StackConfig.perfect_generic_config("alice")
    bob_cfg = StackConfig.perfect_generic_config("bob")
    link = LinkConfig.perfect_config("alice", "bob")
    return StackNetworkConfig(stacks=[alice_cfg, bob_cfg], links=[link])


class AlicePhaseSeq(EHOKRole):
    ROLE = "alice"
    PEER_NAME = "bob"

    def _execute_remaining_phases(self, quantum_result):
        # Wait for commitment then send bases, record times
        # quantum_result provides bases and measurement records
        bases = quantum_result.bases
        csocket = self.context.csockets[self.PEER_NAME]

        # Blocking receive for commit
        t_before_recv = time.time_ns()
        commit_msg = yield from csocket.recv()
        t_commit_recv = time.time_ns()

        # Send bases after receiving commitment
        bases_msg = bases.tobytes().hex()
        t_before_send = time.time_ns()
        csocket.send(bases_msg)
        t_bases_send = time.time_ns()

        result = {
            "role": "alice",
            "t_commit_recv": t_commit_recv,
            "t_bases_send": t_bases_send,
            "measurement_records": quantum_result.measurement_records,
        }
        return result


class BobPhaseSeq(EHOKRole):
    ROLE = "bob"
    PEER_NAME = "alice"

    def _execute_remaining_phases(self, quantum_result):
        csocket = self.context.csockets[self.PEER_NAME]

        # Send commitment before receiving bases
        t_commit_sent = time.time_ns()
        # Use a fixed commitment value for the test
        csocket.send((b"commitment-test").hex())

        # Now block until bases arrive
        t_before_recv = time.time_ns()
        bases_msg = yield from csocket.recv()
        t_bases_recv = time.time_ns()

        result = {
            "role": "bob",
            "t_commit_sent": t_commit_sent,
            "t_bases_recv": t_bases_recv,
            "measurement_records": quantum_result.measurement_records,
        }
        return result


import pytest


@pytest.mark.long
def test_phase_sequencing_commitment_before_bases():
    """Test that Bob's commitment is sent before Alice reveals bases."""
    config = _make_perfect_network_config()

    alice = AlicePhaseSeq()
    bob = BobPhaseSeq()

    results = run(config=config, programs={"alice": alice, "bob": bob}, num_times=1)

    # Find alice and bob results
    alice_res = None
    bob_res = None
    for stack_res in results:
        res = stack_res[0]
        if res["role"] == "alice":
            alice_res = res
        elif res["role"] == "bob":
            bob_res = res

    assert alice_res is not None and bob_res is not None

    # Check ordering
    # Bob must have sent the commitment before Alice sent bases
    assert bob_res["t_commit_sent"] < alice_res["t_bases_send"], (
        "Commitment not sent before bases: "
        f"t_commit_sent={bob_res['t_commit_sent']}, t_bases_send={alice_res['t_bases_send']}"
    )


class AliceNoFlush(EHOKRole):
    ROLE = "alice"
    PEER_NAME = "bob"

    def _execute_remaining_phases(self, quantum_result):
        csocket = self.context.csockets[self.PEER_NAME]
        # Directly call create_measure and try to read without flush
        epr_socket = self.context.epr_sockets[self.PEER_NAME]
        results = epr_socket.create_measure(number=10)

        raised_before_flush = False
        first_val = None
        try:
            # Accessing the future as int before flush should raise NoValueError
            _ = int(results[0].measurement_outcome)
        except Exception:
            raised_before_flush = True

        # Now flush and access again
        yield from self.context.connection.flush()
        first_val = int(results[0].measurement_outcome)

        return {"role": "alice", "raised_before_flush": raised_before_flush, "value_after_flush": first_val}


class BobNoFlush(EHOKRole):
    ROLE = "bob"
    PEER_NAME = "alice"

    def _execute_remaining_phases(self, quantum_result):
        csocket = self.context.csockets[self.PEER_NAME]
        epr_socket = self.context.epr_sockets[self.PEER_NAME]
        results = epr_socket.recv_measure(number=10)

        raised_before_flush = False
        first_val = None
        try:
            _ = int(results[0].measurement_outcome)
        except Exception:
            raised_before_flush = True

        yield from self.context.connection.flush()
        first_val = int(results[0].measurement_outcome)

        return {"role": "bob", "raised_before_flush": raised_before_flush, "value_after_flush": first_val}


@pytest.mark.long
def test_synchronization_flush_required():
    """Test that flush() is required before accessing measurement outcomes."""
    config = _make_perfect_network_config()
    alice = AliceNoFlush()
    bob = BobNoFlush()

    results = run(config=config, programs={"alice": alice, "bob": bob}, num_times=1)

    alice_res = None
    bob_res = None
    for stack_res in results:
        res = stack_res[0]
        if res["role"] == "alice":
            alice_res = res
        elif res["role"] == "bob":
            bob_res = res

    assert alice_res is not None and bob_res is not None
    assert alice_res["raised_before_flush"] is True
    assert bob_res["raised_before_flush"] is True
    assert alice_res["value_after_flush"] in (0, 1)
    assert bob_res["value_after_flush"] in (0, 1)

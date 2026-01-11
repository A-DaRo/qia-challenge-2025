"""Phase E runner for Alice/Bob SquidASM programs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from caligo.protocol.alice import AliceProgram
from caligo.protocol.base import ProtocolParameters
from caligo.protocol.bob import BobProgram
from caligo.simulation.network_builder import (
    perfect_network_config,
    validate_network_config,
)
from caligo.types.phase_contracts import ObliviousTransferOutput
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


def run_protocol(
    params: ProtocolParameters,
    *,
    bob_choice_bit: int = 0,
    network_config: Optional[Any] = None,
) -> Tuple[ObliviousTransferOutput, Dict[str, Any]]:
    """Run the Caligo protocol using SquidASM's stack runner.

    Parameters
    ----------
    params : ProtocolParameters
        Protocol parameters.
    bob_choice_bit : int
        Bob's choice bit (0 or 1) for the receiver output.
    network_config : Optional[Any]
        Pre-built SquidASM StackNetworkConfig. If None, a perfect two-node
        config is created.

    Returns
    -------
    Tuple[ObliviousTransferOutput, Dict[str, Any]]
        The OT output plus raw program result dictionary.

    Raises
    ------
    UnsupportedHardwareError
        If network_config uses NV or other unsupported hardware.
    """

    if network_config is None:
        network_config = perfect_network_config(
            alice_name="Alice", bob_name="Bob", num_qubits=params.num_qubits
        )

    # Validate hardware compatibility before attempting run.
    # This prevents cryptic MOV instruction errors from SquidASM.
    validate_network_config(network_config)

    try:
        from squidasm.run.stack.run import run as squidasm_run  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SquidASM is required to run the protocol; install squidasm and netsquid"
        ) from exc

    alice = AliceProgram(params=params)
    bob = BobProgram(params=params, choice_bit=bob_choice_bit)

    logger.debug(
        "Running Caligo protocol: n=%d, Δt=%s ns",
        params.num_pairs,
        params.nsm_params.delta_t_ns,
    )

    raw_results = squidasm_run(network_config, {"Alice": alice, "Bob": bob})

    # SquidASM stack runner returns List[List[Dict[str, Any]]]
    # where outer list is per stack and inner list is per iteration.
    # Convert to a name-indexed dictionary for Caligo.
    if isinstance(raw_results, list):
        stack_names = [getattr(s, "name", None) for s in getattr(network_config, "stacks", [])]
        if any(name is None for name in stack_names):
            raise TypeError(
                "Unexpected StackNetworkConfig: missing stack name(s)"
            )
        if len(raw_results) != len(stack_names):
            raise TypeError(
                f"Unexpected SquidASM result shape: got {len(raw_results)} stacks, "
                f"expected {len(stack_names)}"
            )

        results: Dict[str, Any] = {}
        for idx, node_name in enumerate(stack_names):
            per_stack = raw_results[idx]
            if not isinstance(per_stack, list):
                raise TypeError(
                    f"Unexpected SquidASM result shape for stack '{node_name}': {type(per_stack)}"
                )
            if len(per_stack) == 0:
                # This can happen if a program aborts early and the peer is left
                # blocked waiting for classical messages. Surface it as an abort
                # rather than a shape error.
                results[str(node_name)] = {
                    "role": str(node_name).lower(),
                    "aborted": True,
                    "reason": "No results produced (peer may have aborted)",
                }
                continue
            # We run each program once (num_times=1).
            results[str(node_name)] = per_stack[0]
    else:
        results = raw_results

    alice_res = results["Alice"]
    bob_res = results["Bob"]

    if bool(alice_res.get("aborted", False)) or bool(bob_res.get("aborted", False)):
        raise RuntimeError(
            f"Protocol aborted. Alice={alice_res.get('reason')}, Bob={bob_res.get('reason')}"
        )

    ot = ObliviousTransferOutput(
        alice_key=alice_res["alice_key"],
        bob_key=bob_res["bob_key"],
        protocol_succeeded=True,
        total_rounds=int(alice_res.get("total_rounds", params.num_pairs)),
        final_key_length=int(alice_res["key_length"]),
        security_parameter=1e-10,
        entropy_rate=float(alice_res.get("entropy_rate", 0.0)),
    )

    return ot, results

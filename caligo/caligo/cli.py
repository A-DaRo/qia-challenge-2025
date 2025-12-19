"""Command-line interface for Caligo utilities.

This CLI is intentionally minimal and primarily targets running the
EPR-generation layer in either sequential or parallel mode.

Notes
-----
Project convention is to prefer logging over printing; this module uses
Caligo's logger and returns a process exit code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict, Optional

import yaml

from caligo.quantum.factory import CaligoConfig, EPRGenerationFactory, ParallelEPRStrategy
from caligo.quantum.parallel import ParallelEPRConfig
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with parallel generation options.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Caligo utilities")

    parser.add_argument("--num-pairs", type=int, default=10_000)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    parallel_group = parser.add_argument_group("parallel generation")
    parallel_group.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel EPR generation",
    )
    parallel_group.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parallel_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="EPR pairs per worker batch",
    )

    network_group = parser.add_argument_group("network")
    network_group.add_argument(
        "--noise",
        type=float,
        default=None,
        help="Depolarizing noise rate (0..1)",
    )

    return parser


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError("YAML config must be a mapping")
    return data


def load_config(args: argparse.Namespace) -> CaligoConfig:
    """Load configuration from CLI args and/or YAML file.

    CLI arguments take precedence over YAML.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    CaligoConfig
        Fully configured Caligo configuration.
    """
    yaml_cfg: Dict[str, Any] = {}
    if args.config is not None:
        yaml_cfg = _load_yaml(args.config)

    num_pairs = int(args.num_pairs if args.num_pairs is not None else yaml_cfg.get("num_epr_pairs", 10_000))

    yaml_parallel = yaml_cfg.get("parallel", {}) if isinstance(yaml_cfg.get("parallel", {}), dict) else {}

    enabled = bool(args.parallel) or bool(yaml_parallel.get("enabled", False))

    workers: Optional[int] = args.workers
    if workers is None:
        workers_val = yaml_parallel.get("workers", None)
        workers = int(workers_val) if workers_val is not None else None

    batch_size: Optional[int] = args.batch_size
    if batch_size is None:
        batch_val = yaml_parallel.get("batch_size", None)
        batch_size = int(batch_val) if batch_val is not None else None

    parallel_cfg = ParallelEPRConfig(
        enabled=enabled,
        num_workers=workers if workers is not None else ParallelEPRConfig().num_workers,
        pairs_per_batch=batch_size if batch_size is not None else ParallelEPRConfig().pairs_per_batch,
        isolation_level=str(yaml_parallel.get("isolation_level", "process")),
        prefetch_batches=int(yaml_parallel.get("prefetch_batches", 2)),
    )

    yaml_network = yaml_cfg.get("network", {}) if isinstance(yaml_cfg.get("network", {}), dict) else {}
    noise = args.noise if args.noise is not None else yaml_network.get("noise", 0.0)

    network_cfg: Dict[str, Any] = {"noise": float(noise)}

    return CaligoConfig(
        num_epr_pairs=num_pairs,
        parallel_config=parallel_cfg,
        network_config=network_cfg,
        security_epsilon=float(yaml_cfg.get("security", {}).get("target_security_parameter", 1e-10))
        if isinstance(yaml_cfg.get("security", {}), dict)
        else 1e-10,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint.

    Parameters
    ----------
    argv : Optional[list[str]]
        Optional argument vector for testing.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args)
    logger.info("Loaded config: %s", asdict(config))

    factory = EPRGenerationFactory(config)
    strategy = factory.create_strategy()

    try:
        alice_out, alice_bases, bob_out, bob_bases = strategy.generate(config.num_epr_pairs)
        logger.info(
            "Generated %d pairs (mode=%s)",
            len(alice_out),
            "parallel" if config.parallel_config.enabled else "sequential",
        )
        # Minimal QBER summary on matching bases.
        matching = [i for i in range(len(alice_bases)) if alice_bases[i] == bob_bases[i]]
        if len(matching) > 0:
            errors = sum(1 for i in matching if alice_out[i] != bob_out[i])
            logger.info("Empirical QBER (matching bases): %.4f", errors / len(matching))
        return 0
    finally:
        if isinstance(strategy, ParallelEPRStrategy):
            strategy.shutdown()

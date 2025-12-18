"""
Reconciliation Factory for Runtime Type Selection.

This module provides a factory pattern for selecting reconciliation strategies
at runtime based on YAML configuration. It supports three reconciliation types:

1. **baseline**: Standard rate-adaptive LDPC reconciliation with explicit QBER
   estimation and code rate selection based on measured channel parameters.

2. **blind**: Martinez-Mateo et al. blind reconciliation protocol that uses
   iterative syndrome decoding without prior QBER estimation. Uses NSM
   channel parameters to inform initial rate selection.

3. **interactive**: Reserved for future cascade-style reconciliation (NYI).

Usage
-----
Via YAML configuration:

.. code-block:: yaml

    reconciliation:
      type: "blind"
      frame_size: 4096
      max_iterations: 3
      use_nsm_informed_start: true

Via Python API:

.. code-block:: python

    from caligo.reconciliation.factory import create_reconciler, ReconciliationType
    from caligo.simulation.physical_model import NSMParameters

    config = ReconciliationConfig.from_yaml(yaml_config)
    nsm_params = NSMParameters.from_erven_experimental()
    reconciler = create_reconciler(config, nsm_params)

References
----------
- Martinez-Mateo et al. (2012): Blind Reconciliation
- Elkouss et al. (2009): Rate-compatible LDPC codes
- Erven et al. (2014): NSM experimental implementation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, TYPE_CHECKING

import numpy as np

from caligo.utils.logging import get_logger

if TYPE_CHECKING:
    from caligo.simulation.physical_model import NSMParameters
    from caligo.simulation.noise_models import ChannelNoiseProfile


logger = get_logger(__name__)


# Default LDPC matrix path
DEFAULT_LDPC_MATRIX_PATH = str(
    Path(__file__).parent.parent / "configs" / "ldpc_matrices"
)


# =============================================================================
# Reconciliation Type Enum
# =============================================================================


class ReconciliationType(Enum):
    """
    Available reconciliation protocol types.

    **QBER Estimation Requirements:**

    - BASELINE: **Requires** prior QBER estimation from test bits
    - BLIND: **Skips** QBER estimation (core advantage of blind method)
    - INTERACTIVE: **Requires** prior QBER estimation

    Attributes
    ----------
    BASELINE : str
        Standard rate-adaptive LDPC reconciliation.
        Requires explicit QBER estimation before reconciliation.
    BLIND : str
        Blind reconciliation (Martinez-Mateo et al. 2012).
        No prior QBER estimation; uses iterative syndrome decoding.
        This is the **key advantage** of blind reconciliation.
    INTERACTIVE : str
        Interactive cascade-style reconciliation.
        Not yet implemented; reserved for future development.
    """

    BASELINE = "baseline"
    BLIND = "blind"
    INTERACTIVE = "interactive"

    @property
    def requires_qber_estimation(self) -> bool:
        """
        Whether this reconciliation type requires prior QBER estimation.

        Returns
        -------
        bool
            True for BASELINE and INTERACTIVE, False for BLIND.

        Notes
        -----
        Blind reconciliation's core advantage is eliminating the need
        for QBER pre-estimation. The protocol iteratively discovers
        the appropriate code rate through syndrome-based feedback.
        """
        return self != ReconciliationType.BLIND

    @classmethod
    def from_string(cls, value: str) -> ReconciliationType:
        """
        Create ReconciliationType from string value.

        Parameters
        ----------
        value : str
            String representation ("baseline", "blind", "interactive").

        Returns
        -------
        ReconciliationType
            Corresponding enum value.

        Raises
        ------
        ValueError
            If value is not a valid reconciliation type.
        """
        value_lower = value.lower().strip()
        try:
            return cls(value_lower)
        except ValueError:
            valid = [t.value for t in cls]
            raise ValueError(
                f"Invalid reconciliation type '{value}'. "
                f"Valid types: {valid}"
            )


# =============================================================================
# Reconciliation Configuration
# =============================================================================


@dataclass
class ReconciliationConfig:
    """
    Configuration for reconciliation protocol execution.

    This dataclass encapsulates all parameters needed to initialize
    and execute a reconciliation protocol. It can be populated from
    YAML configuration or constructed programmatically.

    **QBER Estimation Policy:**
    - BASELINE/INTERACTIVE: Require measured QBER before reconciliation
    - BLIND: Skip QBER estimation (uses iterative rate discovery)

    Parameters
    ----------
    reconciliation_type : ReconciliationType
        Protocol type to use. Default: BASELINE.
    frame_size : int
        LDPC codeword block size in bits. Default: 4096.
    max_iterations : int
        Maximum belief propagation iterations. Default: 50.
    target_rate : float, optional
        Target code rate for baseline reconciliation.
        If None, computed from channel QBER.
    use_nsm_informed_start : bool
        For blind reconciliation, use NSM channel parameters
        to select initial rate. Default: True.
    safety_margin : float
        Capacity margin for rate selection (0-0.1). Default: 0.05.
    max_blind_rounds : int
        Maximum rounds for blind reconciliation. Default: 3.
    puncturing_enabled : bool
        Enable puncturing for rate adaptation. Default: True.
    shortening_enabled : bool
        Enable shortening for rate adaptation. Default: True.
    ldpc_matrix_path : str, optional
        Path to pre-computed LDPC matrices.

    Raises
    ------
    ValueError
        If configuration parameters are invalid.
    """

    reconciliation_type: ReconciliationType = ReconciliationType.BASELINE
    frame_size: int = 4096
    max_iterations: int = 50
    target_rate: Optional[float] = None
    use_nsm_informed_start: bool = True
    safety_margin: float = 0.05
    max_blind_rounds: int = 3
    puncturing_enabled: bool = True
    shortening_enabled: bool = True
    ldpc_matrix_path: Optional[str] = None
    _extra_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.frame_size < 256:
            raise ValueError(f"frame_size must be >= 256, got {self.frame_size}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.target_rate is not None and not 0.1 <= self.target_rate <= 0.99:
            raise ValueError(f"target_rate must be in [0.1, 0.99], got {self.target_rate}")
        if not 0 <= self.safety_margin <= 0.2:
            raise ValueError(f"safety_margin must be in [0, 0.2], got {self.safety_margin}")
        if self.max_blind_rounds < 1 or self.max_blind_rounds > 10:
            raise ValueError(f"max_blind_rounds must be in [1, 10], got {self.max_blind_rounds}")

    @property
    def requires_qber_estimation(self) -> bool:
        """
        Whether QBER must be estimated before reconciliation.

        Returns
        -------
        bool
            True for baseline/interactive, False for blind.
        """
        return self.reconciliation_type.requires_qber_estimation

    @property
    def skips_qber_estimation(self) -> bool:
        """
        Whether this config skips QBER pre-estimation (blind mode).

        This is the core advantage of blind reconciliation.

        Returns
        -------
        bool
            True only for blind reconciliation.
        """
        return not self.requires_qber_estimation

    @classmethod
    def from_dict(cls, config_dict: dict) -> ReconciliationConfig:
        """
        Create configuration from dictionary (e.g., parsed YAML).

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary with keys matching parameter names.
            The "type" key maps to reconciliation_type.

        Returns
        -------
        ReconciliationConfig
            Populated configuration object.

        Examples
        --------
        >>> config_dict = {
        ...     "type": "blind",
        ...     "frame_size": 4096,
        ...     "max_blind_rounds": 3,
        ... }
        >>> config = ReconciliationConfig.from_dict(config_dict)
        >>> config.reconciliation_type
        <ReconciliationType.BLIND: 'blind'>
        """
        # Extract reconciliation type
        type_str = config_dict.get("type", "baseline")
        recon_type = ReconciliationType.from_string(type_str)

        # Build kwargs for dataclass
        kwargs: dict[str, Any] = {
            "reconciliation_type": recon_type,
        }

        # Map config keys to dataclass fields
        field_mapping = {
            "frame_size": "frame_size",
            "max_iterations": "max_iterations",
            "target_rate": "target_rate",
            "use_nsm_informed_start": "use_nsm_informed_start",
            "safety_margin": "safety_margin",
            "max_blind_rounds": "max_blind_rounds",
            "puncturing_enabled": "puncturing_enabled",
            "shortening_enabled": "shortening_enabled",
            "ldpc_matrix_path": "ldpc_matrix_path",
        }

        for config_key, field_name in field_mapping.items():
            if config_key in config_dict:
                kwargs[field_name] = config_dict[config_key]

        # Store any extra parameters
        known_keys = set(field_mapping.keys()) | {"type"}
        extra = {k: v for k, v in config_dict.items() if k not in known_keys}
        kwargs["_extra_params"] = extra

        return cls(**kwargs)

    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> ReconciliationConfig:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file.

        Returns
        -------
        ReconciliationConfig
            Populated configuration object.
        """
        import yaml

        with open(yaml_path, "r") as f:
            full_config = yaml.safe_load(f)

        # Look for reconciliation section
        if "reconciliation" in full_config:
            return cls.from_dict(full_config["reconciliation"])
        else:
            return cls.from_dict(full_config)

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Returns
        -------
        dict
            Configuration as dictionary.
        """
        return {
            "type": self.reconciliation_type.value,
            "frame_size": self.frame_size,
            "max_iterations": self.max_iterations,
            "target_rate": self.target_rate,
            "use_nsm_informed_start": self.use_nsm_informed_start,
            "safety_margin": self.safety_margin,
            "max_blind_rounds": self.max_blind_rounds,
            "puncturing_enabled": self.puncturing_enabled,
            "shortening_enabled": self.shortening_enabled,
            "ldpc_matrix_path": self.ldpc_matrix_path,
            **self._extra_params,
        }


# =============================================================================
# Reconciler Protocol (Interface)
# =============================================================================


class Reconciler(Protocol):
    """
    Protocol defining the reconciler interface.

    Any reconciliation implementation must provide these methods
    to be compatible with the factory pattern.
    """

    def reconcile(
        self,
        alice_bits: bytes,
        bob_bits: bytes,
        *,
        syndrome: Optional[bytes] = None,
    ) -> tuple[bytes, dict]:
        """
        Perform information reconciliation.

        Parameters
        ----------
        alice_bits : bytes
            Alice's sifted key bits.
        bob_bits : bytes
            Bob's sifted key bits (may have errors).
        syndrome : bytes, optional
            Pre-computed syndrome for one-way reconciliation.

        Returns
        -------
        tuple[bytes, dict]
            Reconciled key and metadata dictionary.
        """
        ...

    def compute_syndrome(self, bits: bytes) -> bytes:
        """
        Compute LDPC syndrome for given bits.

        Parameters
        ----------
        bits : bytes
            Input bits.

        Returns
        -------
        bytes
            Syndrome bits.
        """
        ...


# =============================================================================
# Placeholder Reconciler Implementations
# =============================================================================


class BaselineReconciler:
    """
    Baseline rate-adaptive LDPC reconciler.

    **QBER Requirement:** This reconciler REQUIRES explicit QBER estimation
    before reconciliation. The measured QBER is used to select the optimal
    LDPC code rate for efficient syndrome transmission.

    Parameters
    ----------
    config : ReconciliationConfig
        Reconciliation configuration.
    nsm_params : NSMParameters, optional
        NSM parameters for QBER estimation.
    measured_qber : float, optional
        Measured QBER from test bits (required for rate selection).
    """

    def __init__(
        self,
        config: ReconciliationConfig,
        nsm_params: Optional[NSMParameters] = None,
        measured_qber: Optional[float] = None,
    ) -> None:
        self.config = config
        self.nsm_params = nsm_params
        self.measured_qber = measured_qber
        self._initialized = False

        logger.info(
            "BaselineReconciler created (requires QBER estimation)",
            extra={
                "frame_size": config.frame_size,
                "target_rate": config.target_rate,
                "measured_qber": measured_qber,
            },
        )

    def reconcile(
        self,
        alice_bits: bytes,
        bob_bits: bytes,
        *,
        measured_qber: Optional[float] = None,
        syndrome: Optional[bytes] = None,
    ) -> tuple[bytes, dict]:
        """
        Perform baseline LDPC reconciliation.

        **Requires measured QBER** - either at construction or call time.

        Parameters
        ----------
        alice_bits : bytes
            Alice's sifted key.
        bob_bits : bytes
            Bob's sifted key.
        measured_qber : float, optional
            Measured QBER (overrides constructor value).
        syndrome : bytes, optional
            Pre-computed syndrome.

        Returns
        -------
        tuple[bytes, dict]
            Reconciled key and metadata.

        Raises
        ------
        ValueError
            If no QBER estimate is available.
        """
        qber = measured_qber or self.measured_qber
        if qber is None:
            raise ValueError(
                "BaselineReconciler requires measured QBER. "
                "Use BlindReconciler to skip QBER estimation."
            )

        logger.warning("BaselineReconciler.reconcile() is a placeholder")

        metadata = {
            "reconciliation_type": "baseline",
            "frame_size": self.config.frame_size,
            "input_length": len(alice_bits),
            "measured_qber": qber,
            "qber_estimation_required": True,
            "status": "placeholder",
        }

        return alice_bits, metadata

    def compute_syndrome(self, bits: bytes) -> bytes:
        """Compute LDPC syndrome (placeholder)."""
        # Placeholder: return empty syndrome
        return b""


class BlindReconciler:
    """
    Blind reconciliation protocol (Martinez-Mateo et al. 2012).

    **Core Advantage:** NO prior QBER estimation required.

    This reconciler iteratively adjusts the code rate based on decoding
    success, eliminating the need for test bit sacrifice and QBER
    pre-estimation. This is the fundamental advantage over baseline
    reconciliation.

    Parameters
    ----------
    config : ReconciliationConfig
        Reconciliation configuration.
    nsm_params : NSMParameters, optional
        NSM parameters for informed initial rate selection.
        (Optional optimization, not required for correctness)
    """

    def __init__(
        self,
        config: ReconciliationConfig,
        nsm_params: Optional[NSMParameters] = None,
    ) -> None:
        self.config = config
        self.nsm_params = nsm_params
        self._orchestrator = None

        # Determine initial rate (NSM-informed is optional optimization)
        if nsm_params is not None and config.use_nsm_informed_start:
            self.initial_rate = nsm_params.suggested_ldpc_rate(
                safety_margin=config.safety_margin
            )
            self._nsm_qber = nsm_params.qber_channel
            logger.info(
                "BlindReconciler: NSM-informed start (optional optimization)",
                extra={
                    "initial_rate": self.initial_rate,
                    "nsm_qber_estimate": self._nsm_qber,
                },
            )
        else:
            # Default conservative starting rate - works without any QBER info
            self.initial_rate = 0.80
            self._nsm_qber = None
            logger.info(
                "BlindReconciler: No QBER estimation (core advantage)",
                extra={"initial_rate": self.initial_rate},
            )

    def _get_orchestrator(self):
        """Lazy initialization of orchestrator with dependencies."""
        if self._orchestrator is None:
            from caligo.reconciliation.orchestrator import (
                ReconciliationOrchestrator,
                ReconciliationOrchestratorConfig,
            )
            from caligo.reconciliation.matrix_manager import MatrixManager
            from caligo.reconciliation.leakage_tracker import LeakageTracker

            # Initialize matrix manager
            matrix_path = self.config.ldpc_matrix_path or DEFAULT_LDPC_MATRIX_PATH
            matrix_manager = MatrixManager.from_directory(Path(matrix_path))

            # Initialize leakage tracker
            leakage_tracker = LeakageTracker(safety_cap=0)

            # Create orchestrator config
            orch_config = ReconciliationOrchestratorConfig(
                frame_size=self.config.frame_size,
                max_iterations=self.config.max_iterations,
                max_retries=self.config.max_blind_rounds,
            )

            self._orchestrator = ReconciliationOrchestrator(
                matrix_manager=matrix_manager,
                leakage_tracker=leakage_tracker,
                config=orch_config,
            )

        return self._orchestrator

    def reconcile(
        self,
        alice_bits: bytes,
        bob_bits: bytes,
        *,
        syndrome: Optional[bytes] = None,
    ) -> tuple[bytes, dict]:
        """
        Perform blind reconciliation WITHOUT prior QBER estimation.

        The protocol iteratively discovers the correct rate:
        1. Alice sends syndrome at current rate
        2. Bob attempts decoding
        3. If failure, adjust rate and repeat (max 3 rounds)

        Parameters
        ----------
        alice_bits : bytes
            Alice's sifted key.
        bob_bits : bytes
            Bob's sifted key.
        syndrome : bytes, optional
            Pre-computed syndrome (usually None for blind).

        Returns
        -------
        tuple[bytes, dict]
            Reconciled key and metadata.
        """
        # Convert bytes to numpy arrays
        alice_array = np.frombuffer(alice_bits, dtype=np.uint8)
        bob_array = np.frombuffer(bob_bits, dtype=np.uint8)

        orchestrator = self._get_orchestrator()

        # For blind reconciliation, use NSM-estimated QBER or conservative default
        qber_estimate = self._nsm_qber if self._nsm_qber else 0.05

        # Reconcile using orchestrator
        result = orchestrator.reconcile_block(
            alice_key=alice_array,
            bob_key=bob_array,
            qber_estimate=qber_estimate,
            block_id=0,
        )

        # Convert result back to bytes
        corrected_bytes = result.corrected_payload.tobytes()

        metadata = {
            "reconciliation_type": "blind",
            "initial_rate": self.initial_rate,
            "max_rounds": self.config.max_blind_rounds,
            "nsm_informed": self.nsm_params is not None,
            "qber_estimation_required": False,  # Core advantage!
            "converged": result.converged,
            "verified": result.verified,
            "error_count": result.error_count,
            "syndrome_length": result.syndrome_length,
            "status": "success" if result.verified else "failed",
        }

        return corrected_bytes, metadata

    def compute_syndrome(self, bits: bytes) -> bytes:
        """
        Compute LDPC syndrome for given bits.

        Parameters
        ----------
        bits : bytes
            Input bits.

        Returns
        -------
        bytes
            Syndrome bits.
        """
        # Use orchestrator's encoder for consistency
        from caligo.reconciliation.ldpc_encoder import encode_block
        
        orchestrator = self._get_orchestrator()
        bits_array = np.frombuffer(bits, dtype=np.uint8)
        
        # Get default rate matrix
        rate = self.initial_rate
        H = orchestrator.matrix_manager.get_matrix(rate)

        frame_size = H.shape[1]
        payload_len = len(bits_array)
        n_shortened = frame_size - payload_len
        
        # Compute syndrome
        syndrome_block = encode_block(
            bits_array,
            H,
            rate,
            n_shortened,
            0,
        )
        
        return syndrome_block.syndrome.tobytes()


class InteractiveReconciler:
    """
    Interactive (Cascade-style) reconciliation.

    **QBER Requirement:** Requires QBER estimation (like baseline).

    Reserved for future implementation. This class raises
    NotImplementedError to clearly indicate NYI status.

    Parameters
    ----------
    config : ReconciliationConfig
        Reconciliation configuration.
    nsm_params : NSMParameters, optional
        NSM parameters (unused for interactive).
    measured_qber : float, optional
        Measured QBER from test bits (REQUIRED for interactive).
    """

    def __init__(
        self,
        config: ReconciliationConfig,
        nsm_params: Optional[NSMParameters] = None,
        measured_qber: Optional[float] = None,
    ) -> None:
        raise NotImplementedError(
            "InteractiveReconciler is not yet implemented. "
            "Use 'baseline' or 'blind' reconciliation types. "
            "Note: When implemented, interactive WILL require QBER estimation."
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_reconciler(
    config: ReconciliationConfig,
    nsm_params: Optional[NSMParameters] = None,
    channel_profile: Optional[ChannelNoiseProfile] = None,
) -> Reconciler:
    """
    Create a reconciler instance based on configuration.

    This factory function instantiates the appropriate reconciler
    class based on the reconciliation type specified in the config.

    Parameters
    ----------
    config : ReconciliationConfig
        Reconciliation configuration specifying type and parameters.
    nsm_params : NSMParameters, optional
        NSM parameters for QBER estimation and informed rate selection.
        If provided, used to compute channel QBER for blind reconciliation.
    channel_profile : ChannelNoiseProfile, optional
        Alternative noise profile. If nsm_params not provided,
        this can supply QBER estimates.

    Returns
    -------
    Reconciler
        Configured reconciler instance.

    Raises
    ------
    NotImplementedError
        If reconciliation type is INTERACTIVE (not yet implemented).
    ValueError
        If reconciliation type is invalid.

    Examples
    --------
    >>> from caligo.reconciliation.factory import (
    ...     create_reconciler,
    ...     ReconciliationConfig,
    ...     ReconciliationType,
    ... )
    >>> config = ReconciliationConfig(
    ...     reconciliation_type=ReconciliationType.BLIND,
    ...     frame_size=4096,
    ...     max_blind_rounds=3,
    ... )
    >>> reconciler = create_reconciler(config)

    With NSM parameters for informed blind reconciliation:

    >>> from caligo.simulation.physical_model import NSMParameters
    >>> nsm = NSMParameters.from_erven_experimental()
    >>> reconciler = create_reconciler(config, nsm_params=nsm)
    """
    recon_type = config.reconciliation_type

    logger.info(
        f"Creating reconciler",
        extra={
            "type": recon_type.value,
            "nsm_provided": nsm_params is not None,
            "profile_provided": channel_profile is not None,
        },
    )

    # Create appropriate reconciler based on type
    if recon_type == ReconciliationType.BASELINE:
        return BaselineReconciler(config, nsm_params)

    elif recon_type == ReconciliationType.BLIND:
        return BlindReconciler(config, nsm_params)

    elif recon_type == ReconciliationType.INTERACTIVE:
        return InteractiveReconciler(config, nsm_params)

    else:
        raise ValueError(f"Unknown reconciliation type: {recon_type}")


def create_reconciler_from_yaml(
    yaml_path: str,
    nsm_params: Optional[NSMParameters] = None,
) -> Reconciler:
    """
    Create reconciler from YAML configuration file.

    Convenience function that loads configuration from YAML
    and creates the appropriate reconciler.

    Parameters
    ----------
    yaml_path : str
        Path to YAML configuration file.
    nsm_params : NSMParameters, optional
        NSM parameters for informed reconciliation.

    Returns
    -------
    Reconciler
        Configured reconciler instance.

    Examples
    --------
    >>> reconciler = create_reconciler_from_yaml(
    ...     "config/reconciliation.yaml",
    ...     nsm_params=NSMParameters.from_erven_experimental(),
    ... )
    """
    config = ReconciliationConfig.from_yaml_file(yaml_path)
    return create_reconciler(config, nsm_params)

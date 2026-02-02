# SISO Codec Protocol Specification

<metadata>
spec_id: siso-codec-protocol
version: 1.0.0
status: accepted
created: 2026-02-02
depends_on: [ADR-0001]
enables: [ADR-0002, impl/phase1-rust-foundation, impl/phase3-strategy-refactor]
</metadata>

---

## Overview

<overview>
This specification defines a **runtime-checkable Protocol** for Soft-Input Soft-Output (SISO) codecs, providing a unified interface that abstracts over both LDPC and Polar code implementations.

### Design Goals

1. **Strategy Agnosticism:** Reconciliation strategies (Baseline, Blind) operate identically regardless of underlying codec.

2. **Codec Interchangeability:** LDPC and Polar implementations are swappable via configuration without code changes.

3. **Concatenation Readiness:** The `decode_soft()` method returns extrinsic LLRs, enabling future IC-LDPC-Polar concatenation per [1, §III].

4. **Clean Interface:** This is a **clean break** from the legacy `LDPCCodec` API. Legacy code uses an adapter pattern; no backward-compatible method names pollute the protocol.

### Non-Goals

- Runtime codec switching mid-session (requires restart)
- Heterogeneous codec mixing within single block
- Automatic rate adaptation (caller responsibility)
</overview>

---

## Interface Contract

<interface language="python">

```python
"""
SISO Codec Protocol for QKD Reconciliation.

This module defines the SISOCodec Protocol that all codec implementations
must satisfy. The protocol is runtime-checkable via typing.Protocol.

References
----------
[1] Abbas et al., "Concatenated LDPC-Polar Codes Decoding Through BP"
[2] Hua et al., "RC-LDPC-Polar Codes for CV-QKD Reconciliation"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SISOSoftDecodeResult:
    """
    Result container for soft-decision decoding.
    
    Attributes
    ----------
    llr_extrinsic : NDArray[np.float32]
        Extrinsic LLRs for outer decoder consumption in concatenated schemes.
        Shape: (block_length,). Computed as: posterior - prior - channel.
    decoded_message : NDArray[np.uint8]
        Hard-decision decoded message bits. Shape: (message_length,).
    converged : bool
        True if decoder reached valid codeword (BP) or CRC passed (SCL).
    iterations_used : int
        Number of iterations/stages consumed.
    path_metric : float
        Log-likelihood of selected path (SCL) or final syndrome weight (BP).
        Higher is better for SCL; lower is better for BP.
    """
    
    llr_extrinsic: NDArray[np.float32]
    decoded_message: NDArray[np.uint8]
    converged: bool
    iterations_used: int
    path_metric: float


@dataclass(frozen=True)
class SISOHardDecodeResult:
    """
    Result container for hard-decision decoding.
    
    Attributes
    ----------
    decoded_message : NDArray[np.uint8]
        Hard-decision decoded message bits. Shape: (message_length,).
    converged : bool
        True if decoder converged to valid codeword.
    iterations_used : int
        Number of iterations consumed.
    """
    
    decoded_message: NDArray[np.uint8]
    converged: bool
    iterations_used: int


@runtime_checkable
class SISOCodec(Protocol):
    """
    Soft-Input Soft-Output Codec Protocol for composable reconciliation.
    
    This protocol defines the interface that both LDPC and Polar codec
    implementations must satisfy. It enables:
    
    1. Strategy-agnostic reconciliation (Baseline, Blind protocols)
    2. Codec-agnostic implementation selection
    3. Future concatenation via extrinsic LLR exchange
    
    Implementations
    ---------------
    - PolarCodec: Primary implementation using CA-SCL decoding
    - LDPCCodecAdapter: Adapter wrapping legacy LDPCCodec
    
    Type Conventions
    ----------------
    - All bit arrays: np.uint8, values in {0, 1}
    - All LLR arrays: np.float32, LLR = log(P(bit=0) / P(bit=1))
    - All arrays: C-contiguous, 1-dimensional
    
    Thread Safety
    -------------
    Implementations MUST be thread-safe for concurrent decode() calls
    on different data. Encoding state (frozen bits, matrices) is immutable
    after construction.
    
    References
    ----------
    [1] ADR-0001: Polar Codec Adoption
    [2] Abbas et al., "Concatenated LDPC-Polar Codes"
    """
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def block_length(self) -> int:
        """
        Code block length N (codeword size after any shortening).
        
        Returns
        -------
        int
            Block length N. For Phase 1/2: N = 4096.
            Polar: Must be power of 2.
            LDPC: Arbitrary (typically 4096).
        """
        ...
    
    @property
    def message_length(self) -> int:
        """
        Information bit length k (excluding CRC for Polar).
        
        Returns
        -------
        int
            Message length k = ceil(rate × block_length) - crc_length.
        
        Notes
        -----
        For Polar with CRC-16: k = K - 16 where K is total information bits.
        For LDPC: k = N × R where R is code rate.
        """
        ...
    
    @property
    def rate(self) -> float:
        """
        Effective code rate R_eff = k / N.
        
        Returns
        -------
        float
            Code rate in (0, 1). Does not include CRC overhead.
        """
        ...
    
    @property
    def crc_length(self) -> int:
        """
        CRC bit length (0 for LDPC, typically 16 for Polar).
        
        Returns
        -------
        int
            Number of CRC bits appended to message before encoding.
        """
        ...
    
    # =========================================================================
    # Encoding
    # =========================================================================
    
    def encode(
        self,
        message: NDArray[np.uint8],
        *,
        frozen_values: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """
        Encode message bits to codeword.
        
        Parameters
        ----------
        message : NDArray[np.uint8]
            Information bits to encode. Shape: (message_length,).
            Values must be in {0, 1}.
        frozen_values : NDArray[np.uint8] | None, optional
            Polar: Values for frozen bit positions. Shape: (n_frozen,).
            If None, defaults to all-zeros (standard).
            LDPC: Ignored.
        
        Returns
        -------
        NDArray[np.uint8]
            Encoded codeword. Shape: (block_length,).
            For Polar: Includes CRC bits in systematic positions.
        
        Raises
        ------
        ValueError
            If message.shape[0] != self.message_length.
        ValueError
            If message contains values outside {0, 1}.
        TypeError
            If message.dtype != np.uint8.
        
        Notes
        -----
        For Polar codes, CRC is computed and inserted automatically.
        The message should NOT include CRC bits.
        
        Example
        -------
        >>> codec = PolarCodec(block_length=4096, rate=0.5)
        >>> message = np.random.randint(0, 2, codec.message_length, dtype=np.uint8)
        >>> codeword = codec.encode(message)
        >>> assert codeword.shape == (4096,)
        """
        ...
    
    # =========================================================================
    # Hard-Decision Decoding
    # =========================================================================
    
    def decode_hard(
        self,
        received: NDArray[np.uint8],
        *,
        syndrome: NDArray[np.uint8] | None = None,
        max_iterations: int = 60,
    ) -> SISOHardDecodeResult:
        """
        Hard-decision decoding for Baseline protocol.
        
        Performs error correction on received bits using syndrome constraint
        (if provided) or standard decoding.
        
        Parameters
        ----------
        received : NDArray[np.uint8]
            Received bits (possibly corrupted). Shape: (block_length,).
            Values should be in {0, 1}.
        syndrome : NDArray[np.uint8] | None, optional
            Alice's syndrome for syndrome-based decoding.
            Shape: ((1 - rate) × block_length,) for LDPC.
            For Polar: Used to set frozen bit constraints.
        max_iterations : int, optional
            Maximum decoder iterations. Default: 60.
            LDPC: BP iterations.
            Polar: Ignored (SCL is non-iterative per bit).
        
        Returns
        -------
        SISOHardDecodeResult
            Decoding result containing message, convergence, iterations.
        
        Raises
        ------
        ValueError
            If received.shape[0] != self.block_length.
        ValueError
            If syndrome shape mismatches expected size.
        
        Notes
        -----
        For Baseline protocol, syndrome-based decoding corrects Bob's bits
        to match Alice's codeword without revealing error positions.
        
        Example
        -------
        >>> result = codec.decode_hard(bob_bits, syndrome=alice_syndrome)
        >>> if result.converged:
        ...     corrected = result.decoded_message
        """
        ...
    
    # =========================================================================
    # Soft-Decision Decoding
    # =========================================================================
    
    def decode_soft(
        self,
        llr_channel: NDArray[np.float32],
        *,
        llr_prior: NDArray[np.float32] | None = None,
        syndrome: NDArray[np.uint8] | None = None,
        list_size: int = 8,
        max_iterations: int = 60,
    ) -> SISODecodeResult:
        """
        Soft-decision decoding with extrinsic LLR output.
        
        This method is critical for concatenated coding schemes. It accepts
        channel LLRs (and optionally prior LLRs from an outer decoder) and
        returns extrinsic LLRs suitable for iterative decoding.
        
        Parameters
        ----------
        llr_channel : NDArray[np.float32]
            Channel log-likelihood ratios. Shape: (block_length,).
            Convention: LLR = log(P(bit=0) / P(bit=1)).
            Positive LLR indicates bit=0 more likely.
        llr_prior : NDArray[np.float32] | None, optional
            Prior LLRs from outer decoder (for concatenation).
            Shape: (block_length,). If None, assumed all-zero (no prior).
        syndrome : NDArray[np.uint8] | None, optional
            Syndrome constraint for Blind protocol.
            Shape varies by codec; see decode_hard().
        list_size : int, optional
            SCL list size L for Polar codes. Default: 8.
            LDPC: Ignored (BP has no list).
            Valid values: 1, 2, 4, 8, 16, 32.
        max_iterations : int, optional
            Maximum iterations. Default: 60.
            LDPC: BP iterations.
            Polar: Ignored.
        
        Returns
        -------
        SISODecodeResult
            Result containing extrinsic LLRs, decoded message, convergence,
            iterations, and path metric.
        
        Raises
        ------
        ValueError
            If llr_channel.shape[0] != self.block_length.
        ValueError
            If llr_prior provided with mismatched shape.
        ValueError
            If list_size not in {1, 2, 4, 8, 16, 32}.
        
        Notes
        -----
        **Extrinsic LLR Computation:**
        
        The extrinsic LLRs are computed as:
        
            LLR_extrinsic = LLR_posterior - LLR_prior - LLR_channel
        
        This follows the standard SISO convention for turbo/concatenated
        decoding [1, §III]. For LDPC, LLR_posterior is the bit marginal
        from BP. For Polar SCL, it's derived from path metrics.
        
        **Path Metric Interpretation:**
        
        - Polar SCL: Log-likelihood of winning path. Higher = more confident.
        - LDPC BP: Negative syndrome weight. Higher (closer to 0) = converged.
        
        **Concatenation Usage:**
        
        Per [2, §3.2], joint BP decoding iterates:
        
        >>> # Outer LDPC → Inner Polar
        >>> polar_result = polar.decode_soft(llr_ch, llr_prior=ldpc_ext)
        >>> ldpc_result = ldpc.decode_soft(polar_result.llr_extrinsic)
        
        Example
        -------
        >>> llr = build_channel_llr(bob_bits, qber=0.05)
        >>> result = codec.decode_soft(llr, list_size=8)
        >>> if result.converged:
        ...     message = result.decoded_message
        ...     confidence = result.path_metric
        
        References
        ----------
        [1] Abbas et al., "Concatenated LDPC-Polar Codes Through BP"
        [2] Hua et al., "RC-LDPC-Polar Codes for CV-QKD"
        """
        ...
    
    # =========================================================================
    # Syndrome Computation
    # =========================================================================
    
    def compute_syndrome(
        self,
        codeword: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """
        Compute syndrome from codeword (Alice's operation).
        
        Parameters
        ----------
        codeword : NDArray[np.uint8]
            Valid codeword. Shape: (block_length,).
        
        Returns
        -------
        NDArray[np.uint8]
            Syndrome bits. Shape depends on codec:
            - LDPC: ((1 - rate) × block_length,)
            - Polar: (n_frozen,) frozen bit values
        
        Raises
        ------
        ValueError
            If codeword.shape[0] != self.block_length.
        
        Notes
        -----
        For LDPC: syndrome = H @ codeword (mod 2).
        For Polar: syndrome = codeword[frozen_indices].
        
        The syndrome is transmitted from Alice to Bob for syndrome-based
        decoding in both Baseline and Blind protocols.
        """
        ...
```

</interface>

---

## Invariants

<invariants>

### Encoding-Decoding Round-Trip

For any valid message `m` and error-free channel:

```python
codeword = codec.encode(m)
result = codec.decode_hard(codeword)
assert result.converged
assert np.array_equal(result.decoded_message, m)
```

### LLR Sign Convention

Throughout the codebase, LLRs follow the convention:

$$\text{LLR} = \log\frac{P(\text{bit} = 0)}{P(\text{bit} = 1)}$$

- **Positive LLR** → bit = 0 more likely
- **Negative LLR** → bit = 1 more likely
- **LLR = 0** → maximum uncertainty

### Extrinsic LLR Computation

The extrinsic information excludes the inputs to enable iterative decoding:

$$\text{LLR}_{\text{ext}} = \text{LLR}_{\text{post}} - \text{LLR}_{\text{prior}} - \text{LLR}_{\text{channel}}$$

Where:
- $\text{LLR}_{\text{post}}$: Posterior belief after decoding
- $\text{LLR}_{\text{prior}}$: Prior from outer decoder (0 if none)
- $\text{LLR}_{\text{channel}}$: Original channel observation

### Memory Ownership

All returned arrays are **newly allocated**. No aliasing with internal state:

```python
result1 = codec.decode_soft(llr)
result2 = codec.decode_soft(llr)
assert result1.llr_extrinsic is not result2.llr_extrinsic  # No aliasing
```

### Thread Safety

Concurrent `decode_*` calls on the same codec instance with different data are safe:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(codec.decode_soft, llr_batch[i]) for i in range(4)]
    results = [f.result() for f in futures]  # All succeed
```

### Dtype Consistency

| Array Type | Dtype | Value Range |
|------------|-------|-------------|
| Bit arrays | `np.uint8` | {0, 1} |
| LLR arrays | `np.float32` | (-∞, +∞), typically [-100, 100] |
| Syndrome | `np.uint8` | {0, 1} |

</invariants>

---

## Error Conditions

<errors>

| Exception | Condition | Recovery Strategy |
|-----------|-----------|-------------------|
| `ValueError` | Input shape mismatch | Caller must reshape or pad |
| `ValueError` | LLR contains NaN/Inf | Replace with clipped values (±100) |
| `ValueError` | Invalid list_size | Use default (8) or valid power of 2 |
| `TypeError` | Wrong dtype | Convert with `.astype()` |
| `RuntimeError` | Decoder divergence (BP) | Retry with damped LLRs (×0.8) |
| `RuntimeError` | No CRC match in list (SCL) | Return best path, `converged=False` |

### Error Handling Example

```python
def safe_decode(codec: SISOCodec, llr: NDArray[np.float32]) -> SISODecodeResult:
    """Decode with automatic error recovery."""
    # Sanitize LLRs
    llr = np.clip(llr, -100.0, 100.0).astype(np.float32)
    
    result = codec.decode_soft(llr)
    
    if not result.converged:
        # Retry with damped LLRs
        result = codec.decode_soft(llr * 0.8, list_size=32)
    
    return result
```

</errors>

---

## Implementation Guide

<implementation_notes>

### PolarCodec Implementation

**Location:** `caligo/reconciliation/strategies/polar_codec.py`

```python
class PolarCodec:
    """
    Polar code implementation with CA-SCL decoding.
    
    Wraps Rust `polar-scl` crate via PyO3 bindings.
    
    Parameters
    ----------
    block_length : int
        Code block length N. Must be power of 2. Default: 4096.
    rate : float
        Target code rate R. Default: 0.5.
    crc_poly : int
        CRC polynomial. Default: 0x8005 (CRC-16-CCITT).
    frozen_construction : str
        Method for frozen bit selection. Default: "gaussian_approx".
    design_snr : float
        Design SNR for frozen bit construction. Default: 1.0 dB.
    """
    
    def __init__(
        self,
        block_length: int = 4096,
        rate: float = 0.5,
        crc_poly: int = 0x8005,
        frozen_construction: str = "gaussian_approx",
        design_snr: float = 1.0,
    ) -> None:
        # Validate power of 2
        if block_length & (block_length - 1) != 0:
            raise ValueError(f"block_length must be power of 2, got {block_length}")
        
        self._n = block_length
        self._rate = rate
        self._crc_len = 16
        self._k = int(rate * block_length) - self._crc_len
        
        # Initialize Rust backend
        self._rust_codec = polar_scl.PolarCodec(
            n=block_length,
            k=self._k + self._crc_len,
            crc_poly=crc_poly,
            frozen_method=frozen_construction,
            design_snr=design_snr,
        )
    
    # ... implement Protocol methods wrapping Rust calls
```

### LDPCCodecAdapter Implementation

**Location:** `caligo/reconciliation/strategies/ldpc_adapter.py`

```python
class LDPCCodecAdapter:
    """
    Adapter wrapping legacy LDPCCodec to SISOCodec protocol.
    
    This adapter enables existing LDPC infrastructure to work with
    the new strategy interface without modification.
    
    Parameters
    ----------
    legacy_codec : LDPCCodec
        Existing LDPCCodec instance to wrap.
    """
    
    def __init__(self, legacy_codec: LDPCCodec) -> None:
        self._legacy = legacy_codec
        self._n = legacy_codec._frame_size
        self._rate = 0.5  # Mother code rate
    
    @property
    def crc_length(self) -> int:
        return 0  # LDPC has no CRC
    
    def decode_soft(
        self,
        llr_channel: NDArray[np.float32],
        *,
        llr_prior: NDArray[np.float32] | None = None,
        syndrome: NDArray[np.uint8] | None = None,
        list_size: int = 8,  # Ignored for LDPC
        max_iterations: int = 60,
    ) -> SISODecodeResult:
        """
        Soft decode with extrinsic extraction from BP messages.
        
        Extrinsic LLRs are computed from edge messages after BP convergence.
        """
        # Combine prior if provided
        llr_total = llr_channel
        if llr_prior is not None:
            llr_total = llr_channel + llr_prior
        
        # Call legacy decoder (returns DecoderResult)
        legacy_result = self._legacy.decode_baseline(
            syndrome=syndrome,
            llr=llr_total,
            pattern=np.ones(self._n, dtype=np.uint8),
            max_iterations=max_iterations,
        )
        
        # Extract extrinsic: posterior - prior - channel
        llr_posterior = self._compute_posterior_from_messages(
            legacy_result.messages
        )
        llr_extrinsic = llr_posterior - llr_total
        
        return SISODecodeResult(
            llr_extrinsic=llr_extrinsic.astype(np.float32),
            decoded_message=legacy_result.corrected_bits[:self.message_length],
            converged=legacy_result.converged,
            iterations_used=legacy_result.iterations,
            path_metric=-float(np.sum(syndrome != self.compute_syndrome(
                legacy_result.corrected_bits
            ))) if syndrome is not None else 0.0,
        )
```

### Factory Pattern for Codec Selection

**Location:** `caligo/reconciliation/strategies/codec_factory.py`

```python
from enum import Enum, auto
from typing import Union

class CodecType(Enum):
    POLAR = auto()
    LDPC = auto()

def create_codec(
    codec_type: CodecType,
    block_length: int = 4096,
    rate: float = 0.5,
    **kwargs,
) -> SISOCodec:
    """
    Factory function for codec instantiation.
    
    Parameters
    ----------
    codec_type : CodecType
        POLAR or LDPC.
    block_length : int
        Code block length.
    rate : float
        Target code rate.
    **kwargs
        Codec-specific parameters (list_size, crc_poly, etc.)
    
    Returns
    -------
    SISOCodec
        Configured codec instance.
    """
    if codec_type == CodecType.POLAR:
        from caligo.reconciliation.strategies.polar_codec import PolarCodec
        return PolarCodec(
            block_length=block_length,
            rate=rate,
            crc_poly=kwargs.get("crc_poly", 0x8005),
            design_snr=kwargs.get("design_snr", 1.0),
        )
    elif codec_type == CodecType.LDPC:
        from caligo.reconciliation.strategies.ldpc_adapter import LDPCCodecAdapter
        from caligo.reconciliation.strategies.codec import LDPCCodec
        from caligo.reconciliation.matrix_manager import MotherCodeManager
        
        mother = MotherCodeManager.get_instance()
        legacy = LDPCCodec(mother)
        return LDPCCodecAdapter(legacy)
    else:
        raise ValueError(f"Unknown codec type: {codec_type}")
```

</implementation_notes>

---

## Test Vectors

<test_vectors>

### Round-Trip Encoding Test

```python
def test_encode_decode_roundtrip(codec: SISOCodec):
    """Verify encoding-decoding preserves message."""
    rng = np.random.default_rng(42)
    message = rng.integers(0, 2, codec.message_length, dtype=np.uint8)
    
    codeword = codec.encode(message)
    assert codeword.shape == (codec.block_length,)
    
    result = codec.decode_hard(codeword)
    assert result.converged
    np.testing.assert_array_equal(result.decoded_message, message)
```

### Soft Decode with Noise

```python
def test_soft_decode_with_noise(codec: SISOCodec):
    """Verify soft decoding corrects errors at moderate QBER."""
    rng = np.random.default_rng(42)
    message = rng.integers(0, 2, codec.message_length, dtype=np.uint8)
    codeword = codec.encode(message)
    
    # Simulate BSC with QBER = 0.05
    qber = 0.05
    errors = rng.random(codec.block_length) < qber
    received = (codeword ^ errors.astype(np.uint8))
    
    # Build channel LLRs
    llr_magnitude = np.log((1 - qber) / qber)
    llr_channel = llr_magnitude * (1 - 2 * received).astype(np.float32)
    
    result = codec.decode_soft(llr_channel, list_size=8)
    
    # Should decode correctly at QBER=0.05 with rate 0.5
    assert result.converged
    np.testing.assert_array_equal(result.decoded_message, message)
```

### Extrinsic LLR Sanity Check

```python
def test_extrinsic_llr_computation(codec: SISOCodec):
    """Verify extrinsic LLRs have correct sign and magnitude."""
    rng = np.random.default_rng(42)
    
    # Strong positive LLRs (confident bit=0)
    llr_channel = np.full(codec.block_length, 10.0, dtype=np.float32)
    
    result = codec.decode_soft(llr_channel)
    
    # Extrinsic should not amplify beyond reasonable bounds
    assert np.all(np.abs(result.llr_extrinsic) < 200)
    
    # Decoded bits should be all zeros (matching LLR sign)
    assert np.all(result.decoded_message == 0)
```

### Protocol Conformance Test

```python
def test_protocol_conformance(codec: SISOCodec):
    """Verify codec satisfies SISOCodec protocol."""
    from typing import runtime_checkable, Protocol
    
    # Runtime check
    assert isinstance(codec, SISOCodec)
    
    # Property checks
    assert isinstance(codec.block_length, int)
    assert isinstance(codec.message_length, int)
    assert isinstance(codec.rate, float)
    assert isinstance(codec.crc_length, int)
    
    assert codec.block_length > 0
    assert codec.message_length > 0
    assert 0 < codec.rate < 1
    assert codec.crc_length >= 0
    
    # Consistency check
    expected_k = int(codec.rate * codec.block_length)
    assert abs(codec.message_length + codec.crc_length - expected_k) <= 1
```

</test_vectors>

---

## References

<references>

[1] S. M. Abbas, Y. Fan, J. Chen, and C.-Y. Tsui, "Concatenated LDPC-Polar Codes Decoding Through Belief Propagation," *Proc. IEEE ISCAS*, 2017.
    - Key sections: §II-B (SISO message format), §III (joint BP interface)

[2] F. Hua et al., "RC-LDPC-Polar Codes for Information Reconciliation in CV-QKD," *MDPI Electronics*, 2024.
    - Key sections: §2 (decoder interface requirements), §3.2-3.3 (rate-compatible SISO)

[3] ADR-0001: Polar Codec Adoption for QKD Reconciliation
    - Rationale for Polar adoption and parameter selection

[4] Existing codebase references:
    - `caligo/reconciliation/strategies/codec.py`: Legacy LDPCCodec
    - `caligo/reconciliation/strategies/__init__.py`: DecoderResult dataclass

</references>

---

## Changelog

<changelog>

### v1.0.0 (2026-02-02)

- Initial specification
- Defines SISOCodec Protocol with encode, decode_hard, decode_soft
- Establishes SISODecodeResult and SISOHardDecodeResult dataclasses
- Documents LLR conventions, invariants, error conditions
- Provides implementation guide for PolarCodec and LDPCCodecAdapter

</changelog>

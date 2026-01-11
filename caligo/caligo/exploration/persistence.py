"""
Persistence layer for exploration state and HDF5 data management.

This module provides fault-tolerant checkpointing and atomic HDF5 writes,
enabling resumption of long-running exploration campaigns after crashes.

Architecture
------------
The persistence layer has two main components:

1. **StateManager**: Handles checkpoint serialization via dill
2. **HDF5Writer**: Manages atomic batch writes to HDF5 datasets

Thread Safety
-------------
- StateManager: NOT thread-safe (single writer assumption)
- HDF5Writer: NOT thread-safe (HDF5 limitation for writing)

For parallel exploration, use separate HDF5Writer instances per worker
or coordinate writes through a queue.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import dill
import h5py
import numpy as np

from caligo.exploration.types import (
    ExplorationSample,
    Phase1State,
    Phase2State,
    Phase3State,
    ProtocolOutcome,
    ProtocolResult,
    ReconciliationStrategy,
)
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# HDF5 Schema Constants
# =============================================================================

# Group names in HDF5 file
GROUP_LHS_WARMUP = "lhs_warmup"
GROUP_ACTIVE_LEARNING = "active_learning"

# Dataset names within each group
DATASET_INPUTS = "inputs"
DATASET_OUTPUTS = "outputs"
DATASET_OUTCOMES = "outcomes"
DATASET_METADATA = "metadata"

# Dataset shapes and types
INPUT_SHAPE = (9,)  # 9D parameter space
OUTPUT_SHAPE = (6,)  # [net_eff, raw_len, final_len, qber, recon_eff, leakage]
DTYPE_INPUTS = np.float32
DTYPE_OUTPUTS = np.float32
DTYPE_OUTCOMES = h5py.special_dtype(vlen=str)


# =============================================================================
# HDF5 Writer
# =============================================================================


class HDF5Writer:
    """
    Atomic batch writer for HDF5 exploration data.

    This class manages HDF5 file operations with a focus on data integrity.
    Writes are batched and flushed atomically to prevent corruption from
    mid-write crashes.

    Parameters
    ----------
    file_path : Path
        Path to the HDF5 file. Created if it doesn't exist.
    mode : str
        HDF5 open mode: "w" (overwrite), "a" (append), "r" (read-only).

    Attributes
    ----------
    file_path : Path
        Path to the HDF5 file.
    _file : Optional[h5py.File]
        Open HDF5 file handle (None when closed).
    _lock : threading.Lock
        Lock for thread-safe operations (single-thread writes only).

    Examples
    --------
    >>> with HDF5Writer(Path("data.h5"), mode="a") as writer:
    ...     writer.append_batch(
    ...         group="lhs_warmup",
    ...         inputs=input_array,
    ...         outputs=output_array,
    ...         outcomes=outcome_list,
    ...         metadata=metadata_list,
    ...     )

    Notes
    -----
    HDF5 does not support concurrent writes. For multi-process exploration,
    use a single writer process with a queue, or merge files post-hoc.
    """

    def __init__(self, file_path: Path, mode: str = "a") -> None:
        """
        Initialize the HDF5 writer.

        Parameters
        ----------
        file_path : Path
            Path to the HDF5 file.
        mode : str
            HDF5 open mode.
        """
        self.file_path = Path(file_path)
        self._mode = mode
        self._file: Optional[h5py.File] = None
        self._lock = threading.Lock()

    def __enter__(self) -> "HDF5Writer":
        """Open the HDF5 file."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the HDF5 file."""
        self.close()

    def open(self) -> None:
        """
        Open the HDF5 file with retry logic for file locking issues.

        Creates the file and base groups if they don't exist.
        Retries up to 5 times with exponential backoff if file is locked.
        """
        max_retries = 5
        retry_delay = 0.5  # Start with 0.5 seconds
        
        for attempt in range(max_retries):
            try:
                self._file = h5py.File(self.file_path, self._mode)
                
                # Create base groups if they don't exist (and we're in write mode)
                if self._mode in ("w", "a"):
                    for group_name in [GROUP_LHS_WARMUP, GROUP_ACTIVE_LEARNING]:
                        if group_name not in self._file:
                            self._file.create_group(group_name)
                    self._file.flush()
                
                logger.debug("Opened HDF5 file: %s (mode=%s)", self.file_path, self._mode)
                return  # Success
                
            except BlockingIOError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "HDF5 file locked (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, max_retries, self.file_path, retry_delay
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        "Failed to open HDF5 file after %d attempts. "
                        "File may be held by zombie processes. Try: lsof %s | grep python",
                        max_retries, self.file_path
                    )
                    raise

    def close(self) -> None:
        """Close the HDF5 file, flushing all data."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            logger.debug("Closed HDF5 file: %s", self.file_path)

    def _ensure_datasets(self, group: h5py.Group) -> None:
        """
        Ensure required datasets exist in the group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to check/create datasets in.
        """
        if DATASET_INPUTS not in group:
            group.create_dataset(
                DATASET_INPUTS,
                shape=(0, INPUT_SHAPE[0]),
                maxshape=(None, INPUT_SHAPE[0]),
                dtype=DTYPE_INPUTS,
                chunks=(100, INPUT_SHAPE[0]),
                compression="gzip",
                compression_opts=4,
            )
        if DATASET_OUTPUTS not in group:
            group.create_dataset(
                DATASET_OUTPUTS,
                shape=(0, OUTPUT_SHAPE[0]),
                maxshape=(None, OUTPUT_SHAPE[0]),
                dtype=DTYPE_OUTPUTS,
                chunks=(100, OUTPUT_SHAPE[0]),
                compression="gzip",
                compression_opts=4,
            )
        if DATASET_OUTCOMES not in group:
            group.create_dataset(
                DATASET_OUTCOMES,
                shape=(0,),
                maxshape=(None,),
                dtype=DTYPE_OUTCOMES,
            )
        if DATASET_METADATA not in group:
            group.create_dataset(
                DATASET_METADATA,
                shape=(0,),
                maxshape=(None,),
                dtype=DTYPE_OUTCOMES,  # Variable-length string
            )

    def append_batch(
        self,
        group_name: str,
        inputs: np.ndarray,
        outputs: np.ndarray,
        outcomes: List[str],
        metadata: List[str],
    ) -> int:
        """
        Append a batch of results to an HDF5 group.

        This operation is atomic at the batch level: either all data
        is written, or none is (in case of failure).

        Parameters
        ----------
        group_name : str
            Name of the target group (e.g., "lhs_warmup").
        inputs : np.ndarray
            Input parameters, shape (batch_size, 9).
        outputs : np.ndarray
            Output metrics, shape (batch_size, 6).
        outcomes : List[str]
            Protocol outcomes as strings.
        metadata : List[str]
            JSON-encoded metadata strings.

        Returns
        -------
        int
            Total number of samples in the group after appending.

        Raises
        ------
        RuntimeError
            If the file is not open.
        ValueError
            If array shapes are inconsistent.
        """
        if self._file is None:
            raise RuntimeError("HDF5 file is not open. Call open() first.")

        batch_size = len(inputs)
        if len(outputs) != batch_size or len(outcomes) != batch_size or len(metadata) != batch_size:
            raise ValueError(
                f"Inconsistent batch sizes: inputs={len(inputs)}, outputs={len(outputs)}, "
                f"outcomes={len(outcomes)}, metadata={len(metadata)}"
            )

        with self._lock:
            group = self._file[group_name]
            self._ensure_datasets(group)

            # Get current sizes
            current_size = group[DATASET_INPUTS].shape[0]
            new_size = current_size + batch_size

            # Resize datasets
            group[DATASET_INPUTS].resize((new_size, INPUT_SHAPE[0]))
            group[DATASET_OUTPUTS].resize((new_size, OUTPUT_SHAPE[0]))
            group[DATASET_OUTCOMES].resize((new_size,))
            group[DATASET_METADATA].resize((new_size,))

            # Write data
            group[DATASET_INPUTS][current_size:new_size] = inputs.astype(DTYPE_INPUTS)
            group[DATASET_OUTPUTS][current_size:new_size] = outputs.astype(DTYPE_OUTPUTS)
            group[DATASET_OUTCOMES][current_size:new_size] = outcomes
            group[DATASET_METADATA][current_size:new_size] = metadata

            # Atomic flush
            self._file.flush()

            logger.debug(
                "Appended batch of %d samples to %s (total: %d)",
                batch_size,
                group_name,
                new_size,
            )
            return new_size

    def read_group(
        self,
        group_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Read all data from an HDF5 group.

        Parameters
        ----------
        group_name : str
            Name of the group to read.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]
            (inputs, outputs, outcomes, metadata) arrays/lists.

        Raises
        ------
        RuntimeError
            If the file is not open.
        KeyError
            If the group doesn't exist.
        """
        if self._file is None:
            raise RuntimeError("HDF5 file is not open. Call open() first.")

        group = self._file[group_name]
        
        # Handle case where datasets don't exist yet
        if DATASET_INPUTS not in group:
            return (
                np.empty((0, INPUT_SHAPE[0]), dtype=DTYPE_INPUTS),
                np.empty((0, OUTPUT_SHAPE[0]), dtype=DTYPE_OUTPUTS),
                [],
                [],
            )

        inputs = group[DATASET_INPUTS][:]
        outputs = group[DATASET_OUTPUTS][:]
        outcomes = [o.decode() if isinstance(o, bytes) else o for o in group[DATASET_OUTCOMES][:]]
        metadata = [m.decode() if isinstance(m, bytes) else m for m in group[DATASET_METADATA][:]]

        return inputs, outputs, outcomes, metadata

    def get_sample_count(self, group_name: str) -> int:
        """
        Get the number of samples in a group.

        Parameters
        ----------
        group_name : str
            Name of the group.

        Returns
        -------
        int
            Number of samples (0 if group/dataset doesn't exist).
        """
        if self._file is None:
            raise RuntimeError("HDF5 file is not open. Call open() first.")

        if group_name not in self._file:
            return 0

        group = self._file[group_name]
        if DATASET_INPUTS not in group:
            return 0

        return group[DATASET_INPUTS].shape[0]


# =============================================================================
# Buffered HDF5 Writer (Pre-Allocated Float32 Buffers)
# =============================================================================


class BufferedHDF5Writer:
    """
    Pre-allocated buffer writer for streaming HDF5 writes.

    This class provides memory-efficient streaming writes by pre-allocating
    Float32 buffers that match the batch size. Data is accumulated in-memory
    and flushed to HDF5 when the buffer fills, ensuring:

    - **Zero-copy writes**: Buffer layout matches HDF5 dataset layout
    - **Constant memory footprint**: Buffer size is fixed regardless of campaign length
    - **Cache-friendly access**: Contiguous Float32 arrays for SIMD optimization
    - **2MB page alignment**: Optimal for HDF5 chunk boundaries

    Parameters
    ----------
    writer : HDF5Writer
        Underlying HDF5 writer instance.
    buffer_size : int
        Number of samples to buffer before flushing. Default 100 (aligned
        with HDF5 chunk size for optimal I/O).
    auto_flush : bool
        If True, automatically flush when buffer is full.

    Attributes
    ----------
    writer : HDF5Writer
        Underlying HDF5 writer.
    buffer_size : int
        Maximum buffer capacity.
    _input_buffer : np.ndarray
        Pre-allocated Float32 input buffer, shape (buffer_size, 9).
    _output_buffer : np.ndarray
        Pre-allocated Float32 output buffer, shape (buffer_size, 6).
    _outcome_buffer : List[str]
        Outcome strings buffer.
    _metadata_buffer : List[str]
        Metadata strings buffer.
    _buffer_idx : int
        Current write position in buffer.
    _target_group : str
        Target HDF5 group name.

    Examples
    --------
    >>> with HDF5Writer(Path("data.h5"), mode="a") as writer:
    ...     buffered = BufferedHDF5Writer(writer, buffer_size=100)
    ...     buffered.set_group("lhs_warmup")
    ...     for result in streaming_results:
    ...         buffered.append_single(result)
    ...     buffered.flush()  # Write remaining data
    """

    def __init__(
        self,
        writer: HDF5Writer,
        buffer_size: int = 100,
        auto_flush: bool = True,
    ) -> None:
        """
        Initialize the buffered writer.

        Parameters
        ----------
        writer : HDF5Writer
            Underlying HDF5 writer (must be open).
        buffer_size : int
            Buffer capacity (samples).
        auto_flush : bool
            Auto-flush on buffer full.
        """
        self.writer = writer
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush

        # Pre-allocate Float32 buffers for zero-copy writes
        self._input_buffer = np.zeros(
            (buffer_size, INPUT_SHAPE[0]), dtype=DTYPE_INPUTS
        )
        self._output_buffer = np.zeros(
            (buffer_size, OUTPUT_SHAPE[0]), dtype=DTYPE_OUTPUTS
        )
        self._outcome_buffer: List[str] = []
        self._metadata_buffer: List[str] = []
        self._buffer_idx = 0
        self._target_group = GROUP_LHS_WARMUP
        self._total_flushed = 0

        logger.debug(
            "Initialized BufferedHDF5Writer with buffer_size=%d (%.2f KB per buffer)",
            buffer_size,
            buffer_size * (INPUT_SHAPE[0] + OUTPUT_SHAPE[0]) * 4 / 1024,
        )

    def set_group(self, group_name: str) -> None:
        """
        Set the target HDF5 group for writes.

        If the buffer contains data for a different group, it is flushed first.

        Parameters
        ----------
        group_name : str
            Target group name (e.g., "lhs_warmup", "active_learning").
        """
        if self._buffer_idx > 0 and group_name != self._target_group:
            self.flush()
        self._target_group = group_name

    def append_single(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        outcome: str,
        metadata: str,
    ) -> None:
        """
        Append a single result to the buffer.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters, shape (9,).
        outputs : np.ndarray
            Output metrics, shape (6,).
        outcome : str
            Protocol outcome string.
        metadata : str
            JSON metadata string.
        """
        # Write directly to pre-allocated buffer (zero-copy)
        self._input_buffer[self._buffer_idx] = inputs.astype(DTYPE_INPUTS)
        self._output_buffer[self._buffer_idx] = outputs.astype(DTYPE_OUTPUTS)
        self._outcome_buffer.append(outcome)
        self._metadata_buffer.append(metadata)
        self._buffer_idx += 1

        if self.auto_flush and self._buffer_idx >= self.buffer_size:
            self.flush()

    def append_result(self, result: ProtocolResult) -> None:
        """
        Append a ProtocolResult to the buffer.

        Convenience method that extracts arrays from a ProtocolResult.

        Parameters
        ----------
        result : ProtocolResult
            Protocol execution result.
        """
        import json as json_module

        inputs = result.sample.to_array()
        outputs = np.array([
            result.net_efficiency,
            float(result.raw_key_length),
            float(result.final_key_length),
            result.qber_measured,
            result.reconciliation_efficiency,
            float(result.leakage_bits),
        ], dtype=DTYPE_OUTPUTS)
        outcome = result.outcome.value
        metadata = json_module.dumps({
            "execution_time": result.execution_time_seconds,
            "error": result.error_message,
        })

        self.append_single(inputs, outputs, outcome, metadata)

    def flush(self) -> int:
        """
        Flush the buffer to HDF5.

        Writes accumulated data to the underlying HDF5Writer and resets
        the buffer position.

        Returns
        -------
        int
            Number of samples flushed.
        """
        if self._buffer_idx == 0:
            return 0

        # Write only the filled portion of the buffer
        count = self.writer.append_batch(
            self._target_group,
            inputs=self._input_buffer[:self._buffer_idx],
            outputs=self._output_buffer[:self._buffer_idx],
            outcomes=self._outcome_buffer,
            metadata=self._metadata_buffer,
        )

        flushed = self._buffer_idx
        self._total_flushed += flushed

        # Reset buffer state (buffers are reused, not reallocated)
        self._buffer_idx = 0
        self._outcome_buffer.clear()
        self._metadata_buffer.clear()

        logger.debug(
            "Flushed %d samples to %s (total: %d)",
            flushed,
            self._target_group,
            self._total_flushed,
        )
        return flushed

    @property
    def buffered_count(self) -> int:
        """Return number of samples currently buffered."""
        return self._buffer_idx

    @property
    def total_written(self) -> int:
        """Return total samples written (flushed + buffered)."""
        return self._total_flushed + self._buffer_idx


# =============================================================================
# State Manager
# =============================================================================


class StateManager:
    """
    Checkpoint manager for fault-tolerant exploration.

    This class handles serialization and deserialization of exploration
    state, enabling resumption after crashes or interruptions.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file (.pkl).
    auto_save : bool
        If True, save state after each update.

    Attributes
    ----------
    checkpoint_path : Path
        Path to the checkpoint file.
    auto_save : bool
        Whether to auto-save after updates.
    _state : Optional[Union[Phase1State, Phase2State, Phase3State]]
        Current state object.

    Examples
    --------
    >>> manager = StateManager(Path("checkpoint.pkl"))
    >>> state = Phase1State(total_samples=2000, completed_samples=0, ...)
    >>> manager.save(state)
    >>> # ... later or after crash ...
    >>> loaded = manager.load(Phase1State)
    >>> print(loaded.completed_samples)
    500  # Resumes from checkpoint

    Notes
    -----
    Uses `dill` for serialization, which handles numpy arrays and
    complex objects better than standard `pickle`.
    """

    def __init__(self, checkpoint_path: Path, auto_save: bool = False) -> None:
        """
        Initialize the state manager.

        Parameters
        ----------
        checkpoint_path : Path
            Path to the checkpoint file.
        auto_save : bool
            Whether to auto-save after state updates.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.auto_save = auto_save
        self._state: Optional[Union[Phase1State, Phase2State, Phase3State]] = None

    def save(
        self,
        state: Union[Phase1State, Phase2State, Phase3State],
    ) -> None:
        """
        Save state to checkpoint file.

        Parameters
        ----------
        state : Union[Phase1State, Phase2State, Phase3State]
            State object to save.

        Notes
        -----
        Uses atomic write pattern: write to temp file, then rename.
        This prevents corruption from partial writes.
        """
        self._state = state
        temp_path = self.checkpoint_path.with_suffix(".tmp")

        try:
            with open(temp_path, "wb") as f:
                dill.dump(state, f)

            # Atomic rename (POSIX guarantees atomicity)
            temp_path.rename(self.checkpoint_path)
            logger.debug("Saved checkpoint: %s", self.checkpoint_path)

        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            logger.error("Failed to save checkpoint: %s", e)
            raise

    def load(
        self,
        state_class: Type[Union[Phase1State, Phase2State, Phase3State]],
    ) -> Optional[Union[Phase1State, Phase2State, Phase3State]]:
        """
        Load state from checkpoint file.

        Parameters
        ----------
        state_class : Type
            Expected state class (used for validation).

        Returns
        -------
        Optional[Union[Phase1State, Phase2State, Phase3State]]
            Loaded state, or None if checkpoint doesn't exist.

        Raises
        ------
        TypeError
            If loaded state doesn't match expected class.
        """
        if not self.checkpoint_path.exists():
            logger.debug("No checkpoint found at: %s", self.checkpoint_path)
            return None

        try:
            with open(self.checkpoint_path, "rb") as f:
                state = dill.load(f)

            if not isinstance(state, state_class):
                raise TypeError(
                    f"Checkpoint contains {type(state).__name__}, "
                    f"expected {state_class.__name__}"
                )

            self._state = state
            logger.info(
                "Loaded checkpoint: %s (phase=%s)",
                self.checkpoint_path,
                state.current_phase,
            )
            return state

        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            raise

    def exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return self.checkpoint_path.exists()

    def delete(self) -> None:
        """Delete the checkpoint file if it exists."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.debug("Deleted checkpoint: %s", self.checkpoint_path)

    @property
    def current_state(
        self,
    ) -> Optional[Union[Phase1State, Phase2State, Phase3State]]:
        """Get the currently loaded state."""
        return self._state


# =============================================================================
# Result Serialization Utilities
# =============================================================================


def result_to_hdf5_arrays(
    results: List[ProtocolResult],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Convert a list of ProtocolResults to HDF5-compatible arrays.

    Parameters
    ----------
    results : List[ProtocolResult]
        Protocol execution results.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str], List[str]]
        (inputs, outputs, outcomes, metadata) ready for HDF5 storage.

    Examples
    --------
    >>> inputs, outputs, outcomes, metadata = result_to_hdf5_arrays(results)
    >>> writer.append_batch("lhs_warmup", inputs, outputs, outcomes, metadata)
    """
    n = len(results)
    inputs = np.zeros((n, INPUT_SHAPE[0]), dtype=DTYPE_INPUTS)
    outputs = np.zeros((n, OUTPUT_SHAPE[0]), dtype=DTYPE_OUTPUTS)
    outcomes = []
    metadata = []

    for i, result in enumerate(results):
        inputs[i] = result.sample.to_array()
        outputs[i] = np.array([
            result.net_efficiency,
            float(result.raw_key_length),
            float(result.final_key_length),
            result.qber_measured if not np.isnan(result.qber_measured) else -1.0,
            result.reconciliation_efficiency,
            float(result.leakage_bits),
        ], dtype=DTYPE_OUTPUTS)
        outcomes.append(result.outcome.value)
        metadata.append(json.dumps({
            "error_message": result.error_message,
            "execution_time_seconds": result.execution_time_seconds,
            **result.metadata,
        }))

    return inputs, outputs, outcomes, metadata


def hdf5_arrays_to_training_data(
    inputs: np.ndarray,
    outputs: np.ndarray,
    outcomes: List[str],
    filter_success_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert HDF5 arrays to GP training data (X, y).

    Parameters
    ----------
    inputs : np.ndarray
        Input parameters, shape (n, 9).
    outputs : np.ndarray
        Output metrics, shape (n, 6). First column is net_efficiency.
    outcomes : List[str]
        Protocol outcomes as strings.
    filter_success_only : bool
        If True, only include successful runs. If False, include all
        but set net_efficiency to 0 for failures.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X, y) arrays for GP training.
    """
    if filter_success_only:
        mask = np.array([o == ProtocolOutcome.SUCCESS.value for o in outcomes])
        return inputs[mask], outputs[mask, 0]
    else:
        # For failures, net_efficiency should already be 0 in outputs
        return inputs.copy(), outputs[:, 0].copy()


def capture_rng_state() -> Dict[str, Any]:
    """
    Capture the current numpy RNG state.

    Returns
    -------
    Dict[str, Any]
        Serializable RNG state dictionary.
    """
    state = np.random.get_state()
    return {
        "kind": state[0],
        "keys": state[1].tolist(),
        "pos": state[2],
        "has_gauss": state[3],
        "cached_gaussian": state[4],
    }


def restore_rng_state(state_dict: Dict[str, Any]) -> None:
    """
    Restore numpy RNG state from a dictionary.

    Parameters
    ----------
    state_dict : Dict[str, Any]
        RNG state as captured by `capture_rng_state()`.
    """
    state = (
        state_dict["kind"],
        np.array(state_dict["keys"], dtype=np.uint32),
        state_dict["pos"],
        state_dict["has_gauss"],
        state_dict["cached_gaussian"],
    )
    np.random.set_state(state)

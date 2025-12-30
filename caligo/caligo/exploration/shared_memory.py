"""
Zero-Copy Shared Memory Arena for Inter-Process Communication.

This module implements a shared memory architecture that eliminates
serialization overhead when transferring quantum measurement data
between worker processes and the main orchestrator.

Architecture
------------
The SharedMemoryArena pre-allocates a contiguous memory block that
workers write directly into. Instead of returning full data arrays
through pipes (which requires pickling), workers return lightweight
metadata pointers indicating where their data resides in shared memory.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SharedMemoryArena                            │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐      │
│  │   Slot 0    │   Slot 1    │   Slot 2    │   Slot 3    │ ...  │
│  │ (Worker 0)  │ (Worker 1)  │ (Worker 2)  │ (Worker 3)  │      │
│  └─────────────┴─────────────┴─────────────┴─────────────┘      │
│                                                                 │
│  Each slot contains:                                            │
│  - alice_outcomes: Float32[max_pairs]                           │
│  - alice_bases:    Int8[max_pairs]                              │
│  - bob_outcomes:   Float32[max_pairs]                           │
│  - bob_bases:      Int8[max_pairs]                              │
│  - metadata:       Float32[16]                                  │
└─────────────────────────────────────────────────────────────────┘
```

Performance
-----------
- Zero serialization: Workers write directly to shared memory
- Zero copy: Main process creates NumPy views over shared buffer
- Constant memory: Arena size fixed at initialization (num_workers * slot_size)
- SIMD-friendly: All arrays are Float32/Int8 contiguous

Usage
-----
>>> arena = SharedMemoryArena(num_slots=8, max_pairs_per_slot=1_000_000)
>>> # Worker writes:
>>> slot = arena.acquire_slot()
>>> alice_out = arena.get_alice_outcomes_view(slot)
>>> alice_out[:n] = measurement_results  # Direct write
>>> arena.release_slot(slot, n_pairs=n)
>>> # Main process reads:
>>> alice_view = arena.get_alice_outcomes_view(slot)[:n]  # Zero-copy view
"""

from __future__ import annotations

import atexit
import os
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from caligo.exploration.types import DTYPE_FLOAT, Float32Array, DTYPE_INT, Int8Array
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Memory layout constants
METADATA_SIZE = 16  # Float32 values for metadata per slot
ALIGNMENT = 64  # Cache line alignment for SIMD efficiency


# =============================================================================
# Slot Metadata
# =============================================================================


@dataclass
class SlotMetadata:
    """
    Metadata describing the contents of a shared memory slot.

    Parameters
    ----------
    slot_id : int
        Slot index in the arena.
    n_pairs : int
        Number of valid EPR pairs in this slot.
    is_valid : bool
        Whether the slot contains valid data.
    generation_time_ns : int
        Generation time in nanoseconds.
    error_code : int
        Error code (0 = success).
    sample_id : int
        Original sample index for ordering.

    Attributes
    ----------
    slot_id : int
    n_pairs : int
    is_valid : bool
    generation_time_ns : int
    error_code : int
    sample_id : int
    """

    slot_id: int
    n_pairs: int = 0
    is_valid: bool = False
    generation_time_ns: int = 0
    error_code: int = 0
    sample_id: int = -1

    def to_array(self) -> Float32Array:
        """Serialize to Float32 array for shared memory."""
        return np.array([
            self.slot_id,
            self.n_pairs,
            1.0 if self.is_valid else 0.0,
            self.generation_time_ns / 1e9,  # Store as seconds
            self.error_code,
            self.sample_id,
            123456.0,  # Magic marker (approx 0x1E240, just a distinct float)
            0.0, 0.0, 0.0,  # Reserved
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Reserved
        ], dtype=DTYPE_FLOAT)

    @classmethod
    def from_array(cls, arr: Float32Array) -> "SlotMetadata":
        """Deserialize from Float32 array."""
        # Validate magic marker to catch offset/alignment issues
        magic = float(arr[6])
        valid = arr[2] > 0.5
        error_code = int(arr[4])

        if valid and abs(magic - 123456.0) > 0.1:
            # If marked valid but magic is missing, we are reading garbage/wrong offset
            # But don't crash, just mark invalid so we don't process garbage
            valid = False
            error_code = 999  # Internal code for alignment error

        return cls(
            slot_id=int(arr[0]),
            n_pairs=int(arr[1]),
            is_valid=valid,
            generation_time_ns=int(arr[3] * 1e9),
            error_code=error_code,
            sample_id=int(arr[5]),
        )


# =============================================================================
# Shared Memory Arena
# =============================================================================


class SharedMemoryArena:
    """
    Pre-allocated shared memory arena for zero-copy EPR data transfer.

    The arena divides a single shared memory block into fixed-size slots,
    one per worker. Each slot can hold EPR measurement data for up to
    `max_pairs_per_slot` pairs.

    Parameters
    ----------
    num_slots : int
        Number of slots (typically = num_workers).
    max_pairs_per_slot : int
        Maximum EPR pairs each slot can hold.
    name_prefix : str
        Prefix for shared memory segment names.

    Attributes
    ----------
    num_slots : int
        Number of slots in the arena.
    max_pairs_per_slot : int
        Maximum pairs per slot.
    slot_size_bytes : int
        Size of each slot in bytes.
    total_size_bytes : int
        Total arena size in bytes.

    Examples
    --------
    >>> arena = SharedMemoryArena(num_slots=4, max_pairs_per_slot=100000)
    >>> print(f"Arena size: {arena.total_size_bytes / 1e6:.1f} MB")
    >>> arena.cleanup()
    """

    def __init__(
        self,
        num_slots: int,
        max_pairs_per_slot: int,
        name_prefix: str = "caligo_epr",
    ) -> None:
        """
        Initialize the shared memory arena.

        Parameters
        ----------
        num_slots : int
            Number of slots (one per worker).
        max_pairs_per_slot : int
            Maximum EPR pairs per slot.
        name_prefix : str
            Prefix for shared memory segment names.
        """
        self.num_slots = num_slots
        self.max_pairs_per_slot = max_pairs_per_slot
        self._name_prefix = name_prefix

        # Calculate slot layout
        # Per slot: Use int8 for outcomes (bits), not Float32
        self._alice_out_size = max_pairs_per_slot * np.dtype(DTYPE_INT).itemsize
        self._alice_bases_size = max_pairs_per_slot * np.dtype(DTYPE_INT).itemsize
        self._bob_out_size = max_pairs_per_slot * np.dtype(DTYPE_INT).itemsize
        self._bob_bases_size = max_pairs_per_slot * np.dtype(DTYPE_INT).itemsize
        self._metadata_size = METADATA_SIZE * np.dtype(DTYPE_FLOAT).itemsize

        # Align slot size to cache line boundary
        raw_slot_size = (
            self._alice_out_size +
            self._alice_bases_size +
            self._bob_out_size +
            self._bob_bases_size +
            self._metadata_size
        )
        self.slot_size_bytes = ((raw_slot_size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        self.total_size_bytes = self.slot_size_bytes * num_slots

        # Calculate offsets within each slot
        self._alice_out_offset = 0
        self._alice_bases_offset = self._alice_out_size
        self._bob_out_offset = self._alice_bases_offset + self._alice_bases_size
        self._bob_bases_offset = self._bob_out_offset + self._bob_out_size
        self._metadata_offset = self._bob_bases_offset + self._bob_bases_size

        # Create shared memory with unique name
        self._shm_name = f"{name_prefix}_{os.getpid()}"
        try:
            # Try to unlink any existing segment with this name
            try:
                existing = shared_memory.SharedMemory(name=self._shm_name)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass

            self._shm = shared_memory.SharedMemory(
                name=self._shm_name,
                create=True,
                size=self.total_size_bytes,
            )
        except Exception as e:
            logger.error("Failed to create shared memory: %s", e)
            raise

        # Create NumPy buffer view
        self._buffer = np.ndarray(
            shape=(self.total_size_bytes,),
            dtype=np.uint8,
            buffer=self._shm.buf,
        )

        # Slot management
        self._slot_lock = threading.Lock()
        self._free_slots = list(range(num_slots))
        self._active_slots: Dict[int, int] = {}  # slot_id -> sample_id

        # Register cleanup on exit
        atexit.register(self.cleanup)

        logger.info(
            "Created SharedMemoryArena: %d slots × %.1f MB = %.1f MB total",
            num_slots,
            self.slot_size_bytes / 1e6,
            self.total_size_bytes / 1e6,
        )

    @property
    def shm_name(self) -> str:
        """Name of the shared memory segment."""
        return self._shm_name

    def _slot_base_offset(self, slot_id: int) -> int:
        """Get the byte offset for a slot's start."""
        return slot_id * self.slot_size_bytes

    def acquire_slot(self, sample_id: int = -1, timeout: float = 30.0) -> int:
        """
        Acquire an available slot for writing.

        Parameters
        ----------
        sample_id : int
            Sample index to associate with this slot.
        timeout : float
            Timeout in seconds (currently ignored, raises if none available).

        Returns
        -------
        int
            Slot ID.

        Raises
        ------
        RuntimeError
            If no slots are available.
        """
        with self._slot_lock:
            if not self._free_slots:
                raise RuntimeError(
                    f"No free slots available (all {self.num_slots} in use). "
                    "Increase num_slots or ensure slots are released."
                )
            slot_id = self._free_slots.pop(0)
            self._active_slots[slot_id] = sample_id
            return slot_id

    def release_slot(self, slot_id: int) -> None:
        """
        Release a slot back to the pool.

        Parameters
        ----------
        slot_id : int
            Slot ID to release.
        """
        with self._slot_lock:
            if slot_id in self._active_slots:
                del self._active_slots[slot_id]
            if slot_id not in self._free_slots:
                self._free_slots.append(slot_id)

    def get_alice_outcomes_view(self, slot_id: int) -> Int8Array:
        """
        Get a writeable view of Alice's outcome array for a slot.

        Parameters
        ----------
        slot_id : int
            Slot ID.

        Returns
        -------
        Int8Array
            Shape (max_pairs_per_slot,) Int8 array.
        """
        base = self._slot_base_offset(slot_id) + self._alice_out_offset
        return np.ndarray(
            shape=(self.max_pairs_per_slot,),
            dtype=DTYPE_INT,
            buffer=self._shm.buf,
            offset=base,
        )

    def get_alice_bases_view(self, slot_id: int) -> Int8Array:
        """
        Get a writeable view of Alice's bases array for a slot.

        Parameters
        ----------
        slot_id : int
            Slot ID.

        Returns
        -------
        Int8Array
            Shape (max_pairs_per_slot,) int8 array.
        """
        base = self._slot_base_offset(slot_id) + self._alice_bases_offset
        return np.ndarray(
            shape=(self.max_pairs_per_slot,),
            dtype=np.int8,
            buffer=self._shm.buf,
            offset=base,
        )

    def get_bob_outcomes_view(self, slot_id: int) -> Int8Array:
        """
        Get a writeable view of Bob's outcome array for a slot.

        Parameters
        ----------
        slot_id : int
            Slot ID.

        Returns
        -------
        Int8Array
            Shape (max_pairs_per_slot,) Int8 array.
        """
        base = self._slot_base_offset(slot_id) + self._bob_out_offset
        return np.ndarray(
            shape=(self.max_pairs_per_slot,),
            dtype=DTYPE_INT,
            buffer=self._shm.buf,
            offset=base,
        )

    def get_bob_bases_view(self, slot_id: int) -> Int8Array:
        """
        Get a writeable view of Bob's bases array for a slot.

        Parameters
        ----------
        slot_id : int
            Slot ID.

        Returns
        -------
        Int8Array
            Shape (max_pairs_per_slot,) int8 array.
        """
        base = self._slot_base_offset(slot_id) + self._bob_bases_offset
        return np.ndarray(
            shape=(self.max_pairs_per_slot,),
            dtype=np.int8,
            buffer=self._shm.buf,
            offset=base,
        )

    def get_metadata_view(self, slot_id: int) -> Float32Array:
        """
        Get a view of the metadata array for a slot.

        Parameters
        ----------
        slot_id : int
            Slot ID.

        Returns
        -------
        Float32Array
            Shape (METADATA_SIZE,) Float32 array.
        """
        base = self._slot_base_offset(slot_id) + self._metadata_offset
        return np.ndarray(
            shape=(METADATA_SIZE,),
            dtype=DTYPE_FLOAT,
            buffer=self._shm.buf,
            offset=base,
        )

    def write_metadata(self, slot_id: int, metadata: SlotMetadata) -> None:
        """
        Write metadata to a slot.

        Parameters
        ----------
        slot_id : int
            Slot ID.
        metadata : SlotMetadata
            Metadata to write.
        """
        view = self.get_metadata_view(slot_id)
        view[:] = metadata.to_array()

    def read_metadata(self, slot_id: int) -> SlotMetadata:
        """
        Read metadata from a slot.

        Parameters
        ----------
        slot_id : int
            Slot ID.

        Returns
        -------
        SlotMetadata
            Slot metadata.
        """
        view = self.get_metadata_view(slot_id)
        return SlotMetadata.from_array(view.copy())

    def cleanup(self) -> None:
        """
        Clean up shared memory resources.

        This method should be called when the arena is no longer needed.
        It closes and unlinks the shared memory segment.
        """
        try:
            if hasattr(self, '_shm') and self._shm is not None:
                self._shm.close()
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass  # Already unlinked
                self._shm = None
                logger.debug("Cleaned up SharedMemoryArena: %s", self._shm_name)
        except Exception as e:
            logger.warning("Error during SharedMemoryArena cleanup: %s", e)

    def __del__(self) -> None:
        """Destructor - ensure cleanup on garbage collection."""
        self.cleanup()


# =============================================================================
# Worker-Side Shared Memory Attachment
# =============================================================================


class SharedMemorySlotWriter:
    """
    Worker-side interface for writing to a shared memory slot.

    This class is used by worker processes to attach to an existing
    shared memory segment and write EPR data directly.

    Parameters
    ----------
    shm_name : str
        Name of the shared memory segment.
    slot_id : int
        Slot ID assigned to this worker.
    max_pairs : int
        Maximum pairs per slot.
    slot_size_bytes : int
        Total size of each slot in bytes.

    Examples
    --------
    >>> # In worker process:
    >>> writer = SharedMemorySlotWriter(
    ...     shm_name="caligo_epr_12345",
    ...     slot_id=0,
    ...     max_pairs=100000,
    ...     slot_size_bytes=1024000,
    ... )
    >>> writer.write_alice_outcomes(measurements)
    >>> writer.finalize(n_pairs=50000, success=True)
    >>> writer.close()
    """

    def __init__(
        self,
        shm_name: str,
        slot_id: int,
        max_pairs: int,
        slot_size_bytes: int,
    ) -> None:
        """
        Initialize the slot writer by attaching to existing shared memory.

        Parameters
        ----------
        shm_name : str
            Name of the shared memory segment.
        slot_id : int
            Slot ID for this worker.
        max_pairs : int
            Maximum pairs per slot.
        slot_size_bytes : int
            Slot size in bytes.
        """
        self.slot_id = slot_id
        self.max_pairs = max_pairs
        self.slot_size_bytes = slot_size_bytes

        # Calculate offsets (must match arena layout)
        alice_out_size = max_pairs * np.dtype(DTYPE_INT).itemsize
        alice_bases_size = max_pairs * np.dtype(DTYPE_INT).itemsize
        bob_out_size = max_pairs * np.dtype(DTYPE_INT).itemsize
        bob_bases_size = max_pairs * np.dtype(DTYPE_INT).itemsize

        self._alice_out_offset = 0
        self._alice_bases_offset = alice_out_size
        self._bob_out_offset = self._alice_bases_offset + alice_bases_size
        self._bob_bases_offset = self._bob_out_offset + bob_out_size
        self._metadata_offset = self._bob_bases_offset + bob_bases_size

        # Attach to existing shared memory
        self._shm = shared_memory.SharedMemory(name=shm_name)
        self._base_offset = slot_id * slot_size_bytes

    def _get_view(
        self,
        offset: int,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        """Get a view of a portion of the slot."""
        return np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=self._shm.buf,
            offset=self._base_offset + offset,
        )

    def get_alice_outcomes_view(self) -> Int8Array:
        """Get Alice outcomes array view."""
        return self._get_view(
            self._alice_out_offset,
            (self.max_pairs,),
            DTYPE_INT,
        )

    def get_alice_bases_view(self) -> Int8Array:
        """Get Alice bases array view."""
        return self._get_view(
            self._alice_bases_offset,
            (self.max_pairs,),
            DTYPE_INT,
        )

    def get_bob_outcomes_view(self) -> Int8Array:
        """Get Bob outcomes array view."""
        return self._get_view(
            self._bob_out_offset,
            (self.max_pairs,),
            DTYPE_INT,
        )

    def get_bob_bases_view(self) -> Int8Array:
        """Get Bob bases array view."""
        return self._get_view(
            self._bob_bases_offset,
            (self.max_pairs,),
            DTYPE_INT,
        )

    def write_epr_data(
        self,
        alice_outcomes: np.ndarray,
        alice_bases: np.ndarray,
        bob_outcomes: np.ndarray,
        bob_bases: np.ndarray,
    ) -> int:
        """
        Write EPR data to the slot.

        Parameters
        ----------
        alice_outcomes : np.ndarray
            Alice's measurement outcomes.
        alice_bases : np.ndarray
            Alice's measurement bases.
        bob_outcomes : np.ndarray
            Bob's measurement outcomes.
        bob_bases : np.ndarray
            Bob's measurement bases.

        Returns
        -------
        int
            Number of pairs written.
        """
        n_pairs = len(alice_outcomes)
        if n_pairs > self.max_pairs:
            raise ValueError(
                f"Too many pairs ({n_pairs}) for slot capacity ({self.max_pairs})"
            )

        # Write directly to shared memory views
        # Use np.asarray to safely handle both lists and numpy arrays
        alice_out = self.get_alice_outcomes_view()
        alice_out[:n_pairs] = np.asarray(alice_outcomes, dtype=DTYPE_INT)

        alice_bases_view = self.get_alice_bases_view()
        alice_bases_view[:n_pairs] = np.asarray(alice_bases, dtype=DTYPE_INT)

        bob_out = self.get_bob_outcomes_view()
        bob_out[:n_pairs] = np.asarray(bob_outcomes, dtype=DTYPE_INT)

        bob_bases_view = self.get_bob_bases_view()
        bob_bases_view[:n_pairs] = np.asarray(bob_bases, dtype=DTYPE_INT)

        return n_pairs

    def write_metadata(self, metadata: SlotMetadata) -> None:
        """Write metadata to the slot."""
        view = self._get_view(
            self._metadata_offset,
            (METADATA_SIZE,),
            DTYPE_FLOAT,
        )
        view[:] = metadata.to_array()

    def close(self) -> None:
        """Close the shared memory attachment (does not unlink)."""
        if hasattr(self, '_shm') and self._shm is not None:
            self._shm.close()
            self._shm = None


# =============================================================================
# Zero-Copy EPR Data Access
# =============================================================================


@dataclass
class ZeroCopyEPRData:
    """
    Zero-copy view into shared memory EPR data.

    This dataclass provides views into the shared memory arena without
    copying data. The views remain valid as long as the arena exists
    and the slot is not released.

    Parameters
    ----------
    alice_outcomes : Float32Array
        View of Alice's outcomes.
    alice_bases : NDArray[np.int8]
        View of Alice's bases.
    bob_outcomes : Float32Array
        View of Bob's outcomes.
    bob_bases : NDArray[np.int8]
        View of Bob's bases.
    n_pairs : int
        Number of valid pairs.
    slot_id : int
        Source slot ID.

    Notes
    -----
    Use `.copy()` if you need to retain data after slot release.
    """

    alice_outcomes: Int8Array
    alice_bases: Int8Array
    bob_outcomes: Int8Array
    bob_bases: Int8Array
    n_pairs: int
    slot_id: int

    def copy(self) -> "ZeroCopyEPRData":
        """Create a deep copy of the data (no longer zero-copy)."""
        return ZeroCopyEPRData(
            alice_outcomes=self.alice_outcomes[:self.n_pairs].copy(),
            alice_bases=self.alice_bases[:self.n_pairs].copy(),
            bob_outcomes=self.bob_outcomes[:self.n_pairs].copy(),
            bob_bases=self.bob_bases[:self.n_pairs].copy(),
            n_pairs=self.n_pairs,
            slot_id=self.slot_id,
        )


def create_zero_copy_epr_data(
    arena: SharedMemoryArena,
    slot_id: int,
    n_pairs: int,
) -> ZeroCopyEPRData:
    """
    Create a zero-copy EPR data view from an arena slot.

    Parameters
    ----------
    arena : SharedMemoryArena
        Source arena.
    slot_id : int
        Slot containing the data.
    n_pairs : int
        Number of valid pairs.

    Returns
    -------
    ZeroCopyEPRData
        Zero-copy views into the shared memory.
    """
    return ZeroCopyEPRData(
        alice_outcomes=arena.get_alice_outcomes_view(slot_id)[:n_pairs],
        alice_bases=arena.get_alice_bases_view(slot_id)[:n_pairs],
        bob_outcomes=arena.get_bob_outcomes_view(slot_id)[:n_pairs],
        bob_bases=arena.get_bob_bases_view(slot_id)[:n_pairs],
        n_pairs=n_pairs,
        slot_id=slot_id,
    )


# =============================================================================
# Shared Memory Cleanup Utilities
# =============================================================================


def cleanup_shared_memory_orphans(prefix: str = "caligo_epr") -> int:
    """
    Clean up orphaned shared memory segments from previous runs.

    This function scans for shared memory segments with names matching
    the given prefix and unlinks them. Use after abnormal termination
    to prevent resource leaks.

    Parameters
    ----------
    prefix : str
        Prefix to match for cleanup. Default: "caligo_epr".

    Returns
    -------
    int
        Number of segments cleaned up.

    Notes
    -----
    On Linux, shared memory segments are in /dev/shm. This function
    attempts to clean up any matching segments, logging warnings for
    segments that can't be removed (e.g., still in use by another process).

    Examples
    --------
    >>> # Clean up after crash
    >>> n_cleaned = cleanup_shared_memory_orphans()
    >>> print(f"Cleaned up {n_cleaned} orphaned segments")
    """
    cleaned = 0

    # Try /dev/shm on Linux
    shm_path = "/dev/shm"
    if os.path.isdir(shm_path):
        try:
            for name in os.listdir(shm_path):
                if name.startswith(prefix):
                    try:
                        shm = shared_memory.SharedMemory(name=name)
                        shm.close()
                        shm.unlink()
                        cleaned += 1
                        logger.debug("Cleaned up orphaned segment: %s", name)
                    except FileNotFoundError:
                        pass  # Already gone
                    except PermissionError:
                        logger.warning(
                            "Cannot clean segment %s (permission denied, may be in use)", name
                        )
                    except Exception as e:
                        logger.warning("Error cleaning segment %s: %s", name, e)
        except PermissionError:
            logger.debug("Cannot list /dev/shm (permission denied)")
        except Exception as e:
            logger.debug("Error scanning /dev/shm: %s", e)

    if cleaned > 0:
        logger.info("Cleaned up %d orphaned shared memory segments", cleaned)

    return cleaned

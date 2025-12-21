# 4.3 Batching Strategies

## Overview

Batching manages memory and computational resources during EPR generation by processing pairs in fixed-size chunks.

## BatchingManager

```python
class BatchingManager:
    def __init__(self, batch_size: int = 1000):
        self._batch_size = batch_size
        self._current_batch = []
    
    def add(self, measurement_record: MeasurementRecord) -> Optional[BatchResult]:
        self._current_batch.append(measurement_record)
        if len(self._current_batch) >= self._batch_size:
            return self.flush()
        return None
    
    def flush(self) -> BatchResult:
        result = BatchResult(records=self._current_batch)
        self._current_batch = []
        return result
```

## Configuration

- **Small batches** (100-500): Lower memory, more frequent I/O
- **Large batches** (5000-10000): Higher memory, fewer flushes
- **Default**: 1000 pairs (balanced)

## References

- Caligo implementation: [`caligo/quantum/batching.py`](../../caligo/caligo/quantum/batching.py)

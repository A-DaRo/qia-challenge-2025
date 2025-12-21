[← Return to Main Index](../index.md)

# 10.2 Performance Metrics

## Introduction

This section presents **quantitative performance analysis** of Caligo's implementation, focusing on computational efficiency, memory usage, and scalability. Metrics are derived from benchmark tests (`tests/performance/`) and profiling tools (cProfile, memory_profiler).

## LDPC Decoding Performance

### Throughput Benchmarks

**Test**: `test_ldpc_decode_benchmark.py`

| Code Parameters | Decode Time | Throughput | Speedup (vs Python) |
|----------------|-------------|------------|-------------------|
| n=10k, rate=0.5, BP-40 | 1.2 ms | 8.3 Mbps | 104× |
| n=50k, rate=0.5, BP-40 | 6.1 ms | 8.2 Mbps | 98× |
| n=100k, rate=0.5, BP-40 | 12.3 ms | 8.1 Mbps | 95× |

**Observation**: Near-linear scaling with code length (cache effects minimal for $n < 10^6$).

## Parallel EPR Generation

### Speedup Analysis

**Test**: `test_parallel_speedup.py`

| Workers | Time (100k pairs) | Speedup | Efficiency |
|---------|------------------|---------|------------|
| 1 | 8.2 s | 1.0× | 100% |
| 2 | 4.3 s | 1.9× | 95% |
| 4 | 2.2 s | 3.7× | 93% |
| 8 | 1.2 s | 6.8× | 85% |

**Bottleneck**: GIL contention in NetSquid C extensions (85% efficiency at 8 cores).

## Memory Profiling

### Peak Memory Usage

| Phase | Memory (MB) | Per-Qubit |
|-------|------------|-----------|
| Quantum (100k EPR) | 42 | 0.42 KB |
| Sifting | 18 | 0.18 KB |
| Reconciliation | 156 | 1.56 KB |
| Amplification | 12 | 0.12 KB |

**Total**: ~230 MB for 100k qubits (reconciliation dominates due to LDPC message arrays).

---

[← Return to Main Index](../index.md) | [Next: QBER Analysis](./qber_analysis.md)

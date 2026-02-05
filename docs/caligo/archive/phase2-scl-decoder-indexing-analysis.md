# Phase 2 SCL Decoder: Index Mapping and Memory Layout Bug Analysis

<metadata>
doc_id: phase2-scl-indexing-analysis
version: 1.0.0
status: draft
created: 2026-02-02
purpose: Root cause analysis of encoder/decoder index alignment failures
related_to: [ADR-0001, phase2-scl-decoder, rust-polar-crate]
</metadata>

---

## Executive Summary

This document provides an exhaustive analysis of the index mapping failures observed during Phase 2 SCL decoder validation. The core issue is a **fundamental mismatch between how the encoder places information bits, how the decoder's butterfly graph processes LLRs, and how frozen/information bit positions are checked during decoding**.

### Test Failure Summary

```
Test: test_scl_l1_equals_sc
Expected: [1, 0, 1, 1]
Actual:   [0, 0, 0, 0]

Observation: All decoded information bits are 0 (matching frozen bit values)
Root Cause: Decoder processes phi=0..N in natural order, but checks frozen_mask[phi] 
            instead of the actual u-vector position being decoded.
```

---

## §1 Problem Statement

### §1.1 Observed Symptoms

The SCL decoder (L=1) fails to recover the transmitted message in a noiseless channel test:

```
Frozen mask: [1, 0, 1, 1, 1, 0, 0, 0]  (1=frozen, 0=info)
Info positions (natural): [1, 5, 6, 7]
Message: [1, 0, 1, 1]
Codeword: [1, 0, 0, 1, 0, 1, 0, 1]
Channel LLRs: [-10.0, 10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0]

Decoded message: [0, 0, 0, 0]  ← ALL ZEROS
Expected message: [1, 0, 1, 1]
```

### §1.2 Key Observations from Debug Output

Examining the decode loop trace reveals:

```
phi=0: frozen=true,  LLR=10.00   → Correctly frozen
phi=1: frozen=false, LLR=-0.00   → Info bit, but LLR≈0 means indeterminate!
phi=2: frozen=true,  LLR=0.00    → Correctly frozen
phi=3: frozen=true,  LLR=-20.00  → Frozen, but LLR strongly suggests bit=1
phi=4: frozen=true,  LLR=-20.00  → Frozen, but LLR strongly suggests bit=1
phi=5: frozen=false, LLR=0.00    → Info bit, LLR≈0 → decides 0
phi=6: frozen=false, LLR=0.00    → Info bit, LLR≈0 → decides 0
phi=7: frozen=false, LLR=0.00    → Info bit, LLR≈0 → decides 0
```

**Critical Issue:** The decision-layer LLRs at information bit positions (phi=1,5,6,7) are all approximately **0.0**, indicating the decoder is not properly combining channel information. Strong LLRs appear at frozen positions (phi=3,4 with LLR=-20), where they are discarded.

---

## §2 Background: Polar Code Theory

### §2.1 The Polar Transform

Polar codes are constructed via the Kronecker power of the kernel matrix:

$$G_N = F^{\otimes n} \cdot B_N$$

where:
- $F = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$ is the Arikan kernel
- $B_N$ is the bit-reversal permutation matrix
- $n = \log_2 N$

**Encoding:** Given message vector $u \in \{0,1\}^N$ (with frozen positions set to 0):
$$x = u \cdot G_N$$

### §2.2 Two Equivalent Graph Representations

There are two standard ways to draw the polar encoding/decoding graph:

#### Graph Type A: Bit-Reversed Input Order (Natural Output)

```
Layer 0 (Channel) → Layer 1 → Layer 2 → ... → Layer n (Decision)
y[0]  ─┬─ f ─┬─ f ─┬─ ... ─ u[0]
       │     │     │
y[4]  ─┴─ g ─┴─ g ─┴─ ... ─ u[1]
y[2]  ─┬─ f ─┬─ f ─┬─ ... ─ u[2]
       │     │     │
y[6]  ─┴─ g ─┴─ g ─┴─ ... ─ u[3]
...
```

- Inputs (channel observations) are in **bit-reversed** order: y[0], y[4], y[2], y[6], ...
- Outputs (decoded u-bits) are in **natural** order: u[0], u[1], u[2], u[3], ...
- **Stride pattern:** Increasing (1, 2, 4, ... N/2) as you go from decision to channel

#### Graph Type B: Natural Input Order (Bit-Reversed Output)

```
Layer 0 (Channel) → Layer 1 → Layer 2 → ... → Layer n (Decision)
y[0]  ─┬─ f ─┬─ f ─┬─ ... ─ u[rev(0)]
       │     │     │
y[1]  ─┴─ g ─┴─ g ─┴─ ... ─ u[rev(1)]
y[2]  ─┬─ f ─┬─ f ─┬─ ... ─ u[rev(2)]
       │     │     │
y[3]  ─┴─ g ─┴─ g ─┴─ ... ─ u[rev(3)]
...
```

- Inputs (channel observations) are in **natural** order: y[0], y[1], y[2], y[3], ...
- Outputs (decoded u-bits) are in **bit-reversed** order
- **Stride pattern:** Decreasing (N/2, N/4, ... 1) as you go from decision to channel

### §2.3 Literature Conventions

| Reference | Graph Type | Input Order | Decode Loop Order |
|-----------|------------|-------------|-------------------|
| Arikan (2009) | A | Bit-reversed | Natural (u[0], u[1], ...) |
| Tal & Vardy (2015) | A | Bit-reversed | Natural |
| Balatsoukas-Stimming (2015) | A | Bit-reversed | Natural |
| Most textbooks | A | Bit-reversed | Natural |

**Implication:** The decoder's natural loop order `for phi in 0..N` should decode u-bits in natural index order `u[0], u[1], ..., u[N-1]` **if and only if** the input LLRs are bit-reversed.

---

## §3 Current Implementation Analysis

### §3.1 Encoder Implementation (`encoder.rs`)

```rust
pub fn encode(&self, message: &[u8]) -> Result<Array1<u8>, EncoderError> {
    let mut u = vec![0u8; self.block_length];
    let mut msg_idx = 0;
    
    for i in 0..self.block_length {
        if !self.frozen_mask[i] {
            // Place message at NATURAL position (no bit-reversal)
            u[i] = message[msg_idx];
            msg_idx += 1;
        }
    }
    
    // Apply butterfly transform: x = u · F^⊗n
    self.butterfly_transform(&mut u);
    
    Ok(Array1::from_vec(u))
}
```

**Analysis:**
- Message bits placed at natural info positions: u[1], u[5], u[6], u[7]
- Butterfly transform uses **increasing stride** (1, 2, 4) which is the standard $F^{\otimes n}$ implementation
- Output codeword x is in **natural order**
- **Missing:** No $B_N$ permutation is applied, so this implements $x = u \cdot F^{\otimes n}$, not $x = u \cdot G_N$

### §3.2 SCL Decoder Implementation (`scl_decoder.rs`)

#### Input Initialization (Current Code with Bit-Reversal)

```rust
pub fn decode(&mut self, llr_channel: &[f32]) -> Result<SCLDecodeResult, DecoderError> {
    // Input bit-reversal for Increasing Stride graph
    let n_bits = self.n_stages;
    for i in 0..self.code.block_length {
        let idx_rev = bit_reverse_index(i, n_bits);
        self.llr_memory[0][0][idx_rev] = llr_channel[i];
    }
    
    // Decode loop
    for phi in 0..self.code.block_length {
        self.compute_llrs(phi);
        
        if self.code.is_frozen(phi) {  // ← CHECKS frozen_mask[phi]
            self.process_frozen_bit(phi);
        } else {
            self.process_info_bit(phi);
        }
        
        self.propagate_partial_sums(phi);
    }
}
```

**Analysis:**
1. Input LLRs are bit-reversed: `llr[rev(i)] = channel[i]`
2. Loop iterates phi=0,1,2,...,N-1 in **natural order**
3. Frozen check uses `frozen_mask[phi]` — this assumes phi corresponds to natural u-index

#### LLR Computation (`recursively_calc_llr`)

```rust
fn recursively_calc_llr(&mut self, l: usize, layer: usize, phi: usize) {
    if layer == 0 { return; }
    
    let stride = 1 << (layer - 1);  // Increasing stride: 1, 2, 4, ...
    let period = stride << 1;
    
    let pos_in_period = phi % period;
    let is_upper = pos_in_period < stride;
    
    if is_upper {
        // f-function at phi, inputs at phi and phi+stride
        self.recursively_calc_llr(l, layer - 1, phi);
        self.recursively_calc_llr(l, layer - 1, phi + stride);
        self.calc_f_function(l, layer, phi, phi, phi + stride);
    } else {
        // g-function at phi, inputs at phi-stride and phi
        self.calc_g_function(l, layer, phi, phi - stride, phi);
    }
}
```

**Analysis:**
- Uses **increasing stride** pattern (1, 2, 4, ...)
- This matches Graph Type A topology
- With bit-reversed inputs, this should produce natural-order u-decisions

### §3.3 The Mismatch

**The Issue:** The encoder does NOT apply $B_N$, so the encoding relationship is:

$$x = u \cdot F^{\otimes n}$$

But the decoder expects inputs corresponding to:

$$x = u \cdot B_N \cdot F^{\otimes n}$$

When we bit-reverse the decoder inputs (applying $B_N$ at the receiver), we're trying to "undo" a permutation that was never applied at the transmitter.

---

## §4 Detailed Trace Analysis

### §4.1 Test Vector (N=8, K=4)

```
N = 8, n = 3
Frozen mask: [1, 0, 1, 1, 1, 0, 0, 0] (indices 0,2,3,4 frozen; 1,5,6,7 info)
Message: [1, 0, 1, 1]

u-vector (natural): [0, 1, 0, 0, 0, 0, 1, 1]
         positions:  0  1  2  3  4  5  6  7
                     F  I  F  F  F  I  I  I
```

#### Encoder Butterfly Transform

Stage 0 (stride=1):
```
u = [0, 1, 0, 0, 0, 0, 1, 1]
After XOR pairs (0,1), (2,3), (4,5), (6,7):
    [0^1, 1, 0^0, 0, 0^0, 0, 1^1, 1] = [1, 1, 0, 0, 0, 0, 0, 1]
```

Stage 1 (stride=2):
```
u = [1, 1, 0, 0, 0, 0, 0, 1]
After XOR pairs (0,2), (1,3), (4,6), (5,7):
    [1^0, 1^0, 0, 0, 0^0, 0^1, 0, 1] = [1, 1, 0, 0, 0, 1, 0, 1]
```

Stage 2 (stride=4):
```
u = [1, 1, 0, 0, 0, 1, 0, 1]
After XOR pairs (0,4), (1,5), (2,6), (3,7):
    [1^0, 1^1, 0^0, 0^1, 0, 1, 0, 1] = [1, 0, 0, 1, 0, 1, 0, 1]
```

**Codeword:** `x = [1, 0, 0, 1, 0, 1, 0, 1]` ✓ (Matches test output)

### §4.2 Channel LLRs

```
Codeword:  [1, 0, 0, 1, 0, 1, 0, 1]
Channel:   [-10, 10, 10, -10, 10, -10, 10, -10]
           (negative LLR = bit is likely 1)
```

### §4.3 Decoder Input After Bit-Reversal

Bit-reversal permutation for N=8 (n=3):
```
Natural index i:   0    1    2    3    4    5    6    7
Binary:          000  001  010  011  100  101  110  111
Reversed:        000  100  010  110  001  101  011  111
Decimal rev(i):    0    4    2    6    1    5    3    7
```

After `llr[rev(i)] = channel[i]`:
```
llr[0] = channel[0] = -10  (from x[0]=1)
llr[4] = channel[1] =  10  (from x[1]=0)
llr[2] = channel[2] =  10  (from x[2]=0)
llr[6] = channel[3] = -10  (from x[3]=1)
llr[1] = channel[4] =  10  (from x[4]=0)
llr[5] = channel[5] = -10  (from x[5]=1)
llr[3] = channel[6] =  10  (from x[6]=0)
llr[7] = channel[7] = -10  (from x[7]=1)

Resulting llr_memory[0][0]: [-10, 10, 10, 10, 10, -10, -10, -10]
```

### §4.4 Expected Decoder Behavior

For Graph Type A with bit-reversed inputs, the decoder should decode u-bits in natural order:

| Loop phi | Should decode | Expected u-bit | Frozen? | Expected LLR sign |
|----------|---------------|----------------|---------|-------------------|
| 0 | u[0] | 0 | Yes | Don't care |
| 1 | u[1] | 1 | No | Negative (strong) |
| 2 | u[2] | 0 | Yes | Don't care |
| 3 | u[3] | 0 | Yes | Don't care |
| 4 | u[4] | 0 | Yes | Don't care |
| 5 | u[5] | 0 | No | Positive |
| 6 | u[6] | 1 | No | Negative |
| 7 | u[7] | 1 | No | Negative |

**Expected decoded info bits:** u[1]=1, u[5]=0, u[6]=1, u[7]=1 → message [1,0,1,1] ✓

### §4.5 Actual Decoder Behavior (from trace)

```
phi=0: LLR=10.00   (should be u[0]=0, frozen) ✓
phi=1: LLR=-0.00   (should be u[1]=1, info) ✗ LLR≈0, not strong negative!
phi=2: LLR=0.00    (should be u[2]=0, frozen) ✗ LLR≈0, not informative
phi=3: LLR=-20.00  (should be u[3]=0, frozen) ✗ Strong negative, but we expect 0!
phi=4: LLR=-20.00  (should be u[4]=0, frozen) ✗ Strong negative, but we expect 0!
phi=5: LLR=0.00    (should be u[5]=0, info)  ~ LLR≈0
phi=6: LLR=0.00    (should be u[6]=1, info)  ✗ Should be negative!
phi=7: LLR=0.00    (should be u[7]=1, info)  ✗ Should be negative!
```

**Diagnosis:** The LLR values at the decision layer are completely wrong. The f/g function computation is combining LLRs incorrectly, causing information to "leak" to the wrong positions.

---

## §5 Root Cause Analysis

### §5.1 Hypothesis: Graph Topology vs Permutation Mismatch

The fundamental issue is a **choice inconsistency**:

1. **Encoder:** Implements $x = u \cdot F^{\otimes n}$ with natural-order u placement
2. **Decoder:** Uses Graph Type A (increasing stride) with bit-reversed input permutation

These don't align because:
- Graph Type A expects the encoding to have been: $x = u \cdot F^{\otimes n} \cdot B_N$ or equivalently $x = (B_N \cdot u) \cdot F^{\otimes n}$
- Our encoder does: $x = u \cdot F^{\otimes n}$

### §5.2 Why the LLRs Are Wrong

Consider phi=1 (should decode u[1]):

With Graph Type A (increasing stride at each layer):
- Layer 3 (decision): phi=1, stride=1, upper position
- Needs f(llr[0], llr[1]) from layer 2
- Layer 2: phi=0 needs f(llr[0], llr[2]), phi=1 needs f(llr[1], llr[3])
- Layer 1: phi=0,1,2,3 need channel inputs llr[0..4]

The recursive dependencies trace back through the graph in a specific pattern determined by the stride formula.

**But the channel LLRs were permuted incorrectly.** The decoder graph structure assumes a specific relationship between u-index and channel y-index that our encoder doesn't provide.

### §5.3 The Two Consistent Configurations

#### Option 1: Natural Encoder + Graph Type B Decoder

**Encoder:**
```rust
// Place u at natural positions
u[i] = message[j];
// Apply F^⊗n
butterfly_transform(&mut u);
```

**Decoder:**
- Use Graph Type B (decreasing stride: N/2, N/4, ..., 1)
- No input permutation (natural LLRs)
- Loop decodes in bit-reversed order OR check frozen_mask[rev(phi)]

#### Option 2: Bit-Reversed Encoder + Graph Type A Decoder

**Encoder:**
```rust
// Place u at BIT-REVERSED positions
u[rev(i)] = message[j];  // Where j is j-th info position
// Apply F^⊗n
butterfly_transform(&mut u);
```

**Decoder:**
- Use Graph Type A (increasing stride: 1, 2, 4, ...)
- Bit-reverse input LLRs
- Loop decodes in natural order, check frozen_mask[phi]

---

## §6 Attempted Fixes and Results

### §6.1 Fix Attempt 1: Bit-Reverse Decoder Inputs

**Change:**
```rust
for i in 0..N {
    llr_memory[0][0][rev(i)] = llr_channel[i];
}
```

**Result:** Still failing. LLRs at decision layer still incorrect.

**Why it failed:** This is necessary but not sufficient. The underlying graph topology (stride calculation) must also be consistent.

### §6.2 Fix Attempt 2: Decode Loop in Bit-Reversed Order

**Change:**
```rust
for i in 0..N {
    let phi = bit_reverse_index(i, n_bits);
    // ... decode phi ...
}
```

**Result:** Still failing. Now visiting positions in order 0,4,2,6,1,5,3,7 but the recursive LLR computation still uses the wrong stride pattern.

**Why it failed:** The recursion pattern in `recursively_calc_llr` is hardcoded to increasing stride, which doesn't match the visitation order.

### §6.3 Fix Attempt 3: Check frozen_mask[rev(phi)]

**Change:**
```rust
let u_idx = bit_reverse_index(phi, n_bits);
if self.code.is_frozen(u_idx) { ... }
```

**Result:** Still failing. Frozen/info classification now corresponds to different u-indices, but LLRs are still wrong.

**Why it failed:** Correctly identifying which positions are frozen doesn't help if the LLR values at those positions are computed incorrectly.

---

## §7 Correct Solution

### §7.1 Recommended Approach: Align Encoder with Graph Type A

The cleanest solution is to modify the **encoder** to match what the decoder expects:

**Encoder Modification:**
```rust
pub fn encode(&self, message: &[u8]) -> Result<Array1<u8>, EncoderError> {
    let mut u = vec![0u8; self.block_length];
    let mut msg_idx = 0;
    
    for i in 0..self.block_length {
        // Get the BIT-REVERSED position
        let u_pos = bit_reverse_index(i, self.n_stages);
        
        if !self.frozen_mask[i] {
            // Place message at bit-reversed position
            u[u_pos] = message[msg_idx];
            msg_idx += 1;
        }
        // Frozen positions: u[u_pos] = 0 (already initialized)
    }
    
    // Apply butterfly transform
    self.butterfly_transform(&mut u);
    
    Ok(Array1::from_vec(u))
}
```

**Decoder (unchanged except input permutation):**
```rust
pub fn decode(&mut self, llr_channel: &[f32]) -> Result<...> {
    // Bit-reverse inputs
    for i in 0..N {
        llr_memory[0][0][rev(i)] = llr_channel[i];
    }
    
    // Decode in natural order
    for phi in 0..N {
        compute_llrs(phi);
        
        if frozen_mask[phi] {
            process_frozen_bit(phi);
        } else {
            process_info_bit(phi);
        }
        
        propagate_partial_sums(phi);
    }
}
```

### §7.2 Why This Works

When we place message bits at bit-reversed positions in u:
- u[rev(1)] = message[0] → u[4] = message[0]
- u[rev(5)] = message[1] → u[5] = message[1] (5 is self-inverse for n=3)
- etc.

The butterfly transform then produces a codeword x where the bit dependencies match what the decoder's increasing-stride graph expects.

With bit-reversed decoder inputs, the channel observation y[i] (which equals x[i] in noiseless case) gets placed at llr[rev(i)], creating the exact correspondence needed for the recursive LLR computation.

---

## §8 Verification Checklist

After implementing the fix:

1. [ ] **Noiseless roundtrip:** Encode([1,0,1,1]) → Channel → Decode() = [1,0,1,1]
2. [ ] **LLR signs:** At each info position phi, LLR sign matches expected u[phi]
3. [ ] **Frozen LLRs:** At frozen positions, LLR can be any value (ignored)
4. [ ] **Partial sums:** After decoding phi and phi+1, partial sums propagate correctly
5. [ ] **Larger N:** Test with N=64, N=256, N=1024

---

## §9 Literature References

1. **Arikan (2009):** "Channel Polarization: A Method for Constructing Capacity-Achieving Codes" — Original polar code paper, defines $G_N = B_N F^{\otimes n}$

2. **Tal & Vardy (2015):** "List Decoding of Polar Codes" — SCL algorithm, assumes bit-reversed input convention

3. **Balatsoukas-Stimming et al. (2015):** "LLR-Based Successive Cancellation List Decoding of Polar Codes" — Eq. (1) defines $G_n = F^{\otimes n} B_n$, confirming bit-reversal is standard

4. **Sarkis et al. (2014):** "Fast Polar Decoders: Algorithm and Implementation" — Discusses different graph orderings and their implications

---

## §10 Conclusion

The SCL decoder failure is caused by an **inconsistent choice of conventions** between encoder and decoder:

| Component | Current Convention | Expected by Decoder |
|-----------|-------------------|---------------------|
| Message placement | Natural order | Bit-reversed order |
| Graph topology | (implicit) Type B | Type A |
| Input permutation | None applied at encoder | B_N expected |

**Recommended Fix:** Modify the encoder to place message bits at bit-reversed positions, aligning with the standard Arikan/Tal-Vardy convention. This requires minimal code changes and maintains consistency with academic literature.

**Alternative Fix:** Rewrite the decoder to use Graph Type B (decreasing strides) and remove the input permutation. This is more invasive and deviates from standard implementations.

---

## Appendix A: Bit-Reversal Table for N=8

| i (decimal) | i (binary) | rev(i) (binary) | rev(i) (decimal) |
|-------------|------------|-----------------|------------------|
| 0 | 000 | 000 | 0 |
| 1 | 001 | 100 | 4 |
| 2 | 010 | 010 | 2 |
| 3 | 011 | 110 | 6 |
| 4 | 100 | 001 | 1 |
| 5 | 101 | 101 | 5 |
| 6 | 110 | 011 | 3 |
| 7 | 111 | 111 | 7 |

## Appendix B: Butterfly Diagram for N=8

```
Stage 0        Stage 1        Stage 2
(stride=1)     (stride=2)     (stride=4)

u[0] ──┬──XOR── ──┬──XOR── ──┬──XOR── x[0]
       │          │          │
u[1] ──┴──────── ─│─┬──XOR── │──────── x[1]
                  │ │        │
u[2] ──┬──XOR── ──┴─│─────── │──────── x[2]
       │            │        │
u[3] ──┴──────── ───┴─────── │──────── x[3]
                             │
u[4] ──┬──XOR── ──┬──XOR── ──┴──────── x[4]
       │          │
u[5] ──┴──────── ─│─┬──XOR── ──────── x[5]
                  │ │
u[6] ──┬──XOR── ──┴─│─────── ──────── x[6]
       │            │
u[7] ──┴──────── ───┴─────── ──────── x[7]
```

Note: XOR operations are u[i] ^= u[i+stride] for each stage.

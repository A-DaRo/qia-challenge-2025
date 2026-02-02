# Phase E Protocol Bug Report: NetQASM 2.x / SquidASM 0.13.x Instruction Incompatibility

> **Status**: Unresolved  
> **Severity**: Blocker  
> **Affected Components**: `GenericProcessor`, `AppMemory`, `MeasBasisInstruction` dispatch  
> **Package Versions**: NetQASM 2.0+, SquidASM 0.13.6  
> **Test File**: `qia-challenge-2025/caligo/tests/e2e/test_phase_e_protocol.py`  
> **Last Updated**: December 2024

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Protocol Context: 1-out-of-2 OT under NSM](#2-protocol-context-1-out-of-2-ot-under-nsm)
3. [Error Manifestations](#3-error-manifestations)
4. [Root Cause Analysis](#4-root-cause-analysis)
5. [Comparison with Working QKD Example](#5-comparison-with-working-qkd-example)
6. [Source Code Analysis](#6-source-code-analysis)
7. [Call Flow Diagrams](#7-call-flow-diagrams)
8. [Reproduction Steps](#8-reproduction-steps)
9. [Potential Resolutions](#9-potential-resolutions)
10. [References](#10-references)

---

## 1. Executive Summary

The Caligo project implements a **1-out-of-2 Randomized Oblivious Transfer (1-2 ROT)** protocol under the **Noisy Storage Model (NSM)** using the SquidASM/NetQASM/NetSquid simulation stack. During end-to-end testing, critical runtime errors block quantum protocol execution:

| Error | Location | Root Cause |
|-------|----------|------------|
| `KeyError: None` | `squidasm/sim/stack/common.py:201` | Virtual qubit ID not mapped before `phys_id_for()` lookup |
| `RuntimeError: Unsupported instruction` | `squidasm/sim/stack/processor.py:734` | `MeasBasisInstruction` not handled by SquidASM 0.13.x |

### Core Finding

The 1-2 ROT protocol (like BB84 QKD) requires **measurements in both Z and X bases** for security. When the Caligo protocol requests X-basis measurement via `QubitMeasureBasis.X`, NetQASM generates `GenericInstr.MEAS_BASIS` which SquidASM's `GenericProcessor` does not handle. Additionally, certain instruction operand configurations result in `None` being passed to memory lookups.

**Key Insight**: The SquidASM `example_qkd.py` works because it implements basis rotation **manually** via Hadamard gates before Z-basis measurement, rather than using NetQASM's `MEAS_BASIS` instruction.

---

## 2. Protocol Context: 1-out-of-2 OT under NSM

### 2.1 What Caligo Actually Implements

Caligo implements the **Damgård-Fehr-Salvail-Schaffner (DFSS)** 1-out-of-2 Randomized Oblivious Transfer protocol, as experimentally demonstrated by Erven et al. (2014). This is fundamentally different from QKD:

| Aspect | QKD (BB84) | 1-2 ROT (Caligo) |
|--------|------------|------------------|
| **Goal** | Shared secret key | Alice: two random strings $S_0, S_1$; Bob: learns exactly $S_C$ |
| **Security** | Against eavesdropper | Against mutually distrusting parties |
| **Key Output** | Single shared key | Bob's choice bit $C$ determines which string |
| **Post-processing** | Privacy amplification on shared key | Separate amplification on $I_0$ and $I_1$ partitions |

### 2.2 Protocol Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    1-2 ROT PROTOCOL (CALIGO)                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 1: QUANTUM (EPR Distribution + Random Basis Measurement)            │
│  ┌────────────────────────┐    EPR Pairs    ┌────────────────────────┐     │
│  │        ALICE           │ ───────────────►│         BOB            │     │
│  │ • Generate EPR pairs   │                 │ • Receive EPR halves   │     │
│  │ • Measure in random    │                 │ • Measure in random    │     │
│  │   bases (Z or X)       │                 │   bases (Z or X)       │     │
│  │ • Record: α_i, X_i     │                 │ • Record: β_i, Y_i     │     │
│  └────────────────────────┘                 │ • Commit to results    │     │
│                                             └────────────────────────┘     │
│                                                                            │
│  PHASE 2: TIMING BARRIER (Δt wait - NSM security)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Wait time Δt ensures adversary's quantum storage has decohered     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│  PHASE 3: SIFTING + PARTITION                                              │
│  ┌────────────────────────┐                 ┌────────────────────────┐     │
│  │        ALICE           │                 │         BOB            │     │
│  │ • Reveal bases α^m     │ ──────────────► │ • Partition by basis   │     │
│  │ • Receive I_0, I_1     │ ◄────────────── │ • I_C: matching bases  │     │
│  │ • Partition X into     │                 │ • I_{1-C}: non-match   │     │
│  │   X|_{I_0}, X|_{I_1}   │                 │ • Choice bit C hidden  │     │
│  └────────────────────────┘                 └────────────────────────┘     │
│                                                                            │
│  PHASE 4: RECONCILIATION (One-way error correction)                        │
│  ┌────────────────────────┐                 ┌────────────────────────┐     │
│  │        ALICE           │                 │         BOB            │     │
│  │ • Compute syndromes    │                 │ • Correct Y|_{I_C}     │     │
│  │   syn(X|_{I_0}),       │ ──────────────► │ • Cannot correct       │     │
│  │   syn(X|_{I_1})        │                 │   Y|_{I_{1-C}}         │     │
│  └────────────────────────┘                 └────────────────────────┘     │
│                                                                            │
│  PHASE 5: PRIVACY AMPLIFICATION                                            │
│  ┌────────────────────────┐                 ┌────────────────────────┐     │
│  │        ALICE           │                 │         BOB            │     │
│  │ • S_0 = f_0(X|_{I_0})  │ ──seeds f_0,f_1─► • S_C = f_C(Y_{cor})   │     │
│  │ • S_1 = f_1(X|_{I_1})  │                 │                        │     │
│  │ • Outputs (S_0, S_1)   │                 │ • Outputs S_C          │     │
│  └────────────────────────┘                 └────────────────────────┘     │
│                                                                            │
│  SECURITY GUARANTEES (under NSM assumptions):                              │
│  • Sender: Bob learns at most one of S_0, S_1 (min-entropy bound)          │
│  • Receiver: Alice cannot determine C from protocol transcript             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Why Both Bases Are Required

The 1-2 ROT protocol's security critically depends on **uniform random basis selection** by both parties:

1. **Receiver Security**: Bob's choice bit $C$ is hidden because Alice cannot distinguish which partition ($I_0$ or $I_1$) corresponds to matching bases.

2. **Sender Security**: The min-entropy bound (Eq. 3 in Erven et al.) requires BB84 encoding:
   $$\tilde{H}_{\min}^\varepsilon(X^m|\alpha^m K) \ge m_1 \cdot C_{\text{BB84}}$$
   
   Where $C_{\text{BB84}}$ arises from measuring in **conjugate bases** (Z and X).

**Consequence**: Any implementation that only uses Z-basis measurements fundamentally breaks the protocol's security guarantees.

---

## 3. Error Manifestations

### 3.1 Error 1: `KeyError: None`

**Stack Trace**:
```
File ".../squidasm/sim/stack/processor.py", line 721, in _interpret_two_qubit_instr
    virt_id0 = app_mem.get_reg_value(instr.reg0)
File ".../squidasm/sim/stack/common.py", line 201, in phys_id_for
    return self._virt_qubits[virt_id]
KeyError: None
```

**Analysis**: The `get_reg_value()` method returns `None` when a register hasn't been set. This `None` propagates to `phys_id_for()` which performs a dictionary lookup without validation.

### 3.2 Error 2: `RuntimeError: Unsupported instruction`

When Caligo's `MeasurementExecutor` requests X-basis measurement:

```python
# caligo/quantum/measurement.py
measure_basis = (
    QubitMeasureBasis.Z if basis == BASIS_Z else QubitMeasureBasis.X
)
result = qubit.measure(basis=measure_basis)
```

NetQASM generates a `MeasBasisInstruction` (for X/Y basis rotation before measurement), which SquidASM cannot interpret.

---

## 4. Root Cause Analysis

### 4.1 The Measurement Basis Problem

**NetQASM SDK Behavior** (`netqasm/sdk/builder.py`):

```python
def _build_cmds_measure(self, qubit, ...):
    if basis == QubitMeasureBasis.Z:
        meas_command = ICmd(instruction=GenericInstr.MEAS, ...)
    elif basis == QubitMeasureBasis.X:
        # X-basis: Y-rotation by -π/2, then Z-measurement
        meas_command = ICmd(instruction=GenericInstr.MEAS_BASIS, 
                           operands=[..., 0, 24, 0, denom])  # -π/2 angle
    elif basis == QubitMeasureBasis.Y:
        # Y-basis: X-rotation by π/2, then Z-measurement
        meas_command = ICmd(instruction=GenericInstr.MEAS_BASIS,
                           operands=[..., 8, 0, 0, denom])   # +π/2 angle
```

**SquidASM Processor Dispatch** (`squidasm/sim/stack/processor.py`):

```python
def _interpret_instruction(self, app_id, instr):
    # ...
    elif isinstance(instr, core.MeasInstruction):      # Z-basis only!
        return self._interpret_meas(app_id, instr)
    # ...
    # MeasBasisInstruction is NOT handled - falls through to else
    else:
        raise RuntimeError(f"Invalid instruction {instr}")
```

### 4.2 Instruction Support Matrix

| Instruction | NetQASM | SquidASM 0.13.x | Used By |
|-------------|---------|-----------------|---------|
| `MeasInstruction` (Z-basis) | ✅ | ✅ | Both QKD example & Caligo |
| `MeasBasisInstruction` (X/Y) | ✅ | ❌ | Caligo only |
| `HInstruction` (Hadamard) | ✅ | ✅ | QKD example workaround |

---

## 5. Comparison with Working QKD Example

### 5.1 How `example_qkd.py` Works

The SquidASM QKD example (`squidasm/examples/applications/qkd/example_qkd.py`) successfully implements BB84-style basis selection without triggering the bug:

```python
# example_qkd.py - Working approach
def _distribute_states(self, context, is_init):
    for i in range(self._num_epr):
        basis = random.randint(0, 1)
        if is_init:
            q = epr_socket.create_keep(1)[0]
        else:
            q = epr_socket.recv_keep(1)[0]
        
        # KEY DIFFERENCE: Manual basis rotation
        if basis == 1:  # X-basis
            q.H()       # Apply Hadamard gate
        
        m = q.measure()  # Always Z-basis measurement!
        yield from conn.flush()
```

**Key Insight**: The QKD example:
1. Uses `q.H()` (Hadamard gate) for X-basis rotation
2. Always calls `q.measure()` without basis argument (defaults to Z)
3. This generates `HInstruction` + `MeasInstruction`, both supported by SquidASM

### 5.2 How Caligo Differs

```python
# caligo/quantum/measurement.py - Problematic approach
def measure_qubit(self, qubit, basis, round_id, context):
    measure_basis = (
        QubitMeasureBasis.Z if basis == BASIS_Z else QubitMeasureBasis.X
    )
    result = qubit.measure(basis=measure_basis)  # Generates MeasBasisInstruction!
```

**Problem**: Passing `basis=QubitMeasureBasis.X` to `qubit.measure()` generates the unsupported `MeasBasisInstruction`.

---

## 6. Source Code Analysis

### 6.1 NetQASM Measurement Instruction Generation

**File**: `netqasm/sdk/builder.py`, Lines 1140-1180

```python
def _build_cmds_measure(
    self, qubit: Qubit, future: Future, inplace: bool, 
    basis: QubitMeasureBasis = QubitMeasureBasis.Z
) -> None:
    """Build commands for measuring a qubit."""
    
    if basis == QubitMeasureBasis.Z:
        # Standard Z-basis measurement
        meas_command = ICmd(
            instruction=GenericInstr.MEAS,
            operands=[qubit_reg, outcome_reg],
        )
    elif basis == QubitMeasureBasis.X:
        # X-basis: rotate by -π/2 around Y, then Z-measure
        # This generates MeasBasisInstruction with rotation parameters
        denom = 32
        meas_command = ICmd(
            instruction=GenericInstr.MEAS_BASIS,
            operands=[qubit_reg, outcome_reg, 0, 24, 0, denom],
        )
    elif basis == QubitMeasureBasis.Y:
        # Y-basis: rotate by +π/2 around X, then Z-measure
        denom = 32
        meas_command = ICmd(
            instruction=GenericInstr.MEAS_BASIS,
            operands=[qubit_reg, outcome_reg, 8, 0, 0, denom],
        )
```

### 6.2 SquidASM Processor Instruction Dispatch

**File**: `squidasm/sim/stack/processor.py`, Lines 199-235

```python
def _interpret_instruction(
    self, app_id: int, instr: NetQASMInstruction
) -> Optional[Generator[EventExpression, None, None]]:
    
    if isinstance(instr, core.SetInstruction):
        return self._interpret_set(app_id, instr)
    # ... many handlers ...
    elif isinstance(instr, core.MeasInstruction):
        return self._interpret_meas(app_id, instr)  # Only Z-basis!
    elif isinstance(instr, core.SingleQubitInstruction):
        return self._interpret_single_qubit_instr(app_id, instr)
    elif isinstance(instr, core.TwoQubitInstruction):
        return self._interpret_two_qubit_instr(app_id, instr)
    # ... more handlers ...
    else:
        raise RuntimeError(f"Invalid instruction {instr}")
        # ^-- MeasBasisInstruction ends up here!
```

### 6.3 Missing `MeasBasisInstruction` Handler

The `core.MeasBasisInstruction` class exists in NetQASM but SquidASM has no handler for it. A proper handler would need to:

1. Extract rotation angles from instruction operands
2. Apply the rotation gate (ROT_X or ROT_Y)
3. Perform Z-basis measurement
4. Return the outcome

---

## 7. Call Flow Diagrams

### 7.1 Caligo Measurement Flow (Failing)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  caligo/quantum/measurement.py: MeasurementExecutor.measure_qubit()          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ measure_basis = QubitMeasureBasis.X  # For X-basis measurement         │  │
│  │ result = qubit.measure(basis=measure_basis)                            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  netqasm/sdk/qubit.py: Qubit.measure(basis=QubitMeasureBasis.X)              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ builder._build_cmds_measure(self, ..., basis=QubitMeasureBasis.X)      │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  netqasm/sdk/builder.py: _build_cmds_measure()                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ meas_command = ICmd(                                                   │  │
│  │     instruction=GenericInstr.MEAS_BASIS,  # NOT GenericInstr.MEAS!     │  │
│  │     operands=[qubit_reg, outcome_reg, 0, 24, 0, 32]                    │  │
│  │ )                                                                      │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  squidasm/sim/stack/processor.py: _interpret_instruction()                   │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ # isinstance(instr, core.MeasInstruction) → False                      │  │
│  │ # isinstance(instr, core.MeasBasisInstruction) → True, BUT NO HANDLER  │  │
│  │ #                                                                      │  │
│  │ else:                                                                  │  │
│  │     raise RuntimeError(f"Invalid instruction {instr}")  # ← FAILURE    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 QKD Example Flow (Working)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  example_qkd.py: _distribute_states()                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ if basis == 1:                                                         │  │
│  │     q.H()        # Manual Hadamard for X-basis                         │  │
│  │ m = q.measure()  # No basis argument → defaults to Z                   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Generates two instructions:                                                 │
│  1. HInstruction (Hadamard) → handled by _interpret_single_qubit_instr()     │
│  2. MeasInstruction (Z-basis) → handled by _interpret_meas()                 │
│                                                                              │
│  Both instructions are supported → ✅ SUCCESS                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Reproduction Steps

### 8.1 Minimal Reproduction

```python
"""Minimal reproduction of the MeasBasisInstruction bug."""

from squidasm.run.stack.run import run
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
from squidasm.util import create_two_node_network


class AliceProgram(Program):
    PEER = "Bob"

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="alice_program",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        from netqasm.sdk import QubitMeasureBasis
        
        conn = context.connection
        epr_socket = context.epr_sockets[self.PEER]

        q = epr_socket.create_keep(1)[0]
        
        # This triggers MeasBasisInstruction - will fail
        result = q.measure(basis=QubitMeasureBasis.X)
        yield from conn.flush()

        return {"result": int(result)}


# BobProgram similar...

def run_test():
    cfg = create_two_node_network(num_qubits=2)
    # This will raise RuntimeError: Invalid instruction ...
    run(config=cfg, programs={"Alice": AliceProgram(), "Bob": BobProgram()})
```

### 8.2 Expected vs. Actual

| Test | Expected | Actual |
|------|----------|--------|
| Z-basis measurement | ✅ Works | ✅ Works |
| X-basis via `q.measure(basis=X)` | Should work | ❌ `RuntimeError` |
| X-basis via `q.H(); q.measure()` | ✅ Works | ✅ Works |

---

## 9. Potential Resolutions

### 9.1 Resolution 1: Implement `MeasBasisInstruction` Handler (Recommended for SquidASM)

**Location**: `squidasm/sim/stack/processor.py`

```python
def _interpret_instruction(self, app_id: int, instr: NetQASMInstruction):
    # ... existing handlers ...
    elif isinstance(instr, core.MeasBasisInstruction):
        return self._interpret_meas_basis(app_id, instr)
    # ...

def _interpret_meas_basis(
    self, app_id: int, instr: core.MeasBasisInstruction
) -> Generator[EventExpression, None, None]:
    """Handle measurement in arbitrary basis via rotation + Z-measurement."""
    app_mem = self.app_memories[app_id]
    virt_id = app_mem.get_reg_value(instr.qubit_reg)
    phys_id = app_mem.phys_id_for(virt_id)
    
    # Extract rotation angles from instruction operands
    # Apply rotation gate
    # Perform Z-basis measurement
    # Store result in outcome register
```

### 9.2 Resolution 2: Caligo Workaround - Manual Basis Rotation (Recommended for Now)

Modify `caligo/quantum/measurement.py` to follow the QKD example pattern:

```python
class MeasurementExecutor:
    def measure_qubit(
        self, qubit: "Qubit", basis: int, round_id: int, context: Any
    ) -> Generator[Any, None, int]:
        """Measure qubit using manual basis rotation (SquidASM-compatible)."""
        
        if basis == BASIS_X:
            # X-basis: apply Hadamard, then Z-measure
            qubit.H()
        elif basis == BASIS_Y:
            # Y-basis: apply S†H, then Z-measure
            qubit.S()
            qubit.H()
        
        # Always Z-basis measurement (supported by SquidASM)
        result = qubit.measure()
        yield from context.connection.flush()
        
        return int(result)
```

**Advantages**:
- No changes to SquidASM required
- Matches working QKD example pattern
- Semantically equivalent to `MeasBasisInstruction`

### 9.3 Resolution 3: Defensive Guards in AppMemory

**Location**: `squidasm/sim/stack/common.py`

```python
def phys_id_for(self, virt_id: int) -> int:
    if virt_id is None:
        raise ValueError("Virtual qubit ID cannot be None - register not set")
    if virt_id not in self._virt_qubits:
        raise KeyError(f"Virtual qubit ID {virt_id} not mapped to physical qubit")
    return self._virt_qubits[virt_id]
```

---

## 10. References

### 10.1 Protocol References

- Erven et al. (2014), "An Experimental Implementation of Oblivious Transfer in the Noisy Storage Model", arXiv:1308.5098
- Lupo et al. (2023), "Error-tolerant oblivious transfer in the noisy-storage model", arXiv:2308.05098
- Damgård et al. (2005), "Cryptography in the Bounded Quantum Storage Model"
- Schaffner et al. (2009), "Simple protocols for oblivious transfer"

### 10.2 Source Code Locations

| File | Lines | Description |
|------|-------|-------------|
| `caligo/quantum/measurement.py` | 250-300 | `MeasurementExecutor.measure_qubit()` |
| `netqasm/sdk/builder.py` | 1140-1180 | `_build_cmds_measure()` |
| `squidasm/sim/stack/processor.py` | 199-235 | `_interpret_instruction()` dispatch |
| `squidasm/examples/applications/qkd/example_qkd.py` | 52-72 | Working basis rotation pattern |

### 10.3 Package Versions

```
squidasm==0.13.6
netqasm>=2.0
netsquid>=1.1.6
```

---

*Document Version: 2.0*  
*Last Updated: December 2024*  
*Protocol: 1-out-of-2 Randomized Oblivious Transfer under Noisy Storage Model*

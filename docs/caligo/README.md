# Caligo Documentation Hub

<metadata>
version: 1.0.0
status: active
last_updated: 2026-01-30
</metadata>

## Overview

<overview>
This directory contains the technical documentation for the Caligo quantum network simulation framework, with a focus on the **HPC-Ready Polar Codec with LDPC Deprecation Refactor** initiative.

Documentation is structured for consumption by both human developers and AI agents, using XML-tagged sections for machine parsing and MADR conventions for architectural decisions.
</overview>

---

## Directory Structure

```
docs/caligo/
├── README.md                 # This file (navigation index)
├── initialization.md         # Project initialization and reference mapping
│
├── adr/                      # Architectural Decision Records (MADR format)
│   ├── 0000-adr-template.md
│   ├── 0001-polar-codec-adoption.md
│   ├── 0002-rust-native-extension.md
│   └── 0003-kiktenko-baseline.md
│
├── agents/                   # AI Agent System Instructions
│   ├── context_engineer.md
│   ├── systems_architect.md
│   ├── integration_engineer.md
│   └── verification_expert.md
│
├── specs/                    # Interface Contracts (AI-optimized)
│   ├── siso-codec-protocol.md
│   ├── rust-polar-crate.md
│   └── reconciliation-strategy.md
│
├── impl/                     # Implementation Guides (with test vectors)
│   ├── phase1-rust-foundation.md
│   ├── phase2-scl-decoder.md
│   ├── phase3-strategy-refactor.md
│   └── phase4-integration.md
│
└── archive/                  # Historical context (read-only)
    └── hpc_migration_roadmap.md
```

---

## Document Types

### ADR (Architectural Decision Records)

<adr_description>
Immutable rationale for major architectural decisions. Follow MADR v3.0.0 format with XML extensions for AI parsing.

**When to create an ADR:**
- Introducing a new technology (Rust, Polar codes)
- Deprecating existing functionality (LDPC strategies)
- Changing system interfaces (Protocol modifications)

**Template:** [adr/0000-adr-template.md](adr/0000-adr-template.md)
</adr_description>

### Specifications

<specs_description>
Interface contracts defining the "what" without implementation details. Optimized for AI agent consumption with explicit invariants, preconditions, and postconditions.

**Key conventions:**
- `<interface>` blocks contain valid Python code (parseable by `ast.parse()`)
- `<invariants>` define properties that must always hold
- `<errors>` enumerate all exception conditions
</specs_description>

### Implementation Guides

<impl_description>
Step-by-step guides with acceptance criteria and test vectors. Support both human review and AI agent execution.

**Key conventions:**
- `<tasks>` enumerate discrete work items
- `<acceptance_criteria>` provide testable conditions
- `<test_vectors>` enable automated validation
</impl_description>

### AI Agent Instructions

<agents_description>
System prompts for specialized AI agents operating on this project. Each agent has a distinct role and responsibility boundary.

**Available Agents:**
- **Context Engineer** — Documentation, ADRs, strategic planning
- **Systems Architect** — Rust/PyO3, HPC optimization, memory safety
- **Integration Engineer** — Python strategy layer, backward compatibility
- **Verification Expert** — Test suites, scientific validation, statistical rigor

**Usage:** Activate an agent by loading its instruction file as context.
</agents_description>

---

## Quick Navigation

| Document | Purpose | Priority |
|----------|---------|----------|
| [initialization.md](initialization.md) | Reference mapping and project context | P0 |
| [adr/0001-polar-codec-adoption.md](adr/0001-polar-codec-adoption.md) | Why Polar over LDPC | P0 |
| [specs/siso-codec-protocol.md](specs/siso-codec-protocol.md) | Codec interface contract | P0 |
| [impl/phase1-rust-foundation.md](impl/phase1-rust-foundation.md) | Rust crate setup | P1 |
| [adr/0002-rust-native-extension.md](adr/0002-rust-native-extension.md) | Why Rust/PyO3 | P1 |
| [specs/rust-polar-crate.md](specs/rust-polar-crate.md) | Rust API specification | P2 |
| [impl/phase2-scl-decoder.md](impl/phase2-scl-decoder.md) | SCL decoder implementation | P2 |

### Agent Instructions

| Agent | Role | Instruction File |
|-------|------|------------------|
| Context Engineer | The Librarian | [agents/context_engineer.md](agents/context_engineer.md) |
| Systems Architect | The Rustacean | [agents/systems_architect.md](agents/systems_architect.md) |
| Integration Engineer | The BridgeBuilder | [agents/integration_engineer.md](agents/integration_engineer.md) |
| Verification Expert | The Scientist | [agents/verification_expert.md](agents/verification_expert.md) |

---

## Contributing

<contributing>
1. All new ADRs must use the template at `adr/0000-adr-template.md`
2. Specifications must include `<metadata>` with version and status
3. Implementation docs must include `<test_vectors>` for codec tasks
4. All documents must be linked from this README
5. XML tags must be well-formed for AI parsing
</contributing>

---

## Related Resources

<resources>
- **Codebase:** `qia-challenge-2025/caligo/caligo/`
- **Literature:** `qia-challenge-2025/docs/literature/`
- **SquidASM Docs:** `qia-challenge-2025/docs/squidasm_docs/`
- **HPC Context:** [archive/hpc_migration_roadmap.md](archive/hpc_migration_roadmap.md)
</resources>

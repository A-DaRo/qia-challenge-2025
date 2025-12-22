# QIA Challenge 2025

This repository is the main workspace for the **QIA Challenge 2025** project.
It contains the implementation work (Python package), internal documentation, and the written report.
The project builds from the experience at the **QIA's Pan-European Quantum Internet Hackathon 2025**, you can check you our contribution here [Extending-Quantum-Key-Distribution---TU-e-participants](https://github.com/qia-hackathon-2025/Extending-Quantum-Key-Distribution---TU-e-participants)

## Repository Structure

At a glance:

```
qia-challenge-2025/
	caligo/
	docs/
	report/
	LICENSE.md
	README.md
```

### `caligo/`

The **primary codebase** for the challenge.

contents:

- `caligo/pyproject.toml`: Python packaging and tooling configuration.
- `caligo/README.md`: Package-specific usage notes.
- `caligo/caligo/`: The actual Python package source (all new code should live here).
- `caligo/configs/`: Experiment and simulation configuration files.
- `caligo/tests/`: Unit/integration tests for the `caligo` package.

you can find the [Technical README](./caligo/README.md) for more information about the `caligo/` project.

### `docs/`

Project documentation hub.

contents:

- `docs/caligo/`: Documentation specific to the `caligo` implementation.
- `docs/coding_guidelines/`: Project coding standards (e.g., Numpydoc, style conventions).
- `docs/literature/`: References and supporting reading for the theoretical background.
- `docs/squidasm_docs/`: Notes and documentation for SquidASM usage in this project.

### `report/`

The written report for the challenge, you can find the [Index](./report/index.md) to look through the chapters and sections.

## Where to Start

- If you want to run or modify the implementation, start in `caligo/`.
- If you are reading the deliverable narrative, start in `report/index.md`.
- If you want to extend the project, find additional documentation or read relevant literature, check `docs/`.

---

## Acknowledgements

- Rianne S. Lous, for teaching the course *34IQT Introduction to Quantum Technologies*, to it my first glance at the world of quantum physics and technologies. To the curiosity and understanding I obtained from the course, that allowed me to engage in these initiatives.

-  Wojciech Kozlowski, for promoting and hosting the *Pan-European Quantum Internet Hackathon*; to the hands-on experience it offered and the motivation to challenge myself further with this project. To the suggestion of the paper *Generation and Distribution of Quantum Oblivious Keys for Secure Multiparty Computation* by Lemus et al., which introduced me to Oblivious Transfer and allowed me to investigate deeper, forming the basis of this work.
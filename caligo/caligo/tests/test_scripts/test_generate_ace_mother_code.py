import sys
from pathlib import Path
import scipy.sparse as sp
import numpy as np

import caligo.scripts.generate_ace_mother_code as gen_script
from caligo.scripts.peg_generator import DegreeDistribution


def test_main_enforces_filename_convention(tmp_path, monkeypatch):
    # Prepare a tiny matrix to avoid heavy computation
    tiny_H = sp.csr_matrix(np.eye(8, dtype=np.uint8))

    # Monkeypatch the heavy parts
    monkeypatch.setattr(gen_script.ACEPEGGenerator, "generate", lambda self: tiny_H)
    # Provide simple degree distributions
    simple = DegreeDistribution(degrees=[1], probabilities=[1.0])
    monkeypatch.setattr(gen_script, "load_degree_distributions", lambda config_path, rate: (simple, simple))

    output_dir = tmp_path / "ldpc_matrices"
    # Call main with args
    sys_argv = [
        "prog",
        "--block-length",
        "4096",
        "--rate",
        "0.5",
        "--output-path",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", sys_argv)

    rc = gen_script.main()
    assert rc == 0

    expected = output_dir / "ldpc_4096_rate0.50.npz"
    assert expected.exists()
    # Load and verify matrix shape
    loaded = sp.load_npz(expected)
    assert loaded.shape == tiny_H.shape

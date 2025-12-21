import numpy as np
import scipy.sparse as sp
import logging

from caligo.scripts.generate_ldpc_matrices import generate_all
from caligo.scripts.peg_generator import DegreeDistribution


class StubPEG:
    def __init__(self, n, rate, lambda_dist, rho_dist, max_tree_depth, seed):
        self.n = n
        self.rate = rate

    def generate(self):
        m = max(1, int(self.n * (1 - self.rate)))
        # return a tiny matrix of shape (m, n)
        rows = [0]
        cols = [0]
        data = [1]
        return sp.csr_matrix((data, (rows, cols)), shape=(m, self.n), dtype=np.uint8)


def test_generate_all_creates_files(tmp_path, monkeypatch):
    # Monkeypatch distribution loading to avoid external files
    monkeypatch.setattr(
        'caligo.scripts.generate_ldpc_matrices._load_degree_distributions',
        lambda: {0.5: {'lambda': DegreeDistribution([2, 3], [0.5, 0.5]), 'rho': DegreeDistribution([6], [1.0])}}
    )

    # Patch PEG generator to stub implementation
    monkeypatch.setattr('caligo.scripts.generate_ldpc_matrices.PEGMatrixGenerator', StubPEG)

    logger = logging.getLogger('test')
    output_dir = tmp_path / 'ldpc'
    frame_size = 16
    rates = [0.5]

    generate_all(output_dir, logger, frame_size=frame_size, rates=rates)

    # Ensure file exists with expected name format
    from caligo.reconciliation import constants
    expected_name = constants.LDPC_MATRIX_FILE_PATTERN.format(frame_size=frame_size, rate=0.5)
    # Some filesystems or save logic may round/format differently; check for matching prefix
    files = [p.name for p in output_dir.iterdir()] if output_dir.exists() else []
    assert any(name.startswith(expected_name.replace('.npz','')) for name in files), f"no matching file in {files}"

"""Generate LDPC matrices for various code lengths."""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from ..utils.logging import get_logger

logger = get_logger("ldpc_generator")

def generate_regular_ldpc(n: int, rate: float, w_c: int = 3) -> sp.spmatrix:
    """
    Generate regular LDPC matrix.
    
    Parameters
    ----------
    n : int
        Code length.
    rate : float
        Target code rate k/n.
    w_c : int
        Column weight (typical: 3-4).
    
    Returns
    -------
    H : sparse matrix
        Parity check matrix.
    """
    m = int(n * (1 - rate))
    # w_r = (w_c * n) // m  # Row weight
    
    # Progressive edge growth (PEG) algorithm would go here
    # Simplified: random regular construction
    # We use lil_matrix for efficient construction
    H = sp.lil_matrix((m, n), dtype=np.uint8)
    
    # Ensure each column has exactly w_c ones
    for col in range(n):
        # Randomly choose w_c rows for this column
        rows = np.random.choice(m, w_c, replace=False)
        H[rows, col] = 1
    
    return H.tocsr()

# Generate matrices for baseline
if __name__ == "__main__":
    output_dir = Path(__file__).parent / "ldpc_matrices"
    output_dir.mkdir(exist_ok=True)
    
    # Generate for expected sifted key sizes
    # These sizes should align with what we expect from the protocol
    # e.g. 10,000 raw bits -> ~5,000 sifted bits -> ~4,500 after test set
    for n in [1000, 2000, 4500, 5000]:
        H = generate_regular_ldpc(n, rate=0.5)
        filename = output_dir / f"ldpc_{n}_rate05.npz"
        sp.save_npz(filename, H)
        logger.info(f"Generated LDPC matrix: {H.shape} saved to {filename}")

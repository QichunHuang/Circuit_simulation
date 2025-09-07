import numpy as np


def build_inductance_matrix(params):
    """
    Build complete mutual inductance matrix M based on parameters (optimized version)
    Uses vectorized operations instead of double loops for significant performance improvement
    """
    L_diag = params["L"]
    K_matrix = params["K"]

    # Vectorized calculation: M = diag(L) + K * sqrt(L_i * L_j)
    sqrt_L = np.sqrt(L_diag)
    # Use outer product to build sqrt(L_i * L_j) matrix
    sqrt_LL = np.outer(sqrt_L, sqrt_L)

    # Mutual inductance matrix: M = diag(L) + K .* sqrt(L_i * L_j)
    M = np.diag(L_diag) + K_matrix * sqrt_LL

    return M


def calculate_modes(params, verbose=True):
    """
    Calculate eigen modes of resonator array (optimized version)
    Equivalent to fn_ndcoilmodes.m

    Optimization improvements:
    1. Use scipy.linalg.eigh to solve symmetric matrix eigenvalues (faster and more stable)
    2. Directly solve generalized eigenvalue problem, avoiding matrix inversion
    3. Take only real parts to avoid complex arithmetic

    Args:
        params (dict): Parameter dictionary containing 'L', 'C', 'K', 'N'
        verbose (bool): Whether to print calculated frequencies

    Returns:
        tuple: (f0e, V)
            f0e (np.array): Eigen frequencies (Hz), sorted
            V (np.array): Corresponding eigenvectors (mode shapes), column vectors
    """
    from scipy.linalg import eigh

    C_inv = np.diag(1.0 / params["C"])
    M = build_inductance_matrix(params)

    # Use generalized eigenvalue solver, avoid matrix inversion, improve numerical stability
    # Solve: C_inv * V = w^2 * M * V
    w2, V = eigh(C_inv, M)

    # Convert to frequency (Hz)
    f0e = np.sqrt(np.abs(w2)) / (2 * np.pi)

    # Frequencies and eigenvectors are already sorted in ascending order (eigh property)
    if verbose:
        print("--- Eigen-frequencies (Modes) ---")
        for i, f in enumerate(f0e):
            print(f"Mode {i + 1}: {f / 1e6:.4f} MHz")
        print("-" * 30)

    return f0e, V


def calculate_impedance_spectrum(freq_axis, params, driven_coil_index):
    """
    Calculate input impedance spectrum under specified coil drive (vectorized optimized version)
    Equivalent to fn_ndcoileval.m

    Optimization improvements:
    1. Vectorized frequency calculation, process all frequency points at once
    2. Reduce repeated matrix construction operations
    3. Use broadcast operations to improve efficiency
    4. Pre-allocate memory to avoid dynamic expansion

    Args:
        freq_axis (np.array): Frequency points for impedance calculation (Hz)
        params (dict): Parameter dictionary containing 'L', 'C', 'R', 'K', 'N'
        driven_coil_index (int): Driven coil number (1-based index)

    Returns:
        np.array: Complex impedance spectrum Z0 (Ohms)
    """
    N = params["N"]
    Kval = params["K"]
    C_vals = params["C"]
    L_vals = params["L"]
    R_vals = params["R"]

    coiln = driven_coil_index - 1  # Convert to 0-based index
    if not 0 <= coiln < N:
        raise ValueError(f"Driven coil index must be between 1 and {N}")

    # Pre-calculate constant terms
    Cval = np.diag(C_vals)
    Lval = np.diag(L_vals)
    Rval = np.diag(R_vals)

    # Build mutual inductance matrix (reuse build_inductance_matrix logic but more efficient)
    sqrt_L = np.sqrt(L_vals)
    Mval = np.diag(L_vals) + Kval * np.outer(sqrt_L, sqrt_L)

    # A_empty: Mutual inductance matrix with diagonal elements removed
    A_empty = Mval * (1 - np.eye(N))

    # Solution vector: only 1 at driven coil position
    sol = np.zeros(N)
    sol[coiln] = 1.0

    # Vectorized calculation: process all frequencies
    w_array = 2 * np.pi * freq_axis
    Z0_spectrum = np.zeros(len(freq_axis), dtype=complex)

    # Pre-calculate invariant terms
    C_driven = C_vals[coiln]
    R_driven = R_vals[coiln]
    L_driven = L_vals[coiln]

    # Optimized frequency loop - reuse calculation results
    for idx, w in enumerate(w_array):
        # Build system matrix A
        jwC_inv = 1.0 / (1j * w * C_vals)
        A = A_empty * (1j * w) + Rval + np.diag(jwC_inv) + (1j * w) * Lval

        # Modify self-impedance of driven coil
        A[coiln, coiln] = R_driven + 1j * w * L_driven

        # Solve linear equation system
        Z1_ = np.linalg.solve(A, sol)
        Z1_driven = 1.0 / Z1_[coiln]

        # Apply transfer function
        Z0_spectrum[idx] = Z1_driven / (1.0 + 1j * w * C_driven * Z1_driven)

    return Z0_spectrum

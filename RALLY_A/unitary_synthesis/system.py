import numpy as np
from qutip import qeye, sigmax, sigmaz, tensor

# Default number of atoms/qubits and scaling factor for coordinates\N = 3  # Number of atoms
scaling_factor = 8

# Rydberg atom positions
reg = {
    "q0": np.array([-1 * scaling_factor, 0]),
    "q1": np.array([0, 1.2 * scaling_factor]),
    "q2": np.array([1.1 * scaling_factor, 0]),
}

def sigma_x(i: int, n: int) -> np.ndarray:
    """
    Single-qubit Pauli-X operator acting on qubit i in an n-qubit system.

    Args:
        i: Index of the target qubit (0-based).
        n: Total number of qubits.

    Returns:
        The full 2^n x 2^n matrix for sigma_x on qubit i.
    """
    ops = [qeye(2) for _ in range(n)]
    ops[i] = sigmax()
    return tensor(ops)

def sigma_z(i: int, n: int) -> np.ndarray:
    """
    Single-qubit Pauli-Z operator acting on qubit i in an n-qubit system.

    Args:
        i: Index of the target qubit (0-based).
        n: Total number of qubits.

    Returns:
        The full 2^n x 2^n matrix for sigma_z on qubit i.
    """
    ops = [qeye(2) for _ in range(n)]
    ops[i] = sigmaz()
    return tensor(ops)


def rydberg_interaction(i: int, j: int, n: int, C: float) -> np.ndarray:
    """
    Two-qubit Rydberg interaction term between qubits i and j.

    Args:
        i: Index of the first qubit.
        j: Index of the second qubit.
        n: Total number of qubits.
        C: Interaction coefficient.

    Returns:
        The scaled interaction operator.
    """
    ops = [qeye(2) for _ in range(n)]
    # (|1><1|)_i = (I - Z)/2
    projector = (qeye(2) - sigmaz()) / 2
    ops[i] = projector
    ops[j] = projector
    return C * tensor(ops)


def transverse_term(n: int) -> np.ndarray:
    """
    Global transverse field term: sum over X on all qubits.

    Args:
        n: Number of qubits.

    Returns:
        Sum_i sigma_x(i) operator.
    """
    return sum(sigma_x(i, n) for i in range(n))


def longitudinal_term(n: int) -> np.ndarray:
    """
    Global longitudinal field term: sum over Z on all qubits.

    Args:
        n: Number of qubits.

    Returns:
        Sum_i sigma_z(i) operator.
    """
    return sum(sigma_z(i, n) for i in range(n))

def rydberg_term(n: int, distances: dict[tuple[str, str], float], C0: float = 5e6) -> np.ndarray:
    """
    Total Rydberg interaction Hamiltonian based on pairwise distances.

    Args:
        n: Number of qubits.
        distances: Dict of pairwise distances between qubit labels.
        C0: Base interaction coefficient (default=5e6).

    Returns:
        Sum of C0 / dist^6 * interaction operators.
    """
    H = 0
    for (qa, qb), d in distances.items():
        idx_a = int(qa[1])
        idx_b = int(qb[1])
        coeff = C0 * d**(-6)
        H += rydberg_interaction(idx_a, idx_b, n, coeff)
    return H


def compute_distances(coords: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """
    Calculate pairwise Euclidean distances between labeled points.

    Args:
        coords: Mapping from label to 2D coordinates.

    Returns:
        Dictionary mapping qubit-pair (label_i, label_j) to distance.
    """

    return {(i, j): np.linalg.norm(coords[i] - coords[j])
            for i in coords for j in coords if i < j}



def generate_static_H(n: int) -> np.ndarray:
    """
    Generate the static (time-independent) part of the Hamiltonian.

    Args:
        n: Number of qubits.

    Returns:
        Static Hamiltonian matrix (2^n x 2^n).
    """
    distances = compute_distances(reg)
    Hx = transverse_term(n)
    Hrr = rydberg_term(n, distances)
    return Hx + Hrr


def generate_control_H(n: int) -> np.ndarray:
    """
    Generate the control Hamiltonian (longitudinal field) if needed.

    Args:
        n: Number of qubits.

    Returns:
        Control Hamiltonian matrix (2^n x 2^n).
    """
    return longitudinal_term(n)

def generate_random_problem(n: int) -> dict[str, np.ndarray]:
    """
    Generate a random problem instance: initial state and a test unitary.

    Args:
        n: Number of qubits.

    Returns:
        Dictionary containing:
            'dim': Hilbert space dimension (2^n).
            'psi_0': Initial state vector.
            'U_t': A sample two-qubit unitary.
    """
    dim = 2**n
    # Initial state |00...0>
    psi_0 = np.zeros(dim, dtype=complex)
    psi_0[0] = 1

    # Example: CNOT on qubits 0 & 1
    I2 = np.eye(2)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    U_t = np.kron(CNOT, I2)

    return {
        'dim': dim,
        'psi_0': psi_0,
        'U_t': U_t,
    }


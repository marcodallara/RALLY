import numpy as np
from qutip import qeye, sigmax, sigmaz, tensor
from functools import reduce

def cache_single_qubit_ops(N):
    """
    Precompute and cache the single-qubit sigma_x and sigma_z operators for each qubit.
    """
    sigma_x_ops = []
    sigma_z_ops = []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sigma_x_ops.append(tensor(op_list))
        op_list = [qeye(2)] * N
        op_list[i] = sigmaz()
        sigma_z_ops.append(tensor(op_list))
    return sigma_x_ops, sigma_z_ops

def generate_static_H(n, coeffs = None):
    """
    Generate static Hamiltonian terms using the provided coefficients.
    
    Args:
        coeffs: Dictionary containing alpha and beta coefficients
        n: Number of qubits
        
    Returns:
        Hx, Hz, T: The static Hamiltonian components
    """
    alpha = coeffs['alpha']
    beta = coeffs['beta']
    
    sigma_x_ops, sigma_z_ops = cache_single_qubit_ops(n)
    
    # H_x = \sum_i alpha_i \sigma_x^{(i)}
    Hx = sum(alpha[i] * sigma_x_ops[i] for i in range(n))
    
    # H_z = \sum_i beta_i \sigma_z^{(i)}
    Hz = sum(beta[i] * sigma_z_ops[i] for i in range(n))
    
    
    return Hx.full() + Hz.full()

def generate_control_H(n, coeffs = None):
    """
    Generate additional control Hamiltonians if needed.
    
    Args:
        coeffs: Dictionary containing control coefficients
        n: Number of qubits
        
    Returns:
        Control Hamiltonian components
    """
    # total_sigma_zz = sum_{i=0}^{N-2} sigma_z(i) sigma_z(i+1)
    T = 0
    for i in range(n - 1):
        op_list = [qeye(2)] * n
        op_list[i] = sigmaz()
        op_list[i+1] = sigmaz()
        T += tensor(op_list)
        
    return T.full()


def ghz_n_qubits(N):
    if N < 2:
        raise ValueError("Bell state requires at least 2 qubits")

    # Define basis states
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)

    # |00...0>
    zero_state = reduce(np.kron, [zero] * N)
    # |11...1>
    one_state = reduce(np.kron, [one] * N)

    # Bell state generalized: (|00...0> + |11...1>) / sqrt(2)
    bell_state = (zero_state + one_state) / np.sqrt(2)
    return bell_state




def generate_random_problem(n, energy_or_fid="energy", seed=0):
    """
    Generate a random problem instance.
    
    Args:
        n: Number of qubits
        energy_or_fid: String indicating whether to optimize for energy or fidelity
        
    Returns:
        Dictionary with problem parameters
    """
    dim = 2**n
    psi_0 = np.zeros(dim)
    psi_0[0] = 1
    # psi_0 = np.random.rand(dim) + 1j * np.random.rand(dim)
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    
    # # Generate random coefficients
    # coeffs = {
    #     'alpha': np.random.rand(n),
    #     'beta': np.random.rand(n)
    # }
    # coeffs = {
    #     'alpha': np.ones(n),
    #     'beta': np.ones(n)
    # }

    # seed = 42          # fixed seed for reproducibility
    rng  = np.random.default_rng(seed)

    # coeffs = {
    #     'alpha': rng.uniform(0.8, 1.2, size=n),  # values in [0.8, 1.2)
    #     'beta':  rng.uniform(0.8, 1.2, size=n)
    # }
    coeffs = {
        'alpha': rng.uniform(0.5, 1, size=n),  # values in [0.8, 1.2)
        'beta':  rng.uniform(0.5, 1, size=n)
    }
    problem = {
        'dim': dim,
        'psi_0': psi_0,
        'coeffs': coeffs
    }
    
    
    if energy_or_fid == "energy":
        # Create random Hermitian target Hamiltonian
        H_t = (lambda A: (A + A.conj().T)/2)(np.random.rand(dim, dim) + 1j*np.random.rand(dim, dim))
        # Find ground state energy
        w, v = np.linalg.eigh(H_t)
        gs_energy = w[0]
        problem['H_t'] = H_t
        problem['gs_energy'] = gs_energy
    else:  # fidelity
        psi_t = ghz_n_qubits(n)
        # psi_t = np.random.rand(dim) + 1j * np.random.rand(dim)
        # psi_t = psi_t / np.linalg.norm(psi_t)
        problem['psi_t'] = psi_t
    
    return problem



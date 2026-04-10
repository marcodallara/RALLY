import os
import sys
import json
import time
import pickle
import numpy as np
from mpi4py import MPI
from scipy.optimize import minimize
from scipy.linalg import expm, eigh
import argparse

from system import generate_static_H, generate_control_H, generate_random_problem

# IF YOU WANT TO GENERALIZE FOR EACH POSSIBLE SYSTEM YOU NEED TO DO SOME MODIFICATIONS!


def precompute_hamiltonians(H_static, H_control, r):
    K, L = r.shape
    d = H_static.shape[0]
    eigvals = np.zeros((K, L, d))
    eigvecs = np.zeros((K, L, d, d), dtype=complex)
    for k in range(K):
        for i in range(L):
            H_eff = H_static + r[k, i] * H_control
            lam, vec = eigh(H_eff)
            eigvals[k, i] = lam
            eigvecs[k, i] = vec
    return eigvals, eigvecs


def ansatz_optimized(thetas, eigvals, eigvecs):
    K, L, d = eigvals.shape
    U = np.eye(d, dtype=complex)
    for k in range(K):
        for i in range(L):
            phases = np.exp(-1j * thetas[k] * eigvals[k, i])
            U_k_i = eigvecs[k, i]
            U_dag = U_k_i.conj().T
            expH = U_k_i @ (phases[:, None] * U_dag)
            U = expH @ U
    return U


def unitary_fidelity(U_target: np.ndarray, U_actual: np.ndarray) -> float:
    """
    Compute the process fidelity between two unitary operators.

    F_process = |Tr(U_target^† U_actual)|^2 / d^2

    Args:
        U_target: Ideal unitary matrix.
        U_actual: Actual unitary matrix.

    Returns:
        Process fidelity between 0 and 1.
    """
    if U_target.shape != U_actual.shape:
        raise ValueError("Unitary matrices must have the same dimensions.")
    d = U_target.shape[0]
    overlap = np.trace(U_target.conj().T @ U_actual)
    return np.abs(overlap)**2 / (d**2)

def fidelity_cost(thetas: np.ndarray,
                  eigvals: np.ndarray,
                  eigvecs: np.ndarray,
                  U_target: np.ndarray
                  ) -> float:
    """
    Cost function for fidelity maximization (negated fidelity).

    Args:
        thetas: Variational angles.
        params: Control parameters.
        U_target: Target unitary.
        H_static: Static Hamiltonian.
        H_control: Control Hamiltonian.
        n_qubits: System size.
        n_layers: Sub-layers per angle.

    Returns:
        Negative process fidelity (to minimize).
    """
    U_ans = ansatz_optimized(thetas, eigvals, eigvecs)
    # U_ans = ansatz(thetas, params, H_static, H_control, n_qubits, n_layers)
    return -unitary_fidelity(U_target, U_ans)


def run_optimization(problem: dict,
                     n: int,
                     N_layers: int,
                     n_thetas: int,
                     Max_theta_init: float,
                    lambda_penalty: float = 1e-4        # <-- new arg
                     ) -> dict:
    """
    Set up and run the classical optimization routine.

    Args:
        problem: Contains 'dim', 'psi_0', 'U_t'.
        n: Number of qubits.
        N_layers: Sub-layers per variational angle.
        n_thetas: Number of variational angles.
        Max_theta_init: Upper bound for random theta initialization.

    Returns:
        Dictionary with optimization results and metadata.
    """

    # Unpack problem
    U_target = problem['U_t']

    # Construct Hamiltonians
    H_static = generate_static_H(n).full()
    H_control = generate_control_H(n).full()

    # Initialize parameters
    params_list = []
    thetas = np.array([])

    # Generate initial parameters
    for i in range(n_thetas):
        rand_params = np.random.uniform(-10, 10, N_layers)
        params_list.append(rand_params)
        thetas = np.append(thetas, np.random.uniform(16e-3, Max_theta_init))

    # # Define a callback to print cost every N iterations
    # def make_verbose_callback(eigvals, eigvecs, U_target, interval=10):
    #     counter = {'it': 0}
    #     def callback(xk):
    #         counter['it'] += 1
    #         if counter['it'] % interval == 0:
    #             cost = fidelity_cost(xk, eigvals, eigvecs, U_target)
    #             print(f"[Iteration {counter['it']}] cost = {cost:.6e}")
    #     return callback
    fidelity_trajectory = []  # Store fidelity values (1 - cost)

    # --------- cost wrapper with penalty -----------------------
    def wrapped_cost(x, eigvals, eigvecs, U_target, lam):
        base_cost = 1 + fidelity_cost(x, eigvals, eigvecs, U_target)
        penalty   = lam * (np.sum(x) ** 2)          # L2 on Σx
        total_cost = base_cost + penalty

        # Store fidelity (un-penalised)
        fidelity_trajectory.append(base_cost)
        return total_cost

    eigvals, eigvecs = precompute_hamiltonians(H_static, H_control, np.array(params_list))

    start_time = time.time()

    # Set the lower bounds for each parameter (same size as thetas)
    bounds = [(4e-3, None) for _ in thetas]  # No upper bound

    # Perform optimization using L-BFGS-B with bounds
    res = minimize(
        wrapped_cost,
        x0=thetas,
        args=(eigvals, eigvecs, U_target, lambda_penalty),
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'disp': True,
            # 'ftol': 1e-10,
            'maxiter': 10000000,
            'xatol': 1e-8,        # Tighter tolerance for convergence in parameters
            'fatol': 1e-8,        # Tighter tolerance for convergence in function value
            'adaptive': True      # Adaptive simplex size scaling (helps in higher dimensions)
        }
        
    )

    print("total evolution time", N_layers * np.sum(res.x))

    final_value = res.fun  # infidelity
    accuracy = final_value

    elapsed_time = time.time() - start_time

    print("optimization time", elapsed_time)
    print("Final infidelity:", fidelity_trajectory[-1])

    result = {
        'final_value': float(final_value),
        'accuracy': float(accuracy),
        'optimized_thetas': res.x.tolist(),
        'params': [p.tolist() for p in params_list],
        'elapsed_time': elapsed_time,
        'nfev': res.nfev,  # total number of function evaluations
        'njev': None,   # total number of gradient evaluations
        'fidelity_trajectory': fidelity_trajectory
    }

    return result



def make_serializable(obj):
    """
    Recursively convert numpy arrays and complex numbers for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}  # or str(obj) if easier
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(v) for v in obj)
    else:
        return obj
    

def save_result(result, output_dir, config_str, task_id, rank):
    """
    Save result to a file immediately after calculation.
    
    Args:
        result: Result dictionary to save
        output_dir: Directory to save results in
        config_str: Configuration string
        task_id: Task ID for this job
        rank: MPI rank
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save result as both JSON and pickle
    filename_base = f"{output_dir}/{config_str}_task{task_id}_rank{rank}"
    
    # Convert everything to Python built-ins
    serializable = make_serializable(result)

    with open(f"{filename_base}.json", "w") as f:
        json.dump(serializable, f, indent=2)
    with open(f"{filename_base}.pkl", "wb") as f:
        pickle.dump(result, f)

def main():
    parser = argparse.ArgumentParser(description='Run quantum optimization with MPI')
    parser.add_argument('--n', type=int, required=True, help='Number of qubits')
    parser.add_argument('--N_layers', type=int, required=True, help='Number of layers')
    parser.add_argument('--n_thetas', type=int, required=True, help='Number of theta parameters')
    parser.add_argument('--Max_theta_init', type=float, required=True, help='Maximum initial theta value')
    parser.add_argument('--energy_or_fid', type=str, required=True, choices=['energy', 'fidelity'], 
                        help='Whether to optimize for energy or fidelity')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get SLURM array task ID or use 0 if not in SLURM
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    # Create a configuration string
    config_str = f"n{args.n}_layers{args.N_layers}_thetas{args.n_thetas}_maxinit{args.Max_theta_init}_{args.energy_or_fid}"
    
    # Set random seed based on rank and task_id for reproducibility but different across runs
    np.random.seed(1000 * task_id + rank)
    
    # Generate random problem
    problem = generate_random_problem(args.n)
    
    # Run optimization
    result = run_optimization(
        problem, 
        args.n, 
        args.N_layers, 
        args.n_thetas, 
        args.Max_theta_init, 
        lambda_penalty=0.0
    )
    
    # embed the problem into the result dict
    result['problem'] = problem
    
    # Add metadata
    result['rank'] = rank
    result['task_id'] = task_id
    result['config'] = {
        'n': args.n,
        'N_layers': args.N_layers,
        'n_thetas': args.n_thetas,
        'Max_theta_init': args.Max_theta_init,
        'energy_or_fid': args.energy_or_fid
    }
    
    # Save result immediately
    save_result(result, args.output_dir, config_str, task_id, rank)
    
    # Synchronize before finishing
    # comm.Barrier()
    
    # if rank == 0:
    #     print(f"Task {task_id}: All {size} processes completed successfully")

if __name__ == "__main__":
    main()

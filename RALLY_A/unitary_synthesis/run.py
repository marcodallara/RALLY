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


def ansatz(
    thetas: np.ndarray,
    params: np.ndarray,
    H_static: np.ndarray,
    H_control: np.ndarray,
    n_qubits: int,
    dt: float,
) -> np.ndarray:
    """
    Build the unitary from the variational ansatz.

    thetas: shape (n_thetas,)
    params: shape (n_thetas, n_layers)
    """
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=complex)

    n_thetas, n_layers = params.shape

    for layer_idx, theta in enumerate(thetas):
        for j in range(n_layers):
            H_eff = theta * params[layer_idx, j] * H_control + H_static
            U_layer = expm(-1j * dt * H_eff)
            U = U_layer @ U

    return U


def unitary_fidelity(U_target: np.ndarray, U_actual: np.ndarray) -> float:
    """
    Process fidelity:
        F = |Tr(U_target^† U_actual)|^2 / d^2
    """
    if U_target.shape != U_actual.shape:
        raise ValueError("Unitary matrices must have the same dimensions.")
    d = U_target.shape[0]
    overlap = np.trace(U_target.conj().T @ U_actual)
    return np.abs(overlap) ** 2 / (d ** 2)


def fidelity_cost(
    thetas: np.ndarray,
    params: np.ndarray,
    U_target: np.ndarray,
    n_qubits: int,
    dt: float,
    H_static: np.ndarray,
    H_control: np.ndarray,
) -> float:
    """
    Return infidelity = 1 - F.
    """
    U_ans = ansatz(thetas, params, H_static, H_control, n_qubits, dt)
    F = unitary_fidelity(U_target, U_ans)
    return 1.0 - F


def run_optimization(
    problem: dict,
    n: int,
    N_layers: int,
    n_thetas: int,
    Max_theta_init: float,
    lambda_penalty: float = 0,
) -> dict:
    """
    Set up and run the classical optimization.
    problem: must contain 'U_t' (target unitary).
    """

    n_qubits = n

    # Unpack problem
    U_target = problem["U_t"]

    # Construct Hamiltonians
    H_static = generate_static_H(n_qubits).full()
    H_control = generate_control_H(n_qubits).full()

    # Random parameters: shape (n_thetas, N_layers)
    params = np.random.uniform(-10.0, 10.0, size=(n_thetas, N_layers))

    # Initial thetas
    thetas0 = np.random.uniform(16e-3, Max_theta_init, size=n_thetas)

    # Time-step
    dt = 500.0 / (n_thetas * N_layers)

    fidelity_trajectory = []

    def cost_wrapper(x: np.ndarray) -> float:
        # Base infidelity
        base_cost = fidelity_cost(
            x, params, U_target, n_qubits, dt, H_static, H_control
        )
        # L2 penalty on angles
        penalty = lambda_penalty * np.sum(x ** 2)
        total_cost = base_cost + penalty

        # Store fidelity (1 - infidelity, without penalty)
        fidelity_trajectory.append(1.0 - base_cost)
        return total_cost

    # Bounds on thetas
    bounds = [(4e-3, None) for _ in range(n_thetas)]

    start_time = time.time()

    res = minimize(
        cost_wrapper,
        x0=thetas0,
        method="Nelder-Mead",  # supports bounds
        bounds=bounds,
        options={
            'maxiter': 10000000,
            'xatol': 1e-8,        # Tighter tolerance for convergence in parameters
            'fatol': 1e-8,        # Tighter tolerance for convergence in function value
            'adaptive': True      # Adaptive simplex size scaling (helps in higher dimensions)
        },
    )

    elapsed_time = time.time() - start_time

    final_cost = float(res.fun)
    final_fidelity = fidelity_trajectory[-1] if fidelity_trajectory else None

    print("optimization time:", elapsed_time)
    print("Final fidelity:", final_fidelity)

    result = {
        "final_value": final_cost,          # infidelity + penalty
        "accuracy": final_cost,             # kept same as your original code
        "optimized_thetas": res.x.tolist(),
        "params": params.tolist(),
        "elapsed_time": elapsed_time,
        "nfev": res.nfev,
        "njev": getattr(res, "njev", None),
        "fidelity_trajectory": fidelity_trajectory,
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

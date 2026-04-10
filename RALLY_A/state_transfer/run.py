import os
import sys
import json
import time
import pickle
import numpy as np
from mpi4py import MPI
from scipy.optimize import minimize
from scipy.linalg import expm
import argparse
from scipy.linalg import eigh
import scipy.sparse        as sp
from scipy.linalg import expm_frechet

from system import generate_static_H
from system import generate_control_H
from system import generate_random_problem


last_cost = {'val': None}


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

def fidelity_and_grad(theta_vec,
                      H_static, H_control, params,
                      psi_0, psi_target,
                      n_qubits, dt, N_layers,
                      alpha=0.0):
    """
    Cost and gradient for the ansatz, reusing expm/expm_frechet per (theta_k, amp).

    theta_vec: (n_thetas,)
    params   : (n_thetas, N_layers) with entries in a small set (e.g. {-1, +1})
    """
    psi_0 = np.asarray(psi_0, dtype=complex)
    psi_target = np.asarray(psi_target, dtype=complex)

    n_thetas, n_layers = params.shape
    assert n_thetas == theta_vec.shape[0]
    assert n_layers == N_layers

    S = n_thetas * n_layers               # total segments
    dim = psi_0.shape[0]

    # Precompute scaled Hamiltonians
    A_static  = -1j * dt * H_static
    A_control = -1j * dt * H_control

    # Storage for states and segment metadata
    forward_states  = np.empty((S + 1, dim), dtype=complex)
    backward_states = np.empty((S + 1, dim), dtype=complex)
    seg_theta = np.empty(S, dtype=np.int32)          # which theta_k used in segment s
    seg_amp   = np.empty(S, dtype=params.dtype)      # which amplitude was used in segment s

    forward_states[0] = psi_0

    # Caches: for each k, map amp -> A, U, dU
    A_cache  = [dict() for _ in range(n_thetas)]
    U_cache  = [dict() for _ in range(n_thetas)]
    dU_cache = [dict() for _ in range(n_thetas)]

    # ----------------------- Forward propagation -----------------------
    s = 0
    for k in range(n_thetas):
        theta_k = theta_vec[k]

        # Build expm cache for each distinct amplitude in this row
        unique_amps = np.unique(params[k, :])
        for amp in unique_amps:
            A = A_static + theta_k * amp * A_control            # A_s = -i dt H_eff
            U = expm(A)

            A_cache[k][amp] = A
            U_cache[k][amp] = U

        # Use cached U's to propagate
        for j in range(n_layers):
            amp = params[k, j]
            U   = U_cache[k][amp]

            forward_states[s + 1] = U @ forward_states[s]
            seg_theta[s] = k
            seg_amp[s]   = amp
            s += 1

    psi_final = forward_states[S]
    overlap   = np.vdot(psi_target, psi_final)
    F         = np.abs(overlap) ** 2

    # ----------------------- Backward propagation ----------------------
    backward_states[S] = psi_target
    for s in range(S - 1, -1, -1):
        k   = seg_theta[s]
        amp = seg_amp[s]
        U   = U_cache[k][amp]

        backward_states[s] = U.conj().T @ backward_states[s + 1]

    # ----------------------- Gradient of F -----------------------------
    grad_F = np.zeros_like(theta_vec, dtype=float)

    # Precompute Frechet derivatives dU/dθ_k once per (k, amp)
    for k in range(n_thetas):
        for amp, A in A_cache[k].items():
            # A(θ_k) = A_static + θ_k amp A_control
            # dA/dθ_k = amp * A_control  (already includes -i dt)
            E = amp * A_control
            dU_cache[k][amp] = expm_frechet(A, E, compute_expm=False)

    # Accumulate gradient contributions
    for s in range(S):
        k   = seg_theta[s]
        amp = seg_amp[s]
        dU  = dU_cache[k][amp]

        vec        = dU @ forward_states[s]
        d_overlap  = np.vdot(backward_states[s + 1], vec)
        grad_F[k] += 2.0 * np.real(np.conj(overlap) * d_overlap)

    # Optional L1 penalty on pulse area
    penalty      = alpha * np.sum(theta_vec) * N_layers
    grad_penalty = alpha * N_layers * np.ones_like(theta_vec)

    last_cost['val'] = 1.0 - F
    cost = -F + penalty
    grad = -grad_F + grad_penalty
    return cost, grad

# ---------------------------------------------------------------------------
# run_state_transfer_optimization_sparse_diag
# ---------------------------------------------------------------------------
def run_state_transfer_optimization_sparse_diag(problem,
                                                n,
                                                N_layers,
                                                n_thetas,
                                                Max_theta_init,
                                                dt,
                                                r_values=(-1, 1),
                                                alpha=0.0):
    """
    Driver that optimises state-transfer fidelity while diagonalising each
    distinct control amplitude in `r_values` only once.

    Returns
    -------
    dict with final cost, θ*, etc.
    """

    # ────────────────────── unpack problem description ──────────────────────
    psi_0      = problem['psi_0']
    psi_target = problem['psi_t']
    coeffs     = problem['coeffs']

    # ─────────────────────── construct dense Hamiltonians ───────────────────
    H_static  = np.asarray(generate_static_H(n, coeffs), dtype=complex)
    H_control = np.asarray(generate_control_H(n), dtype=complex)

    # ────────────────────── initialise parameters & schedule ─────────────────
    params  = np.random.choice(r_values, size=(n_thetas, N_layers))
    thetas  = np.random.uniform(1e-9, Max_theta_init, size=n_thetas)

    fidelity_trajectory = []  # will store 1 - F (to match your original logic)

    def wrapped_fidelity_and_grad(x, *args):
        cost, grad = fidelity_and_grad(x, *args)
        fidelity = 1.0 + cost  # since cost = -F + penalty, with alpha=0 this is 1 - F
        fidelity_trajectory.append(fidelity)
        return cost, grad

    # ─────────────────────── early-stopping callback ────────────────────────
    best_x = {'val': thetas.copy()}

    def stop_if_good_enough(xk, thr=1e-3):
        best_x['val'] = xk.copy()
        if last_cost['val'] is not None and last_cost['val'] < thr:
            print(f"Early stop: cost {last_cost['val']:.3e} < {thr}")
            raise StopIteration

   # ─────────────────────────── optimisation ───────────────────────────────
    bounds = [(1e-9, None)] * n_thetas
    t_opt0 = time.perf_counter()

    result = minimize(
        wrapped_fidelity_and_grad,
        thetas,
        args=(H_static, H_control, params, psi_0, psi_target, n, dt, N_layers),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        # callback=stop_if_good_enough,
        options={'disp': True, 'maxiter': 150000},
    )

    opt_x = result.x
    nfev = result.nfev

    elapsed = time.perf_counter() - t_opt0
    print(f"⏱  optimisation finished in {elapsed:.3f} s")
    print(f"⏱  total evolution time {float(np.sum(opt_x) * N_layers * dt):.3f} s")

    # ───────────────────── compute final fidelity ───────────────────────────
    U_final   = ansatz(opt_x, params, H_static, H_control, n, dt)
    psi_final = U_final @ psi_0
    fidelity  = np.abs(np.vdot(psi_target, psi_final)) ** 2

    return {
        'final_value'        : float(1 - fidelity),
        'accuracy'           : float(1 - fidelity),
        'optimized_thetas'   : opt_x.tolist(),
        'params'             : params.tolist(),
        'total_time'         : float(np.sum(opt_x) * N_layers * dt),
        'elapsed_time'       : elapsed,
        'nfev'               : nfev,
        'fidelity_trajectory': fidelity_trajectory,
    }

def make_serializable(obj):
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
    
    T = 3000
    dt = T / (args.n_thetas * args.N_layers)  # total time divided by total number of segments

    # Generate random problem
    problem = generate_random_problem(args.n, args.energy_or_fid, seed=task_id)
    # Run optimization
    result = run_state_transfer_optimization_sparse_diag(
        problem,
        args.n,
        args.N_layers,
        args.n_thetas,
        args.Max_theta_init,
        dt,
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

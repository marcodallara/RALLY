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

from system import generate_static_H
from system import generate_control_H
from system import generate_random_problem


# ───────────────────────── helpers ──────────────────────────────────────────
def _apply_evolution(psi, U, U_dag, phases):
    """|ψ’⟩ = U · diag(phases) · U† |ψ⟩   (2 mat-vecs, no expm)."""
    tmp = U_dag @ psi          # go to eigen-basis
    tmp *= phases              # phase kick
    return U @ tmp             # back to computational basis




# ──────────────────────── spectral cache  (only |r_values| items) ───────────
def precompute_unique_hamiltonians(H_static_csr, H_control_csr, r_values=(0, 1)):
    """
    Return {r_val: (eigvals, eigvecs, eigvecs_dag)} for every r_val in r_values.
    Only len(r_values) diagonalisations are carried out.
    """
    cache = {}
    for r_val in r_values:
        H_eff          = H_static_csr + r_val * H_control_csr
        lam, vec       = eigh(H_eff.toarray())          # dense diagonalisation
        cache[r_val]   = (lam,
                          vec,
                          vec.conj().T)                 # U†  (pre-stored)
    return cache

# ───────────────────────── forward pass  (look-up into the cache) ───────────
def ansatz_optimized_state_transfer(thetas, eig_cache, r, psi_0):
    """
    eig_cache : dict  {r_val : (eigvals, eigvecs, eigvecs_dag)}
    r         : (K, L) array with entries in eig_cache.keys()
    """
    K, L = r.shape
    psi  = psi_0
    for k in range(K):
        for i in range(L):
            lam, U, U_dag = eig_cache[r[k, i]]
            phases        = np.exp(-1j * thetas[k] * lam)
            psi           = _apply_evolution(psi, U, U_dag, phases)
    return psi



# ─────────────────────────── gradient  ──────────────────────────
def compute_gradient_forward_backward(
        eig_cache, H_static_csr, H_control_csr, r, theta,
        psi_0, psi_target):

    K, L = r.shape
    total = K * L
    grad  = np.zeros(K, dtype=float)

    # ----- forward -----------------------------------------------------------
    forward = [psi_0]
    for k in range(K):
        for i in range(L):
            lam, U, U_dag = eig_cache[r[k, i]]
            phases        = np.exp(-1j * theta[k] * lam)
            forward.append(_apply_evolution(forward[-1], U, U_dag, phases))

    psi_final = forward[-1]
    overlap   = np.vdot(psi_target, psi_final)

    # ----- backward ----------------------------------------------------------
    backward = [None]*(total + 1)
    backward[-1] = psi_target
    for idx in reversed(range(total)):
        k, i         = divmod(idx, L)
        lam, U, U_dag = eig_cache[r[k, i]]
        phases        = np.exp(-1j * theta[k] * lam)
        backward[idx] = _apply_evolution(backward[idx+1], U, U_dag, phases.conj())

    # ----- gradient ----------------------------------------------------------
    for k in range(K):
        accum = 0j
        for i in range(L):
            idx       = k*L + i
            psi_less  = forward [idx+1]
            psi_more  = backward[idx+1]

            hpsi = (H_static_csr  @ psi_less) + \
                   (r[k, i] * (H_control_csr @ psi_less))

            accum += np.vdot(psi_more, -1j * hpsi)

        grad[k] = 2 * np.real(np.conj(overlap) * accum)
    return grad


last_cost = {'val': None}                    # tiny global cache


# ---------------------------------------------------------------------------
# run_state_transfer_optimization_sparse_diag
# ---------------------------------------------------------------------------
def run_state_transfer_optimization_sparse_diag(problem,
                                                n,                 # system size
                                                N_layers,          # L  (layers per θ_k)
                                                n_thetas,          # K  (independent θ_k)
                                                Max_theta_init,    # init range (0 … Max_theta_init)
                                                r_values=(-1, 1),   # allowed r entries
                                                alpha=0.0):        # L1-penalty weight
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

    # ─────────────────────── construct sparse Hamiltonians ───────────────────
    H_static_csr  = sp.csr_matrix(generate_static_H(n, coeffs))
    H_control_csr = sp.csr_matrix(generate_control_H(n))

    # ────────────────────── initialise parameters & schedule ─────────────────
    r       = np.random.choice(r_values, size=(n_thetas, N_layers))
    thetas  = np.random.uniform(1e-9, Max_theta_init, size=n_thetas)

    # ────────────────────── pre-diagonalise unique Hamiltonians ──────────────
    t_diag0   = time.perf_counter()
    eig_cache = precompute_unique_hamiltonians(H_static_csr,
                                               H_control_csr,
                                               r_values=r_values)
    print(f"⏱  pre-diagonalisation took "
          f"{time.perf_counter() - t_diag0:.3f} s")

    # ─────────────────────────── cost & gradient ────────────────────────────
    def fidelity_and_grad(theta_vec,
                          eig_cache, H_static_csr, H_control_csr, r,
                          psi_0, psi_target, N_layers, alpha=alpha):

        psi_final = ansatz_optimized_state_transfer(theta_vec,
                                                    eig_cache, r, psi_0)
        F         = np.abs(np.vdot(psi_target, psi_final))**2

        grad_F    = compute_gradient_forward_backward(
                        eig_cache,
                        H_static_csr, H_control_csr, r,
                        theta_vec, psi_0, psi_target)

        # optional L1 penalty on total “pulse area”
        penalty      = alpha * np.sum(theta_vec) * N_layers
        grad_penalty = alpha * N_layers * np.ones_like(theta_vec)

        last_cost['val'] = 1.0 - F
        return -F + penalty, -grad_F + grad_penalty

    # ─────────────────────── early-stopping callback ────────────────────────
    best_x = {'val': thetas.copy()}           # keep most recent iterate

    def stop_if_good_enough(xk, thr=1e-3):
        best_x['val'] = xk.copy()
        if last_cost['val'] is not None and last_cost['val'] < thr:
            print(f"Early stop: cost {last_cost['val']:.3e} < {thr}")
            raise StopIteration

    fidelity_trajectory = []  # Store fidelity values (1 - cost)

    def wrapped_fidelity_and_grad(x, *args):
        cost, grad = fidelity_and_grad(x, *args)
        fidelity = 1 + cost  # Since cost = -fidelity
        fidelity_trajectory.append(fidelity)
        return cost, grad

    # ─────────────────────────── optimisation ───────────────────────────────
    bounds  = [(1e-9, None)] * n_thetas
    t_opt0  = time.perf_counter()

    result = minimize(
        wrapped_fidelity_and_grad, thetas,
        args=(eig_cache, H_static_csr, H_control_csr, r,
                psi_0, psi_target, N_layers),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        # callback=stop_if_good_enough,
    )
    opt_x  = result.x
    nfev   = result.nfev

    elapsed = time.perf_counter() - t_opt0
    print(f"⏱  optimisation finished in {elapsed:.3f} s")
    print(f"⏱  total evolution time {float(np.sum(opt_x) * N_layers):.3f} s")
    # ───────────────────── compute final fidelity ───────────────────────────
    psi_final = ansatz_optimized_state_transfer(opt_x, eig_cache, r, psi_0)
    fidelity  = np.abs(np.vdot(psi_target, psi_final))**2

    return {
        'final_value'     : float(1 - fidelity),
        'accuracy'        : float(1 - fidelity),
        'optimized_thetas': opt_x.tolist(),
        'params'          : r.tolist(),
        'total_time'      : float(np.sum(opt_x) * N_layers),
        'elapsed_time'    : elapsed,
        'nfev'            : nfev,
        'fidelity_trajectory': fidelity_trajectory
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
    
    # Generate random problem
    problem = generate_random_problem(args.n, args.energy_or_fid, seed=task_id)
    # Run optimization
    result = run_state_transfer_optimization_sparse_diag(
        problem, 
        args.n, 
        args.N_layers, 
        args.n_thetas, 
        args.Max_theta_init 
        # args.energy_or_fid
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

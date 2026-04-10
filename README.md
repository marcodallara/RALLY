# RALLY

Code for RALLY quantum optimal control algorithm:

- https://arxiv.org/abs/2603.08948

## Structure

```text
RALLY/
├── RALLY_T/
│   ├── state_transfer/      # time-based optimization variant
│   └── unitary_synthesis/   # unitary compilation experiments
├── RALLY_A/
│   ├── state_transfer/      # amplitude-based optimization variant
│   └── unitary_synthesis/   # unitary compilation experiments
└── Hamiltonians/            # shared Hamiltonians
```

Each experiment folder contains:

- `run.py`: main executable script
- `system.py`: system/Hamiltonian and problem definitions
- `script.sh`: SLURM launcher (optional; not required for local runs)

## Example

From an experiment folder, run `run.py` directly.

```bash
cd RALLY/RALLY_T/state_transfer
mkdir -p results

mpirun -np 1 python run.py \
  --n 6 \
  --N_layers 5 \
  --n_thetas 140 \
  --Max_theta_init 5 \
  --energy_or_fid fidelity \
  --output_dir results
```

`python run.py ...` also works in many setups.

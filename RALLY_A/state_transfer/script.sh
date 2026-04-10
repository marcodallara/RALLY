#!/bin/bash
#SBATCH --job-name=RALLY
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --ntasks=1                 # One task per job
#SBATCH --cpus-per-task=4        # 4 CPUs per task
#SBATCH --mem-per-cpu=2GB          
#SBATCH --time=9-23:00:00
#SBATCH --account=qi
#SBATCH --qos=normal
#SBATCH --array=0-49%50             # Update range as needed (0 to N-1 for N configs)


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs
mkdir -p results


# Initialize empty arrays
qubit_sizes=()
layers=()
thetas_counts=()
max_inits=()
optimization_types=()


for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(10)
  max_inits+=(5)
  optimization_types+=("fidelity")
done

for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(12)
  max_inits+=(5)
  optimization_types+=("fidelity")
done

for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(14)
  max_inits+=(5)
  optimization_types+=("fidelity")
done

for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(16)
  max_inits+=(5)
  optimization_types+=("fidelity")
done

for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(18)
  max_inits+=(5)
  optimization_types+=("fidelity")
done


idx=$SLURM_ARRAY_TASK_ID

n_qubits=${qubit_sizes[$idx]}
n_layers=${layers[$idx]}
n_thetas=${thetas_counts[$idx]}
max_theta_init=${max_inits[$idx]}
energy_or_fid=${optimization_types[$idx]}

echo "Running configuration: n=$n_qubits, N_layers=$n_layers, n_thetas=$n_thetas, Max_theta_init=$max_theta_init, energy_or_fid=$energy_or_fid"

source /home/dallara/.bashrc
conda activate scipy_env

mpirun python run.py \
    --n "$n_qubits" \
    --N_layers "$n_layers" \
    --n_thetas "$n_thetas" \
    --Max_theta_init "$max_theta_init" \
    --energy_or_fid "$energy_or_fid" \
    --output_dir results

exit 0

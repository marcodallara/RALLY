#!/bin/bash
#SBATCH --job-name=RALLY
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --ntasks=1                 # One task per job
#SBATCH --cpus-per-task=4        # 8 CPUs per task
#SBATCH --mem-per-cpu=1G          
#SBATCH --time=9-23:00:00
#SBATCH --account=qi
#SBATCH --qos=normal
#SBATCH --array=0-39%40            # Update range as needed (0 to N-1 for N configs)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Create logs/results directories if they don't exist
mkdir -p logs
mkdir -p results


# Initialize empty arrays
qubit_sizes=()
layers=()
thetas_counts=()
max_inits=()

optimization_types=()


# Fill each array with 10 copies of the desired value
for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(60)
  max_inits+=(1) #it's amplitude rescaling now
  optimization_types+=("fidelity")
done

# Fill each array with 10 copies of the desired value
for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(62)
  max_inits+=(1) #it's amplitude rescaling now
  optimization_types+=("fidelity")
done

# Fill each array with 10 copies of the desired value
for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(66)
  max_inits+=(1) #it's amplitude rescaling now
  optimization_types+=("fidelity")
done

# Fill each array with 10 copies of the desired value
for _ in {1..10}; do
  qubit_sizes+=(3)
  layers+=(5)
  thetas_counts+=(68)
  max_inits+=(1) #it's amplitude rescaling now
  optimization_types+=("fidelity")
done

# Get index from array task ID
idx=$SLURM_ARRAY_TASK_ID

# Extract configuration for this array job
n_qubits=${qubit_sizes[$idx]}
n_layers=${layers[$idx]}
n_thetas=${thetas_counts[$idx]}
max_theta_init=${max_inits[$idx]}
energy_or_fid=${optimization_types[$idx]}

echo "Running configuration: n=$n_qubits, N_layers=$n_layers, n_thetas=$n_thetas, Max_theta_init=$max_theta_init, energy_or_fid=$energy_or_fid"

# Load environment
source /home/dallara/.bashrc
conda activate scipy_env

# Run the Python script
mpirun python run.py \
    --n $n_qubits \
    --N_layers $n_layers \
    --n_thetas $n_thetas \
    --Max_theta_init $max_theta_init \
    --energy_or_fid $energy_or_fid \
    --output_dir results

exit 0

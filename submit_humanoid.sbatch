#!/bin/bash
#SBATCH -J rl-training
#SBATCH -c 1
#SBATCH -t 0-72:00:00
#SBATCH -p mweber_gpu
#SBATCH --mem=8000
#SBATCH --open-mode=append
#SBATCH -o /n/home09/annabelma/rl_final_proj/logs_remaining_humanoid/rl_%A_%a.out
#SBATCH -e /n/home09/annabelma/rl_final_proj/logs_remaining_humanoid/rl_%A_%a.err
#SBATCH --array=1-605%30
#SBATCH --gres=gpu:1

# PARALLEL EXECUTION: This script runs multiple jobs in parallel!
# The --array=1-N%M means:
#   - Run jobs 1 through N (total jobs)
#   - %M means run at most M jobs concurrently
# Example: --array=1-100%20 runs 100 jobs with max 20 running at once
#
# To update: Run ./prepare_submit.sh which auto-calculates the array size
# Or manually: wc -l joblist.txt to get N, then update --array=1-N%M

# test with python train_rl.py --task "Hopper-v5" --algorithm "SAC" --seed 0

set -euo pipefail
# Use SLURM_SUBMIT_DIR which is set to the directory where you ran sbatch
# Fall back to script location if SLURM_SUBMIT_DIR is not set
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
# Create logs directory if it doesn't exist (ignore errors if it already exists)
LOGS_DIR="$SCRIPT_DIR/logs"
if [[ ! -d "$LOGS_DIR" ]]; then
    mkdir -p "$LOGS_DIR" 2>/dev/null || {
        echo "WARNING: Could not create logs directory at $LOGS_DIR, but continuing..."
    }
fi

# Load Python module
module load python/3.10.9-fasrc01

# Activate virtual environment
VENV_DIR="$SCRIPT_DIR/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run ./setup_venv.sh first to create the virtual environment"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Set environment variables
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1

# CUDA settings for better error reporting and GPU isolation
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU allocation: SLURM should automatically set CUDA_VISIBLE_DEVICES with --gres=gpu:1
# DO NOT override CUDA_VISIBLE_DEVICES - let SLURM handle GPU isolation
# SLURM will make the allocated GPU appear as device 0 to your process
# If CUDA_VISIBLE_DEVICES is not set, that's a SLURM configuration issue

# Debug: print GPU info
echo "=== GPU Allocation Info ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-not set}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"
echo "SLURM_LOCALID=${SLURM_LOCALID:-not set}"
echo "SLURM_NODEID=${SLURM_NODEID:-not set}"
echo "SLURM_PROCID=${SLURM_PROCID:-not set}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-not set}"
echo "SLURM_GPUS=${SLURM_GPUS:-not set}"
if command -v nvidia-smi &> /dev/null; then
    echo "--- nvidia-smi output (all GPUs on node) ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo "--- Processes using GPUs ---"
    nvidia-smi pmon -c 1 2>/dev/null || echo "nvidia-smi pmon not available"
fi
echo "==========================="

# Job list file - using joblist_humanoid2.txt (missing seeds 5-54)
JOBLIST="${JOBLIST:-$SCRIPT_DIR/joblist_humanoid2.txt}"
[[ -f "$JOBLIST" ]] || { echo "ERROR: $JOBLIST not found. Run script to generate joblist_humanoid2.txt first."; exit 1; }

# Get the line from joblist
NLINES=$(wc -l < "$JOBLIST")
LIDX=${SLURM_ARRAY_TASK_ID:?need array id}
(( LIDX>=1 && LIDX<=NLINES )) || { echo "Index $LIDX out of range (1-$NLINES)"; exit 0; }

# Read task, algorithm, seed from joblist
read -r TASK ALGO SEED < <(sed -n "${LIDX}p" "$JOBLIST")

echo "=== [START] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID ($LIDX/$NLINES) ==="
echo "    Task:      $TASK"
echo "    Algorithm: $ALGO"
echo "    Seed:      $SEED"
echo "------------------------------------------------------------"

# Run training
python -u train_rl.py \
    --task "$TASK" \
    --algorithm "$ALGO" \
    --seed "$SEED" \
    --eval-episodes 20

echo "=== [FINISH] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID ($LIDX/$NLINES) ==="


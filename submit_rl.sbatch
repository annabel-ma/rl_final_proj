#!/bin/bash
#SBATCH -J rl-training
#SBATCH -c 1
#SBATCH -t 0-72:00:00
#SBATCH -p mweber_gpu
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH -o logs/rl_%A_%a.out
#SBATCH -e logs/rl_%A_%a.err
#SBATCH --array=81-100
#SBATCH --account=dam_lab

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
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Load Python module
module load python/3.10.9-fasrc01

# Activate virtual environment
VENV_DIR="${SLURM_SUBMIT_DIR:-$PWD}/venv"
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

# Job list file
JOBLIST="${JOBLIST:-joblist.txt}"
[[ -f "$JOBLIST" ]] || { echo "ERROR: $JOBLIST not found. Run generate_joblist.py first."; exit 1; }

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


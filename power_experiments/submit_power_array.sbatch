#!/bin/bash
#SBATCH --job-name=power_task_array
#SBATCH --output=../logs/power/12_7_%A_%a.out
#SBATCH --error=../logs/power/12_7_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -p mweber_compute
#SBATCH --array=0-4

# Job array runs one task per array index
# Array indices: 0=Hopper-v5, 1=Walker2d-v5, 2=HalfCheetah-v5, 3=Ant-v5, 4=Humanoid-v5

# Task list
TASKS=(
    "Hopper-v5"
    "Walker2d-v5"
    "HalfCheetah-v5"
    "Ant-v5"
    "Humanoid-v5"
)

# Get task for this array job
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "Power Analysis Job Array"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Task: $TASK"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "============================================================"

# Optional: Set custom output directory via environment variable
# Uncomment and modify the line below if needed:

export POWER_OUTPUT_DIR=/n/home09/annabelma/rl_final_proj/power_experiments/results/12_8_6_pm

# Change to script directory
cd /n/home09/annabelma/rl_final_proj/power_experiments

# Run the analysis for this specific task with unbuffered output
python3 -u generate_power_task.py "$TASK"

echo ""
echo "============================================================"
echo "Power analysis completed for $TASK at $(date)"
echo "Results saved to: power_experiments/results/tasks/${TASK//\//_}"
echo "============================================================"

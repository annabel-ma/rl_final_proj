#!/bin/bash
# Helper script to prepare and submit SLURM job array
# Automatically calculates array size from joblist.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if joblist exists
JOBLIST="joblist.txt"
if [[ ! -f "$JOBLIST" ]]; then
    echo "ERROR: $JOBLIST not found. Generating it now..."
    python generate_joblist.py
fi

# Count jobs
NJOBS=$(wc -l < "$JOBLIST")
echo "Found $NJOBS jobs in $JOBLIST"

# Load config to get settings
if command -v python3 &> /dev/null; then
    CONFIG_VALS=$(python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    slurm = config.get('slurm', {})
    print(f\"{slurm.get('max_concurrent_jobs', 20)},{slurm.get('use_gpu', True)},{slurm.get('gpu_count', 1)}\")
" 2>/dev/null || echo "20,True,1")
    MAX_CONCURRENT=$(echo "$CONFIG_VALS" | cut -d',' -f1)
    USE_GPU=$(echo "$CONFIG_VALS" | cut -d',' -f2)
    GPU_COUNT=$(echo "$CONFIG_VALS" | cut -d',' -f3)
else
    MAX_CONCURRENT=20
    USE_GPU=True
    GPU_COUNT=1
fi

echo "Max concurrent jobs: $MAX_CONCURRENT"
echo "Use GPU: $USE_GPU"
if [ "$USE_GPU" = "True" ] || [ "$USE_GPU" = "true" ]; then
    echo "GPU count: $GPU_COUNT"
fi

# Create a temporary SLURM script with correct array size
TMP_SBATCH="submit_rl_auto.sbatch"
sed "s/--array=1-100%20/--array=1-${NJOBS}%${MAX_CONCURRENT}/" submit_rl.sbatch > "$TMP_SBATCH"

# Remove or add GPU flag based on config
if [ "$USE_GPU" = "False" ] || [ "$USE_GPU" = "false" ]; then
    # Remove GPU line if not using GPU
    sed -i '/#SBATCH --gres=gpu/d' "$TMP_SBATCH"
    echo "GPU disabled in config, using CPU"
else
    # Ensure GPU line exists with correct count
    if ! grep -q "#SBATCH --gres=gpu" "$TMP_SBATCH"; then
        sed -i '/#SBATCH --mem=/a #SBATCH --gres=gpu:'"$GPU_COUNT" "$TMP_SBATCH"
    else
        sed -i "s/#SBATCH --gres=gpu:[0-9]*/#SBATCH --gres=gpu:$GPU_COUNT/" "$TMP_SBATCH"
    fi
fi

echo ""
echo "Submitting SLURM job array:"
echo "  Total jobs: $NJOBS"
echo "  Concurrent: $MAX_CONCURRENT"
echo "  Array: 1-${NJOBS}%${MAX_CONCURRENT}"
echo ""

# Submit the job
sbatch "$TMP_SBATCH"

echo ""
echo "Job submitted! Check status with: squeue -u \$USER"
echo "Monitor logs in: logs/"


#!/bin/bash
#SBATCH --job-name=fpr
#SBATCH --output=logs/fpr/12_7_%j.out
#SBATCH --error=logs/fpr/12_7_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH -p mweber_compute

# Optional: Set custom output directory via environment variable
# Uncomment and modify the line below if needed:

export FPR_OUTPUT_DIR=/n/home09/annabelma/rl_final_proj/stat_results/12_7_results

# Load modules if needed (adjust for your system)
# module load python/3.9

# Change to script directory
cd /n/home09/annabelma/rl_final_proj

# Run the analysis with unbuffered output (so output appears immediately)
# Use -u flag or PYTHONUNBUFFERED=1 to disable Python output buffering
python3 -u generate_fpr.py

echo "Power analysis completed at $(date)"
echo "Results saved to: ${FPR_OUTPUT_DIR:-/n/home09/annabelma/rl_final_proj/stat_results/12_7_results}"

#!/usr/bin/env python3
"""
Generate Statistical Power analysis plots and results for a SINGLE TASK.

Usage:
    python generate_power_task.py TASK_NAME
    or set TASK environment variable

Output directory can be configured via environment variable:
    export POWER_OUTPUT_DIR=/path/to/output
    
Default: /n/home09/annabelma/rl_final_proj/stat_results/12_4_results

Outputs:
    - {TASK}_power_empirical_df.csv: Raw power results for this task
    - power_plots/: Directory containing individual power plots for this task
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, rankdata
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripts
import matplotlib.pyplot as plt

# ============================================================================
# Statistical tests from rl_stats (https://github.com/flowersteam/rl_stats)
# Re-implemented here to avoid bootstrapped dependency
# ============================================================================

tests_list = ['t-test', "Welch t-test", 'Mann-Whitney', 'Ranked t-test', 'bootstrap', 'permutation']


def run_permutation_test(all_data, n1, n2):
    """Helper for permutation test."""
    np.random.shuffle(all_data)
    data_a = all_data[:n1]
    data_b = all_data[-n2:]
    return data_a.mean() - data_b.mean()


def run_test(test_id, data1, data2, alpha=0.05):
    """
    Run statistical test comparing data1 and data2 (from rl_stats).
    
    Args:
        test_id: test name from tests_list
        data1, data2: sample arrays
        alpha: significance level
    
    Returns:
        bool: True if H0 is rejected (significant difference)
    """
    
    data1 = np.asarray(data1).squeeze()
    data2 = np.asarray(data2).squeeze()
    n1 = data1.size
    n2 = data2.size

    if test_id == 'bootstrap':
        # Simple bootstrap CI test (without bootstrapped package)
        n_boot = 1000
        diffs = []
        for _ in range(n_boot):
            s1 = np.random.choice(data1, size=n1, replace=True)
            s2 = np.random.choice(data2, size=n2, replace=True)
            diffs.append(np.mean(s1) - np.mean(s2))
        diffs = np.array(diffs)
        lo = np.percentile(diffs, 100 * alpha / 2)
        hi = np.percentile(diffs, 100 * (1 - alpha / 2))
        rejection = np.sign(lo) == np.sign(hi)  # 0 not in CI
        return rejection

    elif test_id == 't-test':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Precision loss.*')
            _, p = ttest_ind(data1, data2, equal_var=True)
        return p < alpha

    elif test_id == "Welch t-test":
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Precision loss.*')
            _, p = ttest_ind(data1, data2, equal_var=False)
        return p < alpha

    elif test_id == 'Mann-Whitney':
        # Handle case where data might be too similar
        try:
            _, p = mannwhitneyu(data1, data2, alternative='two-sided')
            return p < alpha
        except ValueError:
            # If data are too similar, return False (don't reject)
            return False

    elif test_id == 'Ranked t-test':
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        ranks = rankdata(all_data)
        ranks1 = ranks[:n1]
        ranks2 = ranks[n1:n1 + n2]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Precision loss.*')
            _, p = ttest_ind(ranks1, ranks2, equal_var=True)
        return p < alpha

    elif test_id == 'permutation':
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        delta = np.abs(data1.mean() - data2.mean())
        num_samples = 1000
        estimates = []
        for _ in range(num_samples):
            estimates.append(run_permutation_test(all_data.copy(), n1, n2))
        estimates = np.abs(np.array(estimates))
        diff_count = len(np.where(estimates <= delta)[0])
        return (1.0 - (float(diff_count) / float(num_samples))) < alpha

    else:
        raise NotImplementedError(f"Unknown test: {test_id}")


print(f"Statistical tests available: {tests_list}")

# ============================================================================
# Get task from command line or environment variable
# ============================================================================
if len(sys.argv) > 1:
    TASK = sys.argv[1]
elif 'TASK' in os.environ:
    TASK = os.environ['TASK']
else:
    print("Error: Must provide task name as argument or set TASK environment variable")
    print("Usage: python generate_power_task.py TASK_NAME")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"Running power analysis for task: {TASK}")
print(f"{'='*60}")

# ============================================================================
# Configuration: Output directory (configurable via environment variable)
# ============================================================================
# Get project root (parent of power_experiments)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)  # Go up from power_experiments to project root

# Default output directory is in power_experiments/results
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")

OUTPUT_DIR = os.getenv(
    'POWER_OUTPUT_DIR',
    DEFAULT_OUTPUT_DIR
)
# Create task-specific subdirectory
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tasks", TASK.replace('/', '_').replace('-', '_'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Statistical testing configuration
SEED_GRID = [2, 3, 5, 10, 20, 30, 50, 100]  # sample sizes per group for Power analysis 
ALPHAS = [0.05, 0.01]  # Significance levels
EPSILONS = [0.5, 1.0, 2.0]  # Cohen's d effect sizes for power analysis
N_RESAMPLES = 10000  # number of resamples for power (10^4 as per paper)

ALGORITHMS = ["SAC", "TD3", "DDPG", "PPO"]

BASE_DIR = os.path.join(BASE, "rl_experiments")
RESULTS_CSV = os.path.join(BASE_DIR, "final_eval_returns.csv")

GLOBAL_RNG_SEED = 31415
np.random.seed(GLOBAL_RNG_SEED)

print("\n" + "="*60)
print("Loading final evaluation returns...")
final_returns = pd.read_csv(RESULTS_CSV)
print(f"Loaded {len(final_returns)} entries")

# Check for duplicate (task, algorithm, seed) triples
duplicates = final_returns.groupby(['task', 'algorithm', 'seed']).filter(lambda x: len(x) > 1)
if len(duplicates) > 0:
    print(f"\nFound {len(duplicates)} rows with duplicate (task, algorithm, seed) triples")
    
    # Check if duplicates have same or different eval_return_mean
    different_returns = []
    same_returns_to_drop = []
    
    for (task, algo, seed), group in final_returns.groupby(['task', 'algorithm', 'seed']):
        if len(group) > 1:
            unique_returns = group['final_return_mean'].nunique()
            if unique_returns == 1:
                # Same return values - keep first, mark rest for dropping
                same_returns_to_drop.extend(group.index[1:].tolist())
            else:
                # Different return values - print them
                different_returns.append(group)
    
    if different_returns:
        print(f"\n*** WARNING: {len(different_returns)} (task, algo, seed) groups have DIFFERENT final_return_mean values: ***")
        diff_df = pd.concat(different_returns)
        display_cols = ['task', 'algorithm', 'seed', 'final_return_mean']
        if 'timestamp' in diff_df.columns:
            display_cols.append('timestamp')
        print(diff_df[display_cols].to_string())
    
    if same_returns_to_drop:
        print(f"\nDropping {len(same_returns_to_drop)} duplicate rows with same final_return_mean")
        final_returns = final_returns.drop(same_returns_to_drop).reset_index(drop=True)
        print(f"Final returns after deduplication: {len(final_returns)} entries")
else:
    print("No duplicate (task, algorithm, seed) triples found")


def recenter_samples(samples1, samples2, use_median=False):
    """
    Recenter samples so their means (or medians) are equal to 0.
    This creates a 'null world' where H0 is true: µ1 = µ2 = 0.
    Following the paper methodology.
    
    Args:
        samples1, samples2: arrays of samples
        use_median: if True, recenter around median; else use mean
                   (median for Mann-Whitney/ranked t-test, mean for others)
    
    Returns:
        recentered1, recentered2: samples with central tendency = 0
    """
    if use_median:
        center1 = np.median(samples1)
        center2 = np.median(samples2)
    else:
        center1 = np.mean(samples1)
        center2 = np.mean(samples2)
    
    # Align to 0: µ1 = µ2 = 0 (as per paper)
    recentered1 = samples1 - center1
    recentered2 = samples2 - center2
    
    return recentered1, recentered2


def estimate_power_empirical(empirical_samples1, empirical_samples2,
                            test_name, target_n, effect_size_epsilon,
                            alpha=0.05, n_resamples=10000, seed=None):
    """
    Estimate Statistical Power using empirical procedure:
    "If there is a true difference between algorithms A and B, 
    how often does a statistical test correctly reject H0?"
    
    Procedure:
    1. Generate return distributions for A, B and recenter
    2. For synthetic ε, shift one full distribution: X_B^(ε) = X_B + ε · σ_pool
    3. Draw N samples from X_A and X_B^(ε)
    4. Apply test at level α
    5. Repeat for R resamples
    6. Power = (# rejections) / R
    
    Args:
        empirical_samples1, empirical_samples2: actual data samples from algorithms A and B
        test_name: name of test from tests_list
        target_n: target seed budget (sample size)
        effect_size_epsilon: effect size ε ∈ {0.5, 1, 2} (Cohen's d)
        alpha: significance level
        n_resamples: number of resamples R (default 10^4 as per paper)
        seed: random seed
    
    Returns:
        power: empirical statistical power (1 - β*)
        se: standard error
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Determine whether to use median or mean based on test
    use_median = (test_name == 'Mann-Whitney' or test_name == 'Ranked t-test')
    
    # Step 1: Recenter distributions (same as FPR)
    recentered1, recentered2 = recenter_samples(empirical_samples1, empirical_samples2, use_median)
    
    # Step 2: Compute pooled standard deviation for shifting
    if use_median:
        # For median-based tests, use median absolute deviation (MAD) scaled
        mad1 = np.median(np.abs(recentered1 - np.median(recentered1)))
        mad2 = np.median(np.abs(recentered2 - np.median(recentered2)))
        sigma_pool = np.sqrt((mad1**2 + mad2**2) / 2) * 1.4826  # Scale MAD to approximate SD
    else:
        # For mean-based tests, use pooled standard deviation
        std1 = np.std(recentered1, ddof=1)
        std2 = np.std(recentered2, ddof=1)
        n1, n2 = len(recentered1), len(recentered2)
        sigma_pool = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
    
    # Shift distribution 2 by ε · σ_pool: X_B^(ε) = X_B + ε · σ_pool
    shifted2 = recentered2 + effect_size_epsilon * sigma_pool
    
    # Step 3: Sample and test
    true_positives = 0
    
    for _ in range(n_resamples):
        # Draw N samples (with replacement) from X_A and X_B^(ε)
        sample1 = np.random.choice(recentered1, size=target_n, replace=True)
        sample2 = np.random.choice(shifted2, size=target_n, replace=True)
        
        # Apply test at level α
        try:
            reject = run_test(test_name, sample1, sample2, alpha=alpha)
            if reject:
                true_positives += 1
        except:
            continue
    
    # Step 4: Empirical power = (# rejections) / R
    power = true_positives / n_resamples
    
    # Standard error
    se = np.sqrt(power * (1 - power) / n_resamples) if n_resamples > 0 else 0
    
    return power, se


# ============================================================================
# Main Power Analysis for Single Task
# ============================================================================

print(f"\nUsing empirical data from final_returns for task: {TASK}")
print(f"Testing across {len(SEED_GRID)} seed budgets: {SEED_GRID}")
print(f"Effect sizes (ε): {EPSILONS}")
print(f"Alpha levels: {ALPHAS}")
print(f"Resamples per estimate: {N_RESAMPLES}")

# Filter data for this task only
task_df = final_returns[final_returns['task'] == TASK]
if len(task_df) == 0:
    print(f"\nERROR: No data found for task '{TASK}'")
    print(f"Available tasks: {sorted(final_returns.task.dropna().unique())}")
    sys.exit(1)

available_algos = [a for a in ALGORITHMS if a in task_df['algorithm'].unique()]

if len(available_algos) < 2:
    print(f"\nERROR: Insufficient algorithms for task '{TASK}'")
    print(f"Available algorithms: {available_algos}")
    sys.exit(1)

print(f"\nAvailable algorithms for {TASK}: {available_algos}")

power_empirical_results = []

# Get empirical samples from actual data
print("\n" + "="*60)
print(f"Estimating Statistical Power using empirical procedure for {TASK}")
print("="*60)

# Iterate over all algorithm pairs for this task
for i, algo1 in enumerate(available_algos):
    for algo2 in available_algos[i+1:]:
        empirical1 = task_df[task_df['algorithm'] == algo1]['final_return_mean'].values
        empirical2 = task_df[task_df['algorithm'] == algo2]['final_return_mean'].values
        
        if len(empirical1) < 5 or len(empirical2) < 5:
            print(f"  Skipping {algo1} vs {algo2}: insufficient samples")
            continue
        
        print(f"  Using empirical samples: {algo1} (n={len(empirical1)}), {algo2} (n={len(empirical2)})")
        
        for alpha in ALPHAS:
            print(f"    Alpha = {alpha}")
            for epsilon in EPSILONS:
                print(f"      Effect size ε = {epsilon}")
                for target_n in SEED_GRID:
                    if target_n > min(len(empirical1), len(empirical2)):
                        continue
                    
                    print(f"        Target seed budget N = {target_n}")
                    for test_name in tests_list:
                        try:
                            power, se = estimate_power_empirical(
                                empirical1, empirical2,
                                test_name, target_n,
                                effect_size_epsilon=epsilon,
                                alpha=alpha,
                                n_resamples=min(N_RESAMPLES, 1000),  # Limit for faster execution
                                seed=None
                            )
                            power_empirical_results.append({
                                'task': TASK,
                                'algo1': algo1,
                                'algo2': algo2,
                                'test': test_name,
                                'alpha': alpha,
                                'epsilon': epsilon,
                                'target_n': target_n,
                                'power': power,
                                'se': se
                            })
                            print(f"          {test_name:20s}: Power = {power:.4f} ± {se:.4f}")
                        except Exception as e:
                            print(f"          {test_name:20s}: Error - {e}")

power_empirical_df = pd.DataFrame(power_empirical_results)

print("\n" + "="*60)
print(f"Empirical Power Results Summary for {TASK}")
print("="*60)
print("Following paper: Power estimated as proportion of H0 rejections over R resamples")
print("="*60)

if len(power_empirical_df) > 0:
    print(f"\nTotal results: {len(power_empirical_df)} rows")
    
    # Average power across all algorithm pairs for this task
    print(f"\nAverage Power by test and effect size (for {TASK}):")
    avg_power = power_empirical_df.groupby(['test', 'epsilon', 'alpha']).agg({
        'power': 'mean',
        'se': 'mean'
    })
    for (test, epsilon, alpha), row in avg_power.iterrows():
        power_val = row['power']
        se_val = row['se']
        print(f"  {test:20s} (ε={epsilon:.1f}, α={alpha:.3f}): Power = {power_val:.4f} ± {se_val:.4f}")
    
    # ============================================================================
    # Generate Individual Power Plots for This Task
    # Subplot layout: 3 rows (epsilon: 0.5, 1.0, 2.0) × 2 columns (alpha: 0.01, 0.05)
    # ============================================================================
    print("\n" + "="*60)
    print(f"Generating individual power plots for {TASK}")
    print("Subplot layout: 3 rows (ε) × 2 columns (α)")
    print("="*60)
    
    # Create output subdirectory for individual plots
    plot_dir = os.path.join(OUTPUT_DIR, "power_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get all unique (algo1, algo2) combinations for this task
    unique_pairs = power_empirical_df.groupby(['algo1', 'algo2']).size().reset_index()
    print(f"\nFound {len(unique_pairs)} unique algorithm pairs for {TASK}")
    
    # Color map for tests (matching paper style)
    test_colors = {
        't-test': '#1f77b4',           # blue
        'Welch t-test': '#ff7f0e',     # orange
        'Mann-Whitney': '#2ca02c',     # green
        'Ranked t-test': '#9467bd',    # purple
        'bootstrap': '#17becf',        # cyan
        'permutation': '#bcbd22'       # yellow-green
    }
    
    # Alpha and epsilon order
    alpha_order = [0.01, 0.05]  # Columns: left to right
    epsilon_order = [0.5, 1.0, 2.0]  # Rows: top to bottom
    
    plots_created = 0
    
    # Generate one figure per (algo1, algo2) pair for this task
    for idx, row in unique_pairs.iterrows():
        algo1 = row['algo1']
        algo2 = row['algo2']
        
        # Filter data for this pair
        pair_data = power_empirical_df[
            (power_empirical_df['algo1'] == algo1) &
            (power_empirical_df['algo2'] == algo2)
        ]
        
        if len(pair_data) == 0:
            continue
        
        # Create figure with subplots: 3 rows × 2 columns
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Power vs Sample Size: {TASK}\n{algo1} vs {algo2}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Iterate over rows (epsilon) and columns (alpha)
        for row_idx, epsilon in enumerate(epsilon_order):
            for col_idx, alpha in enumerate(alpha_order):
                ax = axes[row_idx, col_idx]
                
                # Filter data for this (alpha, epsilon) combination
                combo_data = pair_data[
                    (pair_data['alpha'] == alpha) &
                    (pair_data['epsilon'] == epsilon)
                ]
                
                if len(combo_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'α = {alpha:.2f}, ε = {epsilon:.1f}', fontsize=11)
                    continue
                
                # Group by test and target_n
                plot_data = combo_data.groupby(['test', 'target_n']).agg({
                    'power': 'mean',
                    'se': 'mean'
                }).reset_index()
                
                if len(plot_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'α = {alpha:.2f}, ε = {epsilon:.1f}', fontsize=11)
                    continue
                
                # Plot each test
                for test_name in tests_list:
                    test_data = plot_data[plot_data['test'] == test_name].sort_values('target_n')
                    if len(test_data) > 0:
                        color = test_colors.get(test_name, '#000000')
                        ax.plot(test_data['target_n'], test_data['power'], 
                               marker='o', label=test_name, linewidth=1.5, color=color, markersize=4)
                        # Add error bars
                        ax.errorbar(test_data['target_n'], test_data['power'], 
                                   yerr=test_data['se'], 
                                   fmt='none', color=color, alpha=0.3, capsize=2)
                
                # Add reference line at 0.8 power (target)
                ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, 
                          alpha=0.7, zorder=0)
                
                # Formatting
                if row_idx == 2:  # Bottom row
                    ax.set_xlabel('Sample size N (log scale)', fontsize=10)
                if col_idx == 0:  # Left column
                    ax.set_ylabel('Power (1 - β*)', fontsize=10)
                
                ax.set_title(f'α = {alpha:.2f}, ε = {epsilon:.1f}', fontsize=11, fontweight='bold')
                ax.set_xscale('log')
                
                # Set x-axis ticks
                available_n = sorted(plot_data['target_n'].unique())
                ax.set_xticks(available_n)
                ax.set_xticklabels([str(int(n)) for n in available_n], fontsize=9)
                
                # Y-axis: show from 0 to 1
                ax.set_ylim([0, 1.05])
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.tick_params(axis='y', labelsize=9)
                
                ax.grid(True, alpha=0.3, which='both')
                
                # Add legend only to top-right subplot
                if row_idx == 0 and col_idx == 1:
                    ax.legend(loc='upper left', frameon=True, fontsize=8, ncol=2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle
        
        # Save plot
        safe_task = TASK.replace('/', '_').replace('-', '_')
        safe_algo1 = algo1.replace('/', '_')
        safe_algo2 = algo2.replace('/', '_')
        filename = f"power_{safe_task}_{safe_algo1}_vs_{safe_algo2}.png"
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()  # Close to free memory
        
        plots_created += 1
        print(f"Created plot: {filename}")
    
    print(f"\n{'='*60}")
    print(f"Created {plots_created} individual power plots for {TASK}")
    print(f"Plots saved to: {plot_dir}")
    print(f"{'='*60}")
    
else:
    print(f"No power results for {TASK}. Check data availability.")
    power_empirical_df = pd.DataFrame()

# Save results CSV
if len(power_empirical_df) > 0:
    safe_task = TASK.replace('/', '_').replace('-', '_')
    csv_path = os.path.join(OUTPUT_DIR, f"{safe_task}_power_empirical_df.csv")
    power_empirical_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print(f"\nTask {TASK} power analysis completed successfully!")

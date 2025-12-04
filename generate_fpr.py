#!/usr/bin/env python3
"""
Generate False Positive Rate (FPR) analysis plots and results.

Output directory can be configured via environment variable:
    export FPR_OUTPUT_DIR=/path/to/output
    
Default: /n/home09/annabelma/rl_final_proj/stat_results/12_4_results

Outputs:
    - fpr_empirical_df.csv: Raw FPR results
    - fpr_aggregated_alpha*.png: Aggregated FPR plots by alpha level
    - fpr_plots/: Directory containing individual FPR plots for each (task, algo pair, alpha)
"""

import os
import warnings
from typing import Tuple

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


def compute_central_tendency_and_error(id_central, id_error, sample):
    """
    Compute central tendency and error bands (from rl_stats).
    
    Args:
        id_central: 'mean' or 'median'
        id_error: 'std', 'sem', or int (percentile, e.g., 80)
        sample: array of shape (n_steps, n_seeds)
    
    Returns:
        central, low, high arrays
    """
    try:
        id_error = int(id_error)
    except:
        pass

    if id_central == 'mean':
        central = np.nanmean(sample, axis=1)
    elif id_central == 'median':
        central = np.nanmedian(sample, axis=1)
    else:
        raise NotImplementedError

    if isinstance(id_error, int):
        low = np.nanpercentile(sample, q=int((100 - id_error) / 2), axis=1)
        high = np.nanpercentile(sample, q=int(100 - (100 - id_error) / 2), axis=1)
    elif id_error == 'std':
        low = central - np.nanstd(sample, axis=1)
        high = central + np.nanstd(sample, axis=1)
    elif id_error == 'sem':
        low = central - np.nanstd(sample, axis=1) / np.sqrt(sample.shape[1])
        high = central + np.nanstd(sample, axis=1) / np.sqrt(sample.shape[1])
    else:
        raise NotImplementedError

    return central, low, high


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
# Configuration: Output directory (configurable via environment variable)
# ============================================================================
BASE = os.getcwd()
OUTPUT_DIR = os.getenv(
    'FPR_OUTPUT_DIR',
    '/n/home09/annabelma/rl_final_proj/stat_results/12_4_results'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Statistical testing configuration
SEED_GRID = [2, 3, 5, 10, 20, 30] #[5, 10, 20, 30, 50, 100, 150, 200]  # sample sizes per group for FPR/Power analysis 
ALPHAS = [0.05, 0.01]  # Significance levels
EPSILONS = [0.5, 1.0, 2.0]  # cohen's d effect sizes for power analysis
N_RESAMPLES = 1000  # number of bootstrap/permutation resamples (reduce for faster iteration)

TASKS = [
    "Hopper-v5",
    "Walker2d-v5",
    "HalfCheetah-v5",
    "Ant-v5",
    "Humanoid-v5",
]

ALGORITHMS = ["SAC", "TD3", "DDPG", "PPO"]

EVAL_EPISODES = 20

TIMESTEPS_PER_TASK = {
    "Hopper-v5":      1_000_000,
    "Walker2d-v5":    1_000_000,
    "HalfCheetah-v5": 3_000_000,
    "Ant-v5":         3_000_000,
    "Humanoid-v5":   10_000_000,
}

DEFAULT_TOTAL_TIMESTEPS = 5_000_000

BASE_DIR = os.path.join(BASE, "rl_experiments")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_CSV = os.path.join(BASE_DIR, "final_eval_returns.csv")
LEARNING_CURVES_CSV = os.path.join(BASE_DIR, "learning_curves.csv")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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

print(f"\nUnique tasks in final_returns: {sorted(final_returns.task.dropna().unique())}")
print(f"Unique algorithms in final_returns: {sorted(final_returns.algorithm.dropna().unique())}")


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


def estimate_fpr_empirical(empirical_samples1, empirical_samples2, 
                           test_name, target_n, alpha=0.05, 
                           n_resamples=1000, seed=None):
    """
    Estimate False Positive Rate following paper methodology:
    "Enforce H0 by aligning central performances: µ1 = µ2 = 0 
    (median for Mann-Whitney/ranked t-test, mean for others)"
    
    Args:
        empirical_samples1, empirical_samples2: actual data samples from algorithms A and B
        test_name: name of test from tests_list
        target_n: target seed budget (sample size)
        alpha: significance level (default 0.05 as per paper)
        n_resamples: number of resamples Nr = 10^3
        seed: random seed
    
    Returns:
        fpr: empirical false positive rate α*
        se: standard error se(α*) = sqrt(α*(1-α*)/Nr)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Determine whether to use median or mean based on test
    # Paper: "median for Mann-Whitney and ranked t-test, mean for others"
    use_median = (test_name == 'Mann-Whitney' or test_name == 'Ranked t-test')
    
    # Step 1: Recenter to create null world (µ1 = µ2 = 0)
    recentered1, recentered2 = recenter_samples(empirical_samples1, empirical_samples2, use_median)
    
    # Step 2: Sample and test
    false_positives = 0
    
    for _ in range(n_resamples):
        # Draw N-sized samples from recentered distributions (with replacement)
        sample1 = np.random.choice(recentered1, size=target_n, replace=True)
        sample2 = np.random.choice(recentered2, size=target_n, replace=True)
        
        # Run test at level alpha
        try:
            reject = run_test(test_name, sample1, sample2, alpha=alpha)
            if reject:
                false_positives += 1
        except:
            continue
    
    # Step 3: Empirical FPR α* = (# rejections) / Nr
    fpr = false_positives / n_resamples
    
    # Standard error: se(α*) = sqrt(α*(1-α*)/Nr)
    se = np.sqrt(fpr * (1 - fpr) / n_resamples) if n_resamples > 0 else 0
    
    return fpr, se

    # Use configuration variables if available, otherwise use defaults
try:
    seed_grid = SEED_GRID
    alphas = ALPHAS
    n_resamples = N_RESAMPLES
except NameError:
    print("Warning: Configuration variables not defined. Using defaults.")
    print("Make sure to run Cell 3 (configuration) first.")
    # Use wider range matching paper (for better plots)
    seed_grid = [2, 3, 5, 10, 20, 30, 50, 100]
    alphas = [0.05, 0.01]
    n_resamples = 1000

print(f"Using empirical data from final_returns")
print(f"Testing across {len(seed_grid)} seed budgets: {seed_grid}")
print(f"Alpha levels: {alphas}")
print(f"Resamples per estimate: {n_resamples}")

fpr_empirical_results = []

# Get empirical samples from actual data
print("\n" + "="*60)
print("Estimating FPR using empirical procedure")
print("="*60)

for task in TASKS:
    task_df = final_returns[final_returns['task'] == task]
    available_algos = [a for a in ALGORITHMS if a in task_df['algorithm'].unique()]
    
    if len(available_algos) < 2:
        continue
    
    print(f"\nTask: {task}")
    
    for i, algo1 in enumerate(available_algos):
        for algo2 in available_algos[i+1:]:
            empirical1 = task_df[task_df['algorithm'] == algo1]['final_return_mean'].values
            empirical2 = task_df[task_df['algorithm'] == algo2]['final_return_mean'].values
            
            if len(empirical1) < 5 or len(empirical2) < 5:
                print(f"  Skipping {algo1} vs {algo2}: insufficient samples")
                continue
            
            print(f"  Using empirical samples: {algo1} (n={len(empirical1)}), {algo2} (n={len(empirical2)})")
            
            for alpha in alphas:
                print(f"    Alpha = {alpha}")
                for target_n in seed_grid:
                    if target_n > min(len(empirical1), len(empirical2)):
                        continue
                    
                    print(f"      Target seed budget N = {target_n}")
                    for test_name in tests_list:
                        try:
                            fpr, se = estimate_fpr_empirical(
                                empirical1, empirical2, 
                                test_name, target_n, 
                                alpha=alpha, 
                                n_resamples=n_resamples,
                                seed=None
                            )
                            fpr_empirical_results.append({
                                'task': task,
                                'algo1': algo1,
                                'algo2': algo2,
                                'test': test_name,
                                'alpha': alpha,
                                'target_n': target_n,
                                'fpr': fpr,
                                'se': se,
                                'expected_fpr': alpha
                            })
                            print(f"        {test_name:20s}: α* = {fpr:.4f} ± {se:.4f} (expected ≈ {alpha:.4f})")
                        except Exception as e:
                            print(f"        {test_name:20s}: Error - {e}")

fpr_empirical_df = pd.DataFrame(fpr_empirical_results)

print("\n" + "="*60)
print("Empirical FPR Results Summary")
print("="*60)
print("Following paper: α* estimated as proportion of H0 rejections over Nr = 10^3 resamples")
print("Standard error: se(α*) = sqrt(α*(1-α*)/Nr)")
print("="*60)

if len(fpr_empirical_df) > 0:
    # Summary table by test and alpha (with standard errors)
    summary = fpr_empirical_df.groupby(['test', 'alpha']).agg({
        'fpr': ['mean', 'std', 'count'],
        'se': 'mean'
    })
    print("\nSummary by test and alpha level:")
    print(summary.round(4))
    
    # Average FPR across all tasks with standard errors
    print(f"\nAverage FPR by test (across all tasks and seed budgets):")
    avg_stats = fpr_empirical_df.groupby(['test', 'alpha']).agg({
        'fpr': 'mean',
        'se': 'mean'
    })
    for (test, alpha), row in avg_stats.iterrows():
        fpr_val = row['fpr']
        se_val = row['se']
        print(f"  {test:20s} (α={alpha:.3f}): α* = {fpr_val:.4f} ± {se_val:.4f} (expected ≈ {alpha:.4f})")
    
    print(f"\nTotal results: {len(fpr_empirical_df)} rows")
    
    # ============================================================================
    # Plot FPR vs Sample Size (matching paper Figure 4.2)
    # ============================================================================
    print("\n" + "="*60)
    print("Plotting FPR vs Sample Size")
    print("="*60)
    
    # Create plots for each alpha level
    for alpha in alphas:
        alpha_data = fpr_empirical_df[fpr_empirical_df['alpha'] == alpha]
        if len(alpha_data) == 0:
            continue
        
        # Aggregate across tasks and algorithm pairs (average FPR for each test and sample size)
        plot_data = alpha_data.groupby(['test', 'target_n']).agg({
            'fpr': 'mean',
            'se': 'mean'
        }).reset_index()
        
        if len(plot_data) == 0:
            continue
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Color map for tests (matching paper style)
        test_colors = {
            't-test': '#1f77b4',           # blue
            'Welch t-test': '#ff7f0e',     # orange
            'Mann-Whitney': '#2ca02c',     # green
            'Ranked t-test': '#9467bd',    # purple
            'bootstrap': '#17becf',        # cyan
            'permutation': '#bcbd22'       # yellow-green
        }
        
        # Plot each test
        for test_name in tests_list:
            test_data = plot_data[plot_data['test'] == test_name].sort_values('target_n')
            if len(test_data) > 0:
                color = test_colors.get(test_name, '#000000')
                ax.plot(test_data['target_n'], test_data['fpr'], 
                       marker='o', label=test_name, linewidth=2, color=color, markersize=6)
                # Add error bars (standard errors)
                ax.errorbar(test_data['target_n'], test_data['fpr'], 
                           yerr=test_data['se'], 
                           fmt='none', color=color, alpha=0.3, capsize=3)
        
        # Add reference line at alpha
        ax.axhline(y=alpha, color='black', linestyle='--', linewidth=1.5, 
                  label=f'α = {alpha}', zorder=0)
        
        # Formatting
        ax.set_xlabel('Sample size N (log scale)', fontsize=12)
        ax.set_ylabel('False positive rate α*', fontsize=12)
        ax.set_title(f'False Positive Rate vs Sample Size (α = {alpha})', 
                    fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        
        # Set x-axis ticks to match available data and seed_grid
        available_n = sorted(plot_data['target_n'].unique())
        ax.set_xticks(available_n)
        ax.set_xticklabels([str(int(n)) for n in available_n])
        
        # Y-axis: show from 0 to at least 0.3, or higher if needed
        y_max = max(0.3, alpha_data['fpr'].max() * 1.15) if len(alpha_data) > 0 else 0.3
        ax.set_ylim([0, y_max])
        ax.set_yticks([0.0, 0.05, 0.1, 0.2, 0.3])
        
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', frameon=True, fontsize=10, ncol=2)
        
        plt.tight_layout()
        
        # Save aggregated plot
        plot_filename = f"fpr_aggregated_alpha{alpha:.2f}.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"\nPlot saved: {plot_path}")
else:
    print("No FPR results to display. Make sure final_returns is loaded.")
    fpr_empirical_df = pd.DataFrame()

# Save results CSV
if len(fpr_empirical_df) > 0:
    csv_path = os.path.join(OUTPUT_DIR, "fpr_empirical_df.csv")
    fpr_empirical_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

# ============================================================================
# Generate Individual FPR Plots for Each Algorithm Pair and Task
# ============================================================================

if len(fpr_empirical_df) > 0:
    print("="*60)
    print("Generating individual FPR plots for each algorithm pair and task")
    print("="*60)
    
    # Create output subdirectory for individual plots
    plot_dir = os.path.join(OUTPUT_DIR, "fpr_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get all unique combinations
    unique_combos = fpr_empirical_df.groupby(['task', 'algo1', 'algo2', 'alpha']).size().reset_index()
    print(f"\nFound {len(unique_combos)} unique (task, algo1, algo2, alpha) combinations")
    
    # Color map for tests (matching paper style)
    test_colors = {
        't-test': '#1f77b4',           # blue
        'Welch t-test': '#ff7f0e',     # orange
        'Mann-Whitney': '#2ca02c',     # green
        'Ranked t-test': '#9467bd',    # purple
        'bootstrap': '#17becf',        # cyan
        'permutation': '#bcbd22'       # yellow-green
    }
    
    plots_created = 0
    
    # Generate plot for each combination
    for idx, row in unique_combos.iterrows():
        task = row['task']
        algo1 = row['algo1']
        algo2 = row['algo2']
        alpha = row['alpha']
        
        # Filter data for this combination
        combo_data = fpr_empirical_df[
            (fpr_empirical_df['task'] == task) &
            (fpr_empirical_df['algo1'] == algo1) &
            (fpr_empirical_df['algo2'] == algo2) &
            (fpr_empirical_df['alpha'] == alpha)
        ]
        
        if len(combo_data) == 0:
            continue
        
        # Group by test and target_n
        plot_data = combo_data.groupby(['test', 'target_n']).agg({
            'fpr': 'mean',
            'se': 'mean'
        }).reset_index()
        
        if len(plot_data) == 0:
            continue
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot each test
        for test_name in tests_list:
            test_data = plot_data[plot_data['test'] == test_name].sort_values('target_n')
            if len(test_data) > 0:
                color = test_colors.get(test_name, '#000000')
                ax.plot(test_data['target_n'], test_data['fpr'], 
                       marker='o', label=test_name, linewidth=2, color=color, markersize=6)
                # Add error bars (standard errors)
                ax.errorbar(test_data['target_n'], test_data['fpr'], 
                           yerr=test_data['se'], 
                           fmt='none', color=color, alpha=0.3, capsize=3)
        
        # Add reference line at alpha
        ax.axhline(y=alpha, color='black', linestyle='--', linewidth=1.5, 
                  label=f'α = {alpha}', zorder=0)
        
        # Formatting
        ax.set_xlabel('Sample size N (log scale)', fontsize=12)
        ax.set_ylabel('False positive rate α*', fontsize=12)
        ax.set_title(f'FPR vs Sample Size: {task}\n{algo1} vs {algo2} (α = {alpha})', 
                    fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        
        # Set x-axis ticks
        available_n = sorted(plot_data['target_n'].unique())
        ax.set_xticks(available_n)
        ax.set_xticklabels([str(int(n)) for n in available_n])
        
        # Y-axis: show from 0 to at least 0.3, or higher if needed
        y_max = max(0.3, combo_data['fpr'].max() * 1.15) if len(combo_data) > 0 else 0.3
        ax.set_ylim([0, y_max])
        ax.set_yticks([0.0, 0.05, 0.1, 0.2, 0.3])
        
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best', frameon=True, fontsize=10, ncol=2)
        
        plt.tight_layout()
        
        # Save plot
        safe_task = task.replace('/', '_').replace('-', '_')
        safe_algo1 = algo1.replace('/', '_')
        safe_algo2 = algo2.replace('/', '_')
        filename = f"fpr_{safe_task}_{safe_algo1}_vs_{safe_algo2}_alpha{alpha:.2f}.png"
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()  # Close to free memory
        
        plots_created += 1
        
        # Show progress every 10 plots
        if plots_created % 10 == 0:
            print(f"Created {plots_created} plots...")
    
    print(f"\n{'='*60}")
    print(f"Created {plots_created} individual FPR plots")
    print(f"Plots saved to: {plot_dir}")
    print(f"{'='*60}")
    
else:
    print("No FPR results available. FPR analysis must complete successfully first.")
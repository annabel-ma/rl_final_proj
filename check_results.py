#!/usr/bin/env python3
"""
Check which seeds are completed for each task/algorithm combination
"""

import os
import yaml
import json
from collections import defaultdict
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    import csv
    
    config = load_config()
    
    # Get directories from config
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(SCRIPT_DIR, config["directories"]["base"])
    
    # Handle paths - they might be relative to BASE_DIR or absolute
    runs_path = config["directories"]["runs"]
    if os.path.isabs(runs_path):
        RUNS_DIR = runs_path
    else:
        RUNS_DIR = os.path.join(SCRIPT_DIR, runs_path)
    
    results_csv_path = config["directories"]["results_csv"]
    if os.path.isabs(results_csv_path):
        RESULTS_CSV = results_csv_path
    else:
        RESULTS_CSV = os.path.join(SCRIPT_DIR, results_csv_path)
    
    tasks = config["tasks"]
    algorithms = config["algorithms"]
    # Check seeds 0-200
    all_seeds = list(range(201))  # 0 to 200 inclusive
    
    # Dictionary to store completed seeds: {(task, algo): set([seed1, seed2, ...])}
    completed = defaultdict(set)
    
    # Method 1: Check CSV file (most reliable)
    if os.path.exists(RESULTS_CSV):
        try:
            with open(RESULTS_CSV, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task = row.get("task", "").strip()
                    algo = row.get("algorithm", "").strip()
                    seed_str = row.get("seed", "").strip()
                    # Don't filter by tasks/algorithms from config - check all
                    if task and algo and seed_str:
                        try:
                            seed_int = int(seed_str)
                            if 0 <= seed_int <= 200:
                                completed[(task, algo)].add(seed_int)
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            print(f"Warning: Could not read CSV file: {e}")
    
    # Method 2: Check JSON files in runs directory
    if os.path.exists(RUNS_DIR):
        for filename in os.listdir(RUNS_DIR):
            if filename.endswith(".json"):
                # Parse filename: {task}_{algo}_seed{seed}.json
                try:
                    base = filename[:-5]  # Remove .json
                    if "_seed" in base:
                        parts = base.rsplit("_seed", 1)
                        if len(parts) == 2:
                            task_algo = parts[0]
                            seed_str = parts[1]
                            
                            # Try to find matching task and algorithm
                            for task in tasks:
                                for algo in algorithms:
                                    expected = f"{task}_{algo}"
                                    if task_algo == expected:
                                        try:
                                            seed = int(seed_str)
                                            if 0 <= seed <= 200:
                                                # Verify the file is valid JSON
                                                filepath = os.path.join(RUNS_DIR, filename)
                                                with open(filepath, "r") as f:
                                                    data = json.load(f)
                                                    # Check if it has required fields
                                                    if "final_return_mean" in data:
                                                        completed[(task, algo)].add(seed)
                                        except (ValueError, json.JSONDecodeError, FileNotFoundError):
                                            pass
                except Exception:
                    pass
    
    # Convert sets to sorted lists for each combination
    completed_lists = {}
    for key in completed:
        completed_lists[key] = sorted(list(completed[key]))
    
    # Print results
    print("=" * 80)
    print("COMPLETED RESULTS SUMMARY (Seeds 0-200)")
    print("=" * 80)
    print()
    
    total_combinations = len(tasks) * len(algorithms)
    completed_combinations = len(completed_lists)
    total_possible = len(tasks) * len(algorithms) * 201
    
    total_completed = sum(len(seeds) for seeds in completed_lists.values())
    
    print(f"Total task/algorithm combinations: {total_combinations}")
    print(f"Combinations with at least one seed: {completed_combinations}")
    print(f"Total completed runs: {total_completed}/{total_possible}")
    print()
    
    # Group by task
    for task in tasks:
        print(f"\n{'='*80}")
        print(f"TASK: {task}")
        print(f"{'='*80}")
        
        for algo in algorithms:
            key = (task, algo)
            if key in completed_lists:
                seeds_done = completed_lists[key]
                done_count = len(seeds_done)
                missing = [s for s in all_seeds if s not in seeds_done]
                
                print(f"\n  {algo:8s}: {done_count:3d}/201 seeds completed")
                if seeds_done:
                    # Show first 10 and last 10 if many seeds
                    if len(seeds_done) <= 20:
                        print(f"           Completed: {seeds_done}")
                    else:
                        print(f"           Completed: {seeds_done[:10]} ... {seeds_done[-10:]}")
                        print(f"           ({len(seeds_done)} total)")
                if missing and len(missing) <= 30:
                    print(f"           Missing:   {missing}")
                elif missing:
                    print(f"           Missing:   {missing[:15]} ... ({len(missing)} total)")
            else:
                print(f"\n  {algo:8s}:   0/201 seeds completed")
                print(f"           Missing:   All seeds (0-200)")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Task':<20} {'Algorithm':<12} {'Completed':<12} {'Progress':<15}")
    print("-" * 80)
    
    for task in tasks:
        for algo in algorithms:
            key = (task, algo)
            if key in completed_lists:
                seeds_done = completed_lists[key]
                done_count = len(seeds_done)
                progress_pct = (done_count / 201) * 100
                progress_str = f"{progress_pct:5.1f}%"
            else:
                done_count = 0
                progress_str = "0.0%"
            
            print(f"{task:<20} {algo:<12} {done_count:3d}/201      {progress_str:<15}")
    
    print()

if __name__ == "__main__":
    main()

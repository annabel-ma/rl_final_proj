#!/usr/bin/env python3
"""
Generate joblist.txt for SLURM array jobs from YAML config
"""

import os
import yaml
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate joblist from YAML config")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--output", type=str, default="joblist.txt", help="Output joblist file")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    tasks = config["tasks"]
    algorithms = config["algorithms"]
    seeds = config["seeds"]
    
    joblist_path = args.output
    with open(joblist_path, "w") as f:
        # Order: for each seed, for each algorithm, for each task
        # This groups all tasks for all algorithms for each seed together
        for seed in seeds:
            for algo in algorithms:
                for task in tasks:
                    # Format: TASK ALGORITHM SEED
                    f.write(f"{task} {algo} {seed}\n")
    
    total_jobs = len(tasks) * len(algorithms) * len(seeds)
    print(f"Generated {joblist_path} with {total_jobs} jobs")
    print(f"  Order: seeds × algorithms × tasks")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Seeds: {len(seeds)}")


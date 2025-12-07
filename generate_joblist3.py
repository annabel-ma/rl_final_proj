#!/usr/bin/env python3
"""
Generate joblist3.txt with all jobs that have yet to run, excluding Humanoid tasks.
"""

import os
import csv
from collections import defaultdict

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
JOBLIST_PATH = os.path.join(BASE, "joblist.txt")
RESULTS_CSV = os.path.join(BASE, "rl_experiments", "final_eval_returns.csv")
OUTPUT_PATH = os.path.join(BASE, "joblist3.txt")

print("="*60)
print("Generating joblist3.txt (unrun jobs, excluding Humanoid)")
print("="*60)

# Read all jobs from joblist.txt
print(f"\nReading jobs from: {JOBLIST_PATH}")
all_jobs = []
with open(JOBLIST_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 3:
                task = parts[0]
                algo = parts[1]
                seed = parts[2]
                all_jobs.append((task, algo, seed))

print(f"Total jobs in joblist.txt: {len(all_jobs)}")

# Filter out Humanoid tasks
non_humanoid_jobs = [(task, algo, seed) for task, algo, seed in all_jobs if "Humanoid" not in task]
print(f"Jobs excluding Humanoid: {len(non_humanoid_jobs)}")

# Load completed jobs from results CSV
completed_jobs = set()
if os.path.exists(RESULTS_CSV):
    print(f"\nReading completed jobs from: {RESULTS_CSV}")
    try:
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = str(row.get('task', '')).strip()
                algo = str(row.get('algorithm', '')).strip()
                seed = str(row.get('seed', '')).strip()
                if task and algo and seed:
                    completed_jobs.add((task, algo, seed))
        print(f"Found {len(completed_jobs)} completed jobs in CSV")
    except Exception as e:
        print(f"Warning: Could not read CSV: {e}")
else:
    print(f"Warning: Results CSV not found: {RESULTS_CSV}")

# Find unrun jobs (non-Humanoid)
unrun_jobs = []
for task, algo, seed in non_humanoid_jobs:
    if (task, algo, seed) not in completed_jobs:
        unrun_jobs.append((task, algo, seed))

print(f"\nUnrun jobs (excluding Humanoid): {len(unrun_jobs)}")

# Write to joblist3.txt
print(f"\nWriting to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w") as f:
    for task, algo, seed in unrun_jobs:
        f.write(f"{task} {algo} {seed}\n")

print(f"\n{'='*60}")
print(f"Generated {OUTPUT_PATH} with {len(unrun_jobs)} jobs")
print(f"{'='*60}")

# Print summary by task
if unrun_jobs:
    by_task = defaultdict(int)
    for task, algo, seed in unrun_jobs:
        by_task[task] += 1
    
    print("\nUnrun jobs by task:")
    for task in sorted(by_task.keys()):
        print(f"  {task}: {by_task[task]} jobs")

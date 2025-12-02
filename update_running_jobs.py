#!/usr/bin/env python3
"""
Update the list of currently running jobs by querying SLURM.
This should be run periodically to keep running_jobs.txt up to date.
"""

import subprocess
import os
import sys

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBLIST_PATH = os.path.join(SCRIPT_DIR, "joblist.txt")
JOBLIST2_PATH = os.path.join(SCRIPT_DIR, "joblist2.txt")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "rl_experiments", "running_jobs.txt")

def get_joblist_for_job(job_id, job_base, script_dir, array_id, joblist_lines, joblist2_lines, nlines, nlines2):
    """Determine which joblist file a job uses by reading the log file and checking which joblist matches."""
    # Try to find the log file and extract task/algorithm/seed
    log_dirs = ['logs_whatsleft', 'logs', 'logs_mweber', 'logs_reverse', 'logs_new_script']
    
    task = None
    algo = None
    seed = None
    
    for log_dir in log_dirs:
        log_path = os.path.join(script_dir, log_dir, f"rl_{job_id}.out")
        if os.path.exists(log_path):
            try:
                # Read log file to find task/algorithm/seed
                with open(log_path, 'r') as f:
                    for line in f:
                        # Pattern 1: "=== Training Hopper-v5 | PPO | seed=10 ==="
                        if 'Training' in line and '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 3:
                                # Extract task
                                task_part = parts[0].strip()
                                task = task_part.replace('Training', '').replace('===', '').strip()
                                
                                # Extract algorithm
                                algo = parts[1].strip()
                                
                                # Extract seed
                                seed_part = parts[2].strip()
                                if 'seed=' in seed_part:
                                    seed_str = seed_part.split('seed=')[1].replace('===', '').strip()
                                    try:
                                        seed = int(seed_str)
                                        break
                                    except:
                                        pass
                        
                        # Pattern 2: Individual lines "Task: Hopper-v5", "Algorithm: PPO", "Seed: 10"
                        if 'Task:' in line and not task:
                            parts = line.split('Task:')
                            if len(parts) > 1:
                                task = parts[1].strip().split()[0]
                        elif 'Algorithm:' in line and not algo:
                            parts = line.split('Algorithm:')
                            if len(parts) > 1:
                                algo = parts[1].strip().split()[0]
                        elif 'Seed:' in line and seed is None:
                            parts = line.split('Seed:')
                            if len(parts) > 1:
                                try:
                                    seed = int(parts[1].strip().split()[0])
                                except:
                                    pass
                        
                        # If we have all three, we're done
                        if task and algo and seed is not None:
                            break
                
                # If we found task/algo/seed, check which joblist contains this at array_id
                if task and algo and seed:
                    # Check joblist2.txt first (forward)
                    if nlines2 > 0 and 1 <= array_id <= nlines2:
                        line = joblist2_lines[array_id - 1]
                        parts = line.split()
                        if len(parts) == 3 and parts[0] == task and parts[1] == algo and int(parts[2]) == seed:
                            return 'joblist2.txt'
                    
                    # Check joblist.txt (forward)
                    if 1 <= array_id <= nlines:
                        line = joblist_lines[array_id - 1]
                        parts = line.split()
                        if len(parts) == 3 and parts[0] == task and parts[1] == algo and int(parts[2]) == seed:
                            return 'joblist.txt'
                    
                    # Check joblist.txt (reverse)
                    if 1 <= array_id <= nlines:
                        line_idx = nlines - array_id
                        line = joblist_lines[line_idx]
                        parts = line.split()
                        if len(parts) == 3 and parts[0] == task and parts[1] == algo and int(parts[2]) == seed:
                            return 'joblist.txt'  # (reverse, but same file)
                    
                    # Also check if it exists anywhere in joblist2.txt (for safety)
                    for idx, line in enumerate(joblist2_lines):
                        parts = line.split()
                        if len(parts) == 3 and parts[0] == task and parts[1] == algo and int(parts[2]) == seed:
                            # Found in joblist2, so likely uses that
                            return 'joblist2.txt'
            except Exception as e:
                pass
    
    return None

def main():
    # Get running jobs with their names
    result = subprocess.Popen(['squeue', '-u', 'annabelma', '--format=%i|%j', '--noheader'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()
    
    if result.returncode != 0:
        print(f"Error running squeue: {stderr.decode('utf-8')}", file=sys.stderr)
        sys.exit(1)
    
    job_lines = stdout.decode('utf-8').strip().split('\n') if stdout else []

    # Read joblist.txt
    if not os.path.exists(JOBLIST_PATH):
        print(f"Error: joblist.txt not found at {JOBLIST_PATH}", file=sys.stderr)
        sys.exit(1)
    
    with open(JOBLIST_PATH, 'r') as f:
        joblist_lines = [line.strip() for line in f.readlines()]
    nlines = len(joblist_lines)
    
    # Read joblist2.txt if it exists
    joblist2_lines = []
    nlines2 = 0
    if os.path.exists(JOBLIST2_PATH):
        with open(JOBLIST2_PATH, 'r') as f:
            joblist2_lines = [line.strip() for line in f.readlines()]
        nlines2 = len(joblist2_lines)
        print(f"Found joblist2.txt with {nlines2} entries", file=sys.stderr)

    # Parse jobs and determine direction and joblist
    running_tasks = []
    for line in job_lines:
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) >= 2:
            job_id = parts[0].strip()
            job_name = parts[1].strip()
            job_parts = job_id.split('_')
            if len(job_parts) == 2:
                job_base = job_parts[0]
                try:
                    array_id = int(job_parts[1])
                    # Reverse jobs: 48488999 (might be), 48482358 (is reverse)
                    is_reverse = ('reverse' in job_name.lower() or job_base == '48482358')
                    
                    # Determine which joblist this job uses by reading log files
                    joblist_file = get_joblist_for_job(job_id, job_base, SCRIPT_DIR, array_id,
                                                       joblist_lines, joblist2_lines, nlines, nlines2)
                    if joblist_file is None:
                        # Fallback: use job ID patterns (more reliable than array_id range)
                        # 48596451 -> joblist2.txt
                        # 48482358, 48488999 -> joblist.txt (reverse)
                        # Others -> joblist.txt (forward)
                        if job_base == '48596451':
                            joblist_file = 'joblist2.txt'
                        else:
                            joblist_file = 'joblist.txt'
                    
                    running_tasks.append((job_base, array_id, is_reverse, joblist_file))
                except ValueError:
                    pass

    # Map array IDs to task/algorithm/seed using the detected joblist for each job
    running_combos = set()
    for job_base, array_id, is_reverse, joblist_file in running_tasks:
        # Select the appropriate joblist and lines
        if joblist_file == 'joblist2.txt' and nlines2 > 0:
            joblist_to_use = joblist2_lines
            nlines_to_use = nlines2
        else:
            joblist_to_use = joblist_lines
            nlines_to_use = nlines
        
        if 1 <= array_id <= nlines_to_use:
            if is_reverse:
                # Reverse: array_id 1 -> line nlines, array_id 2 -> line nlines-1, etc.
                line_idx = nlines_to_use - array_id
                line = joblist_to_use[line_idx]
            else:
                # Forward: array_id 1 -> line 0, etc.
                line = joblist_to_use[array_id - 1]
            
            parts = line.split()
            if len(parts) == 3:
                task, algo, seed = parts
                running_combos.add((task, algo, int(seed)))

    # Write to file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for task, algo, seed in sorted(running_combos, key=lambda x: (x[0], x[1], x[2])):
            f.write(f"{task} {algo} {seed}\n")

    print(f"Updated {OUTPUT_FILE} with {len(running_combos)} running job combinations")


if __name__ == "__main__":
    main()

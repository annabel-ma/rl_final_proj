#!/usr/bin/env python3
"""
RL Training Script for SLURM
Takes command line arguments: task, algorithm, seed
"""

import sys
import os
import argparse
import yaml
import fcntl

# Set base directory before imports
DRIVE_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DRIVE_BASE)

# Install packages if needed (commented out for SLURM - should be in environment)
# !pip install "numpy<2.0" "scipy<1.11"
# !pip install mujoco
# !pip install "gymnasium[mujoco]"
# !pip install stable-baselines3 pandas matplotlib tqdm

import gymnasium as gym
from stable_baselines3 import SAC, TD3, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import time
import json
from typing import Any, Dict, List

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(DRIVE_BASE, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def load_running_jobs(running_jobs_file: str) -> set:
    """Load set of currently running job combinations from file.
    
    Returns set of tuples (task, algorithm, seed) for jobs currently running.
    Returns empty set if file doesn't exist.
    """
    running_jobs = set()
    if os.path.exists(running_jobs_file):
        try:
            with open(running_jobs_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 3:
                            task, algo, seed = parts
                            running_jobs.add((task, algo, int(seed)))
        except Exception as e:
            print(f"[WARNING] Could not load running jobs file {running_jobs_file}: {e}")
    return running_jobs


# Load configuration
CONFIG = load_config()

TASKS = CONFIG["tasks"]
ALGORITHMS = CONFIG["algorithms"]
EVAL_EPISODES = CONFIG["eval_episodes"]
TIMESTEPS_PER_TASK = CONFIG["timesteps_per_task"]
DEFAULT_TOTAL_TIMESTEPS = CONFIG["default_total_timesteps"]
GLOBAL_RNG_SEED = CONFIG["global_rng_seed"]

# Set up directories
BASE_DIR = os.path.join(DRIVE_BASE, CONFIG["directories"]["base"])
RUNS_DIR = os.path.join(DRIVE_BASE, CONFIG["directories"]["runs"])
MODELS_DIR = os.path.join(DRIVE_BASE, CONFIG["directories"]["models"])
RESULTS_CSV = os.path.join(DRIVE_BASE, CONFIG["directories"]["results_csv"])
LEARNING_CURVES_CSV = os.path.join(DRIVE_BASE, CONFIG["directories"]["learning_curves_csv"])
LEARNING_CURVES_DIR = os.path.join(DRIVE_BASE, CONFIG["directories"]["base"], "learning_curves")
RUNNING_JOBS_FILE = os.path.join(DRIVE_BASE, CONFIG["directories"]["base"], "running_jobs.txt")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LEARNING_CURVES_DIR, exist_ok=True)

np.random.seed(GLOBAL_RNG_SEED)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    _ = env.reset(seed=seed)
    return Monitor(env)


def _build_action_noise(env, sigma: float = 0.1):
    assert hasattr(env, "action_space")
    action_dim = env.action_space.shape[0]
    return NormalActionNoise(mean=np.zeros(action_dim),
                             sigma=sigma * np.ones(action_dim))


def make_model(algo: str, env: gym.Env, seed: int):
    algo = algo.upper()
    policy = "MlpPolicy"
    algo_settings = CONFIG.get("algorithm_settings", {}).get(algo, {})
    
    # Try to use GPU if available, fall back to CPU
    device = "cpu"
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available - cannot run training")
    
    # Check if CUDA is available
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        # Try to use GPU with retries (for SLURM GPU allocation issues)
        import random
        # Use array task ID to further stagger jobs
        array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        base_delay = random.uniform(2.0, 8.0)
        task_delay = (array_id % 20) * 0.3  # Stagger based on task ID
        initial_delay = base_delay + task_delay
        print(f"[INFO] GPU detected, attempting to use GPU with {initial_delay:.2f}s delay (base: {base_delay:.2f}s + task: {task_delay:.2f}s)...")
        time.sleep(initial_delay)
        
        # Wait for GPU to be available with retries and exponential backoff
        max_retries = 60  # More retries for SLURM allocation issues
        base_retry_delay = 3.0  # Base delay between retries
        gpu_available = False
        
        for attempt in range(max_retries):
            try:
                # Test GPU availability with a more thorough check
                torch.cuda.empty_cache()
                # Use explicit device index 0 (the GPU made visible by CUDA_VISIBLE_DEVICES)
                test_tensor = torch.zeros(1, device="cuda:0")
                result = test_tensor + 1
                del test_tensor, result
                torch.cuda.synchronize(device=0)  # Wait for GPU operations to complete
                torch.cuda.empty_cache()
                gpu_available = True
                break
            except RuntimeError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    backoff_delay = base_retry_delay * (1.15 ** min(attempt // 5, 4))  # Cap at ~4x
                    jitter = random.uniform(0.8, 1.2)
                    actual_delay = backoff_delay * jitter
                    
                    if attempt % 5 == 0:  # Only print every 5th attempt to reduce log spam
                        print(f"[WARNING] GPU not ready (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"[INFO] Waiting {actual_delay:.2f}s before retry (backoff: {backoff_delay:.2f}s)...")
                    time.sleep(actual_delay)
                else:
                    # GPU unavailable after retries, fall back to CPU
                    print(f"[WARNING] GPU unavailable after {max_retries} attempts, falling back to CPU")
                    print(f"[WARNING] Last error: {e}")
                    gpu_available = False
                    break
        
        if gpu_available:
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_id = torch.cuda.current_device()
            print(f"[INFO] Using GPU {gpu_id}: {gpu_name}")
            print(f"[INFO] CUDA_VISIBLE_DEVICES={cuda_visible}")
        else:
            print(f"[INFO] GPU not available, using CPU")
            print(f"[INFO] CUDA_VISIBLE_DEVICES={cuda_visible}")
            print(f"[INFO] torch.cuda.is_available()={cuda_available}")
    else:
        # No CUDA available, use CPU
        print(f"[INFO] CUDA not available, using CPU")
        print(f"[INFO] CUDA_VISIBLE_DEVICES={cuda_visible}")
        print(f"[INFO] torch.cuda.is_available()={cuda_available}")
    
    # Create model on selected device (GPU or CPU)
    try:
        if algo == "SAC":
            model = SAC(policy, env, verbose=0, seed=seed, device=device)
        elif algo == "TD3":
            sigma = algo_settings.get("action_noise", {}).get("sigma", 0.1)
            noise = _build_action_noise(env, sigma=sigma)
            model = TD3(policy, env, action_noise=noise, verbose=0, seed=seed, device=device)
        elif algo == "DDPG":
            sigma = algo_settings.get("action_noise", {}).get("sigma", 0.1)
            noise = _build_action_noise(env, sigma=sigma)
            model = DDPG(policy, env, action_noise=noise, verbose=0, seed=seed, device=device)
        elif algo == "PPO":
            model = PPO(policy, env, verbose=0, seed=seed, device=device)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            raise RuntimeError(f"Failed to create model on GPU: {e}. GPU may be busy or improperly allocated.") from e
        else:
            raise  # Re-raise non-CUDA errors
    
    return model


def evaluate_policy_deterministic(model, env_id: str, n_episodes: int,
                                  eval_seed_base: int = 10_000):
    """Deterministic eval (mean action); returns list of episode returns and mean."""
    returns = []
    env = gym.make(env_id)
    for ep in range(n_episodes):
        obs, info = env.reset(seed=eval_seed_base + ep)
        terminated = False
        truncated = False
        ep_ret = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
        returns.append(ep_ret)
    env.close()
    return returns, float(np.mean(returns))


class EvalLoggerCallback(BaseCallback):
    def __init__(self,
                 env_id: str,
                 algo: str,
                 seed: int,
                 eval_freq: int,
                 eval_episodes: int,
                 csv_path: str = LEARNING_CURVES_CSV,
                 verbose: int = 0):
        super().__init__(verbose)
        self.env_id = env_id
        self.algo = algo
        self.seed = seed
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.csv_path = csv_path

    def _on_step(self) -> bool:
        # self.num_timesteps is provided by BaseCallback / SB3
        if self.num_timesteps % self.eval_freq == 0:
            # run deterministic evaluation using your helper
            ep_returns, mean_ret = evaluate_policy_deterministic(
                self.model,          # SB3 model
                self.env_id,
                n_episodes=self.eval_episodes,
                eval_seed_base=100_000 + self.seed * 1000
            )
            row = {
                "timestamp": time.time(),
                "task": self.env_id,
                "algorithm": self.algo,
                "seed": self.seed,
                "env_steps": int(self.num_timesteps),
                "eval_episodes": self.eval_episodes,
                "eval_return_mean": float(mean_ret),
                "eval_return_std": float(np.std(ep_returns)),
            }
            df = pd.DataFrame([row])
            header = not os.path.exists(self.csv_path)
            # Each task/algorithm/seed has its own CSV file, so no file locking contention
            # Still use locking for safety, but it should rarely be needed
            try:
                with open(self.csv_path, "a") as f:
                    # Try to acquire lock (non-blocking), but don't fail if unavailable
                    lock_acquired = False
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # Non-blocking
                        lock_acquired = True
                    except (OSError, IOError) as lock_err:
                        # Lock unavailable - that's okay, we'll write without it
                        # Since each file is unique per task/algorithm/seed, this is very unlikely
                        pass
                    
                    # Write the data (with or without lock)
                    df.to_csv(f, mode="a", header=header, index=False)
                    
                    # Release lock if we acquired it
                    if lock_acquired:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except:
                            pass  # Ignore errors when releasing
            except Exception as e:
                # If file writing completely fails, log but don't crash training
                if self.verbose > 0:
                    print(f"[ERROR] Failed to write learning curve data: {e}")
            if self.verbose > 0:
                print(f"[EVAL] {self.env_id} | {self.algo} | seed={self.seed} "
                      f"| steps={self.num_timesteps} | return={mean_ret:.1f}")
        return True  # keep training


def train_one_seed(env_id: str, algo: str, seed: int,
                   total_timesteps: int | None = None,
                   eval_episodes: int = EVAL_EPISODES,
                   cache_dir: str = RUNS_DIR,
                   models_dir: str = MODELS_DIR,
                   skip_if_cached: bool = True) -> Dict[str, Any] | None:
    if total_timesteps is None:
        total_timesteps = TIMESTEPS_PER_TASK.get(env_id, DEFAULT_TOTAL_TIMESTEPS)
    run_tag = f"{env_id}_{algo}_seed{seed}"
    run_json = os.path.join(cache_dir, f"{run_tag}.json")
    model_path = os.path.join(models_dir, f"{run_tag}.zip")
    
    # Use separate CSV file per task/algorithm/seed to avoid file locking contention
    learning_curve_csv = os.path.join(LEARNING_CURVES_DIR, f"{run_tag}.csv")
    
    # Check skip conditions in order:
    # 1. First check if final result (JSON) exists - this means the run completed successfully
    if skip_if_cached and os.path.exists(run_json):
        print(f"[SKIP] {run_tag} already completed (final result JSON found)")
        with open(run_json, "r") as f:
            return json.load(f)
    
    # 2. Check if this job is currently running (to avoid duplicate runs)
    running_jobs = load_running_jobs(RUNNING_JOBS_FILE)
    if skip_if_cached and (env_id, algo, seed) in running_jobs:
        print(f"[SKIP] {run_tag} is currently running (found in running_jobs.txt)")
        # Return None to indicate skip (caller should handle this)
        return None
    
    # 3. If CSV exists but no JSON and not running, it might be a partial run
    #    Delete the CSV so we start fresh (will be recreated during training)
    if os.path.exists(learning_curve_csv):
        if skip_if_cached:
            print(f"[INFO] {run_tag} has partial CSV but no final result - deleting CSV and retraining")
        try:
            os.remove(learning_curve_csv)
        except Exception as e:
            print(f"[WARNING] Could not delete partial CSV {learning_curve_csv}: {e}")
    
    env = make_env(env_id, seed)
    model = make_model(algo, env, seed=seed)
    eval_freq = max(total_timesteps // 100, 10_000)
    
    eval_callback = EvalLoggerCallback(
        env_id=env_id,
        algo=algo,
        seed=seed,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        csv_path=learning_curve_csv,
        verbose=1,
    )
    
    print(f"[TRAIN] {run_tag} starting, timesteps={total_timesteps}")
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=eval_callback,
    )
    t1 = time.time()
    print(f"[TRAIN] {run_tag} finished, elapsed seconds={t1 - t0:.2f}")
    
    try:
        model.save(model_path)
    except Exception as e:
        print(f"Warning: could not save model for {run_tag}: {e}")
    
    env.close()
    
    ep_returns, mean_return = evaluate_policy_deterministic(
        model, env_id, n_episodes=eval_episodes,
        eval_seed_base=100_000 + seed * 1000
    )
    
    result = {
        "timestamp": time.time(),
        "task": env_id,
        "algorithm": algo,
        "seed": seed,
        "total_timesteps": int(total_timesteps),
        "eval_episodes": int(eval_episodes),
        "episode_returns": ep_returns,
        "final_return_mean": float(mean_return),
        "model_path": model_path if os.path.exists(model_path) else None,
    }
    
    with open(run_json, "w") as f:
        json.dump(result, f, cls=NpEncoder)
    
    return result


def append_to_master_csv(record: Dict[str, Any], csv_path: str = RESULTS_CSV):
    row = {
        "timestamp": record.get("timestamp", time.time()),
        "task": record["task"],
        "algorithm": record["algorithm"],
        "seed": record["seed"],
        "total_timesteps": record.get("total_timesteps", np.nan),
        "eval_episodes": record.get("eval_episodes", np.nan),
        "final_return_mean": record["final_return_mean"],
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    # Use file locking for thread-safe CSV writes
    with open(csv_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            df.to_csv(f, mode="a", header=header, index=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--task", type=str, required=True, help="Environment ID")
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm name")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes (overrides config)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--no-skip-cached", action="store_true", help="Don't skip if cached")
    
    args = parser.parse_args()
    
    # Reload config if custom path provided
    global CONFIG, EVAL_EPISODES
    if args.config:
        CONFIG = load_config(args.config)
        EVAL_EPISODES = CONFIG["eval_episodes"]
    
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else EVAL_EPISODES
    
    print(f"=== Training {args.task} | {args.algorithm} | seed={args.seed} ===")
    
    rec = train_one_seed(
        args.task,
        args.algorithm,
        args.seed,
        total_timesteps=None,  # Will use TIMESTEPS_PER_TASK
        eval_episodes=eval_episodes,
        skip_if_cached=not args.no_skip_cached
    )
    
    # Only append to CSV if we actually ran or completed (not skipped)
    if rec is not None:
        append_to_master_csv(rec, RESULTS_CSV)
        print(f"=== Completed {args.task} | {args.algorithm} | seed={args.seed} ===")
        if rec.get('final_return_mean') is not None:
            print(f"Final return: {rec['final_return_mean']:.2f}")
        else:
            print("Job was skipped (already running or completed)")
    else:
        print(f"=== Skipped {args.task} | {args.algorithm} | seed={args.seed} (already running) ===")


if __name__ == "__main__":
    main()


import os
import time
import itertools
from concurrent.futures import ThreadPoolExecutor
import GPUtil  # Requires: pip install gputil

# ================= CONFIGURATION =================
# Define scenes to train
scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]

# Define output directory
output_dir = "output/release_version_nerf_synthetic"

# Define dataset root directory (Update this path!)
dataset_dir = "./data/nerf_synthetic"


# Factors (kept from original logic)
factors = [1]

# Execution settings
dry_run = False
excluded_gpus = set([])  # Add GPU IDs here to ignore them, e.g. {0, 1}
max_workers = 8          # Max concurrent threads
# =================================================

jobs = list(itertools.product(scenes, factors))

def train_scene(gpu, scene, factor):
    """Runs the training, rendering, and metrics pipeline for a single scene."""
    
    # Calculate specific port to avoid conflicts
    port = 6209 + int(gpu)
    
    # 1. Train
    cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
           f"python train.py -s {dataset_dir}/{scene} -m {output_dir}/{scene} "
           f"--eval --white_background --port {port} --config configs/nerf_synthetic.json")
    print(f"[GPU {gpu}] {cmd}")
    if not dry_run:
        os.system(cmd)

    # 2. Render
    cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
           f"python render.py -m {output_dir}/{scene} --config configs/nerf_synthetic.json")
    print(f"[GPU {gpu}] {cmd}")
    if not dry_run:
        os.system(cmd)
        
    # 3. Metrics
    cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
           f"python metrics.py -m {output_dir}/{scene}")
    print(f"[GPU {gpu}] {cmd}")
    if not dry_run:
        os.system(cmd)
    
    return True

    
def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")

    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set() 

    while jobs or future_to_job:
        # Get list of available GPUs using GPUtil
        # Adjust maxMemory and maxLoad if needed (0.5 = 50%)
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0) # job is (scene, factor)
            
            # Submit job to thread pool
            future = executor.submit(worker, gpu, *job)
            future_to_job[future] = (gpu, job)
            
            reserved_gpus.add(gpu)

        # Check for completed jobs
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)
            gpu = job[0]
            reserved_gpus.discard(gpu)
            print(f"Job {job} has finished, releasing GPU {gpu}")

        time.sleep(5)
        
    print("All jobs have been processed.")


if __name__ == "__main__":
    print(f"Dataset Dir: {dataset_dir}")
    print(f"Output Dir:  {output_dir}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        dispatch_jobs(jobs, executor)
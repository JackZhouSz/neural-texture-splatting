import os
import time
import itertools
from concurrent.futures import ThreadPoolExecutor
import GPUtil  # Requires: pip install gputil

# ================= CONFIGURATION =================
# Define DTU scenes to process (Default validation scenes)
scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

# Define paths (Update these paths!)
dataset_dir = "/path/to/DTU_dataset"      
eval_gt_dir = "/path/to/DTU_eval_GT_data"
output_dir = "output/release_version_dtu"

# Config file
config_file = "configs/dtu.json"

# Factors (kept for consistency with logic, usually 2 for DTU)
factors = [2]

# Execution settings
dry_run = False
excluded_gpus = set([])  # Add GPU IDs here to ignore them, e.g. {0, 1}
max_workers = 8          # Max concurrent threads
# =================================================

# Create job list (scene, factor)
jobs = list(itertools.product(scenes, factors))

def run_pipeline(gpu, scene_id, factor):
    """Runs the training, mesh extraction, and evaluation pipeline for a single scene."""
    
    scene_data_path = os.path.join(dataset_dir, f"scan{scene_id}")
    scene_output_path = os.path.join(output_dir, f"scan{scene_id}")

    # 1. Train
    # Note: --lambda_distortion 1000 and --use_decoupled_appearance are specific to this pipeline
    cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
           f"python train.py -s {scene_data_path} -m {scene_output_path} "
           f"-r {factor} --use_decoupled_appearance --lambda_distortion 1000 "
           f"--config {config_file}")
    
    print(f"[GPU {gpu} | Scan {scene_id}] Training...")
    print(f"  CMD: {cmd}")
    if not dry_run:
        if os.system(cmd) != 0: return False

    # 2. Mesh Extraction (TSDF Fusion)
    # Note: Iteration 30000 hardcoded
    cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
           f"python extract_mesh_tsdf.py -m {scene_output_path} "
           f"--iteration 30000")
    
    print(f"[GPU {gpu} | Scan {scene_id}] Extracting Mesh...")
    print(f"  CMD: {cmd}")
    if not dry_run:
        if os.system(cmd) != 0: return False
        
    # 3. Evaluation
    cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} "
           f"python evaluate_dtu_mesh.py -m {scene_output_path} "
           f"--scan_id {scene_id} --iteration 30000 "
           f"--DTU {eval_gt_dir}")
    
    print(f"[GPU {gpu} | Scan {scene_id}] Evaluating...")
    print(f"  CMD: {cmd}")
    if not dry_run:
        if os.system(cmd) != 0: return False
    
    return True

    
def worker(gpu, scene_id, factor):
    print(f"Starting job on GPU {gpu} with scan{scene_id}\n")
    run_pipeline(gpu, scene_id, factor)
    print(f"Finished job on GPU {gpu} with scan{scene_id}\n")

    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set() 

    while jobs or future_to_job:
        # Get list of available GPUs using GPUtil
        # Adjust maxMemory and maxLoad if needed (0.5 = 50% usage threshold)
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0) # job is (scene_id, factor)
            
            # Submit job to thread pool
            future = executor.submit(worker, gpu, *job)
            future_to_job[future] = (gpu, job)
            
            reserved_gpus.add(gpu)

        # Check for completed jobs
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            gpu, job_details = future_to_job.pop(future)
            reserved_gpus.discard(gpu)
            
            # Error handling wrapper
            try:
                future.result()
                print(f"Job {job_details} has finished, releasing GPU {gpu}")
            except Exception as e:
                print(f"Job {job_details} on GPU {gpu} FAILED with error: {e}")
                reserved_gpus.discard(gpu) # Ensure GPU is released even on error

        time.sleep(5)
        
    print("All jobs have been processed.")


if __name__ == "__main__":
    print(f"Dataset Dir: {dataset_dir}")
    print(f"Eval GT Dir: {eval_gt_dir}")
    print(f"Output Dir:  {output_dir}")
    
    # Simple check for directories
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        exit(1)
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        dispatch_jobs(jobs, executor)
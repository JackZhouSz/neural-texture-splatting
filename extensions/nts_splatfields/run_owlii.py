# training scripts for the nerf-synthetic datasets
# this script is adopted from GOF
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_nerf_synthetic.py
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import itertools


scenes = ["dancer", "exercise", "model", "basketball"]

factors = [1]
n_views = 10

output_dir = f"output/owlii_views{n_views}/"



dataset_dir = "./DATA_OWLII"

dry_run = False

excluded_gpus = set([])


jobs = list(itertools.product(scenes, factors))

def train_scene(gpu, scene, factor):

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/{scene} --white_background --eval --load_time_step 100  -m {output_dir}/{scene} --flow_model se3 --all_training --train_cam_names cam_train_0 cam_train_1 cam_train_2 cam_train_3 cam_train_4 cam_train_5 cam_train_6 cam_train_7 cam_train_8 cam_train_9 --pts_samples hull --iterations 200000 --encoder_type VarTriPlaneEncoder --num_pts 100000 --num_views 5 --composition_rank 40  --renderer_3dgs --deform_lr_max_steps 200000 --position_lr_init 0.00005"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -s {dataset_dir}/{scene} --white_background --eval --load_time_step 100  -m {output_dir}/{scene} --flow_model se3 --all_training --train_cam_names cam_train_0 cam_train_1 cam_train_2 cam_train_3 cam_train_4 cam_train_5 cam_train_6 cam_train_7 cam_train_8 cam_train_9 --pts_samples hull --iterations 100000 --encoder_type VarTriPlaneEncoder --num_pts 100000 --num_views 5 --composition_rank 40  --renderer_3dgs"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    return True

    
def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        available_gpus = list(set(all_available_gpus) - reserved_gpus)


        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
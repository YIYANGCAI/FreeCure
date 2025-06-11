#!/bin/bash
#SBATCH --job-name=photomaker          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node (adjust when using MPI)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks, adjust when using OMP)
#SBATCH --time=24:00:00          # total run time limit (D-HH:MM:SS)
#SBATCH --gpus-per-node=1        # gpu
#SBATCH --partition=digital-human        # partition(queue) where you submit (amd/intel/gpu-a30/gpu-l20)
#SBATCH -o "./logs/${%j}-indexing.txt"


export CUDA_VISIBLE_DEVICES=0
/aifs4su/caiyiyang/anaconda/envs/photomaker/bin/python run_photomaker_freecure.py \
    --prompt "a woman img with long black curly hair" \
    --reference-folder ./examples/scarletthead_woman \
    --output-folder "./outputs/"
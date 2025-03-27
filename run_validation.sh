#!/bin/bash

#SBATCH --job-name=llm_gpu
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project
#SBATCH --time=1-16:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem-per-cpu=32GB
#SBATCH --output=/scratch/mcity_project_root/mcity_project/zihaooo/slurm_output/%x-%j.out
#SBATCH --error=/scratch/mcity_project_root/mcity_project/zihaooo/slurm_output/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zihaooo@umich.edu

ulimit -c 0
source /home/zihaooo/miniconda3/bin/activate iti

cd /scratch/mcity_project_root/mcity_project/zihaooo/iti/validation

python validate_2fold.py --model_name llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default
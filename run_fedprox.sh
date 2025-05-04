#!/bin/bash
#SBATCH --job-name=run_fedprox
#SBATCH --output=run/run_fedprox_v2_1.out
#SBATCH --error=run/run_fedprox_v2_1.err
#SBATCH --time=02:00:00             
#SBATCH --partition=gpu          
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4            
#SBATCH --mem=64G                     
#SBATCH --gpus=h100:1

module load CUDA/11.8.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ddl-env
export WANDB_API_KEY=467ef7609483ffc67883540f9aff415f436814a9
python ./src/main.py \
          --model fedprox \
          --dataset fl_officecaltech \
          --backbone resnet18 \
          --wandb 0

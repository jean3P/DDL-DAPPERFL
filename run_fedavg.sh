#!/bin/bash
#SBATCH --job-name=run_fedavg     
#SBATCH --output=run/run_fedavg.out
#SBATCH --error=run/run_fedavg.err
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
          --model fedavg \
          --dataset fl_officecaltech \
          --backbone resnet18 \
	        --wandb 0

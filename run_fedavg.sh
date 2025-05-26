#!/bin/bash
#SBATCH --job-name=run_fedavg     
#SBATCH --output=run/run_fedavg_TPR_0.8.out
#SBATCH --error=run/run_fedavg_TPR_0.8.err
#SBATCH --time=02:00:00             
#SBATCH --partition=gpu          
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4            
#SBATCH --mem=32G
#SBATCH --gpus=rtx3090:1

module load CUDA/11.8.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ddl-env
export WANDB_API_KEY=467ef7609483ffc67883540f9aff415f436814a9
python ./src/main.py \
          --model fedavg \
          --dataset fl_officecaltech \
          --backbone resnet18 \
	        --wandb 0 \
          --noise_var 0.8 \
          --noise_clients 6 7 8 9

#!/bin/bash
#SBATCH --job-name=run_dapperfl     
#SBATCH --output=run/run_dapperfl_PP.out
#SBATCH --error=run/run_dapperfl_PP.err
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
         --model dapperfl \
         --dataset fl_officecaltech \
         --backbone resnet18 \
         --communication_epoch 100 \
         --local_epoch 5 \
         --parti_num 10 \
         --pr_strategy progressive \
         --alpha 0.9 \
         --alpha_min 0.1 \
         --epsilon 0.2 \
         --reg_coeff 0.01 \
         --wandb 0 \
         --device_id 0

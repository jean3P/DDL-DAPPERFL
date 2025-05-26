#!/bin/bash
#SBATCH --job-name=run_dapperfl_ds
#SBATCH --output=run/run_dapperfl_CD_PR_7_75.out
#SBATCH --error=run/run_dapperfl_CD_PR_7_75.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu          
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4            
#SBATCH --mem=64G                     
#SBATCH --gpus=rtx4090:1

module load CUDA/11.8.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ddl-env
export WANDB_API_KEY=467ef7609483ffc67883540f9aff415f436814a9
python ./src/continuous_domain_shift_experiment.py \
        --model dapperfl \
        --dataset fl_officecaltech \
        --backbone resnet18 \
        --parti_num 10 \
        --communication_epoch 100 \
        --local_epoch 5 \
        --shift_frequency 7 \
        --shift_ratio 0.75 \
        --pr_strategy progressive \
        --alpha 0.9 \
        --alpha_min 0.1 \
        --epsilon 0.2  \
        --experiment_name CD_PR_7_75

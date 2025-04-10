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

module load Workspace_Home
module load CUDA/11.8.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ddl-env
export WANDB_API_KEY=268b0fb16203164678d2e0cddc9f291c4285c791
python ./src/main.py --model fedavg --dataset fl_officecaltech --backbone resnet18

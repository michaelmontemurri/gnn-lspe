#!/bin/bash
#SBATCH --account=def-kolaczyk         # replace with your PI’s account
#SBATCH --gres=gpu:1                 # request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00              # max runtime (hh:mm:ss)
#SBATCH --job-name=lspe-run
#SBATCH --output=logs/lspe-%j.out    # output log (%j = job ID)

# Load required modules
module load python/3.10 scipy-stack cuda/12.1

# Activate your virtual environment
source ~/final_project/gnn-lspe/gnn_lspe_env/bin/activate

# Run your model 
python main_OGBMOL_graph_classification.py --config configs/PNA_MOLTOX21_LSPE_anchor.json

#!/bin/bash
#SBATCH --job-name=lspe-cpu-debug
#SBATCH --account=def-kolaczyk
#SBATCH --time=00:30:00               # e.g. 30 minutes for a quick debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/cpu_debug-%j.out

# Load required modules
module load python/3.10 scipy-stack

# Activate your virtual environment
source ~/final_project/gnn-lspe/gnn_lspe_env/bin/activate

# Run your script
python main_OGBMOL_graph_classification.py --config configs/GatedGCN_MOLTOX21_LSPE.json

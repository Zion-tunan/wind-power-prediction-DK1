#!/bin/bash
#SBATCH --job-name=lstm_attention_predict
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32

# Load environment
source /etc/profile.d/modules.sh
conda activate predict_test

# Move to code folder and run the script
cd codes
python wind_run.py

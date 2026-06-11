#!/bin/bash
#SBATCH --job-name=MACELES
#SBATCH --account=<your_account>
#SBATCH --partition=savio4_gpu
#SBATCH --mem=64G
#SBATCH --qos=<your_qos>
#SBATCH --cpus-per-task=8 --gres=gpu:L40:1 --ntasks=1
#SBATCH --time=240:00:00
#SBATCH -o out.%j
#SBATCH -e err.%j

source ~/.bashrc

module load gcc/11.4.0 openmpi/4.1.6 cuda/11.8.0

conda activate <your_env>

# Set your WandB API key before submitting:
#   export WANDB_API_KEY="<your_key>"
# or log in once with: wandb login

python ./run_train.py \
     --name="H2O-UMA-M-runMD-64" \
     --train_file="../train-H2O_UMA-M-500-configs_MACE_omol.xyz" \
     --valid_fraction=0.05 \
     --test_file="../test-H2O_UMA-M-50-configs_MACE_omol.xyz" \
     --E0s='average' \
     --model="MACELES" \
     --num_interactions=2 \
     --num_channels=64 \
     --max_L=1 \
     --correlation=2 \
     --r_max=4.5 \
     --energy_key="energy" \
     --forces_key="forces" \
     --batch_size=4 \
     --valid_batch_size=4 \
     --max_num_epochs=1000 \
     --stage_two \
     --start_stage_two=500 \
     --ema \
     --ema_decay=0.99 \
     --amsgrad \
     --default_dtype="float64" \
     --device=cuda \
     --seed=5 \
     --restart_latest \
     --wandb \
     --wandb_project="water-MD-500train-MACE-omol"

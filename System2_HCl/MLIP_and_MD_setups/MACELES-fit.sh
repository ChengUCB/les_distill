#!/bin/bash
#SBATCH --job-name=MACE_H100
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-cpu=18400M
#SBATCH --time=10:00:00
#SBATCH -o out.%j
#SBATCH -e err.%j


source ~/.bashrc
#load an CUDA software module
module load gcc/11.4.0 cuda/12.8.0

conda activate mace

python ./run_train.py \
     --name="2M-HCl_UMA-S-1p2-omol" \
     --train_file="./train_2M_H3O_1000_UMA-S-omol-1p2.xyz" \
     --valid_fraction=0.05 \
     --test_file="./test_2M_H3O_100_UMA-S-omol-1p2.xyz" \
     --energy_key="energy" \
     --forces_key="forces" \
     --E0s='average' \
     --model="MACELES" \
     --num_channels=64 \
     --r_max=5.5 \
     --num_interactions=2 \
     --max_L=1 \
     --correlation=2 \
     --batch_size=8 \
     --valid_batch_size=8 \
     --max_num_epochs=1000 \
     --stage_two \
     --start_stage_two=500 \
     --ema \
     --ema_decay=0.99 \
     --amsgrad \
     --restart_latest \
     --device=cuda \
     --default_dtype="float64"\
     --seed=2026 \

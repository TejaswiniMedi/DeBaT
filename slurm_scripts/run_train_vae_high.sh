#!/bin/bash
#SBATCH --job-name=VAVAE                        # Total 2 nodes
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --mail-user=tejaswini.medi@uni-mannheim.de
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:2                     # One GPU per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 2
#SBATCH --partition=gpu-vram-94gb
#SBATCH -o train_vae_high_output_%j.log
#SBATCH -e train_vae_high_error_%j.log

# activate Conda environment
source ~/.bashrc
conda activate vavae

cd vavae

# Set master address and port for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((12001 + RANDOM % 10000))


echo "Start time: $(date)"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Node list:"
scontrol show hostnames $SLURM_JOB_NODELIST

# start distributed training
srun python3 main.py \
    --train \
    --base configs/f16d32_vfdinov2_high.yaml \
    --logdir logs 

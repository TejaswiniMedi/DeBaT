## DeBaT
Decoupling High and Low Frequencies for Faithful Image Generation with Fine Details ( ECCV 2026)

## Training Latents 

# 1. Clone git clone git@github.com:TejaswiniMedi/DeBaT.git cd DeBaT 

# 2. Activate environment conda activate vavae 

# 3. Set your ImageNet path in # vavae/configs/f16d32_vfdinov2_low.yaml # vavae/configs/f16d32_vfdinov2_high.yaml 

# 4. Train low-frequency VAE sbatch slurm_scripts/run_train_vae_low.sh # 

# 5. Train high-frequency VAE sbatch slurm_scripts/run_train_vae_high.sh

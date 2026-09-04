# DeBaT

**Decoupling High and Low Frequencies for Faithful Image Generation with Fine Details (ECCV 2026)**

Official implementation of **DeBaT**, a frequency-decoupled latent generative framework for high-fidelity image generation.

DeBaT decomposes images into low- and high-frequency components, learns separate VA-VAE representations, concatenates both latents, and trains LightningDiT in the resulting latent space.

## Pipeline

```text
Image
  ↓
Haar Wavelet Decomposition
  ↓
Low Frequency ──→ Low VA-VAE ──┐
                               ├─→ Concatenated Latent ─→ LightningDiT
High Frequency ─→ High VA-VAE ─┘
```


## Repository

```text
DeBaT/
├── vavae/                  # Low/high-frequency VAE training
├── LightningDiT/           # Diffusion training
├── slurm_scripts/          # Example SLURM jobs
├── extract_dual_features.py
└── README.md
```

## Setup

```bash
git clone git@github.com:TejaswiniMedi/DeBaT.git
cd DeBaT
```

Set your ImageNet path:

```bash
export IMAGENET_ROOT=/path/to/imagenet
```

Avoid machine-specific absolute paths such as `/ceph/...` in configs. Prefer environment variables or paths relative to the repository.

## Installation

Create a Python 3.10 Conda environment:

```bash
conda create -n debat python=3.10 -y
conda activate debat
```

Install the VA-VAE dependencies:

```bash
pip install -r vavae/requirements.txt
```

Install the LightningDiT dependencies:

```bash
pip install -r LightningDiT/requirements.txt
```

The provided setup uses PyTorch 2.2.0 with CUDA 12.1. Users with a different CUDA setup should install the corresponding PyTorch build.


## 1. Train VAEs

Low-frequency branch:

```bash
cd vavae

python main.py \
    --train \
    --base configs/f16d32_vfdinov2_low.yaml \
    --logdir logs
```

High-frequency branch:

```bash
python main.py \
    --train \
    --base configs/f16d32_vfdinov2_high.yaml \
    --logdir logs
```

## 2. Extract Dual-Frequency Latents

From the repository root:

```bash
python extract_dual_features.py \
    --vavae_root ./vavae \
    --data_path "$IMAGENET_ROOT/train" \
    --output_dir ./data/imagenet256_debat_latents \
    --low_config ./vavae/configs/f16d32_vfdinov2_low.yaml \
    --high_config ./vavae/configs/f16d32_vfdinov2_high.yaml \
    --low_ckpt ./vavae/logs/f16d32_vfdinov2_low/checkpoints/last.ckpt \
    --high_ckpt ./vavae/logs/f16d32_vfdinov2_high/checkpoints/last.ckpt
```

## 3. Train LightningDiT

Set the latent path in:

```text
LightningDiT/configs/lightningdit_xl_vavae_f16d32.yaml
```

to:

```yaml
data:
  data_path: ../data/imagenet256_debat_latents
```

Single GPU:

```bash
cd LightningDiT

python train.py \
    --config configs/lightningdit_xl_vavae_f16d32.yaml \
    --mixed_precision bf16
```

Multiple GPUs:

```bash
GPUS_PER_NODE=2 \
bash run_train.sh configs/lightningdit_xl_vavae_f16d32.yaml
```


## Acknowledgements

This repository builds upon **VA-VAE / latent diffusion** and **LightningDiT**. Please refer to the corresponding source directories for their licenses and citations.

## Citation (will be updated)

```bash
@misc{medi2026decouplinghighlowfrequencies,
      title={Decoupling High and Low Frequencies for Faithful Image Generation with Fine Details}, 
      author={Tejaswini Medi and Hsien-Yi Wang and Arianna Rampini and Margret Keuper},
      year={2026},
      eprint={2509.05441},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.05441}, 
}```

## We are process of optimizing the repo

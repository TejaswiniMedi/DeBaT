#!/usr/bin/env python3
"""
Extract concatenated low/high-frequency VA-VAE latents in LightningDiT format.

Output shard format (compatible with LightningDiT ImgLatentDataset):
    latents_rank00_shard000.safetensors
      - latents:      [N, 32, 16, 16]
      - latents_flip: [N, 32, 16, 16]
      - labels:       [N]
    latents_stats.pt
      - mean: [1, 32, 1, 1]
      - std:  [1, 32, 1, 1]

The frequency preprocessing exactly follows this DeBaT codebase:
  low:  Haar-DWT LL / 2 -> bicubic back to 256x256 -> low VA-VAE
  high: Haar-DWT LH/HL/HH / 2 -> reshape to 9x128x128 -> high VA-VAE

LightningDiT's VA-VAE extraction uses posterior.sample(), so this script does too
by default. Pass --posterior_mode to use posterior.mode() instead.
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM/LightningDiT center crop."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size,
                               crop_x: crop_x + image_size])


def image_transform(image_size: int):
    return transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            inplace=True,
        ),
    ])


def setup_distributed(seed: int):
    """Works with torchrun; falls back to one GPU when not launched distributed."""
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for VA-VAE latent extraction.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    return distributed, rank, world_size, local_rank, device


def barrier(distributed: bool):
    if distributed:
        dist.barrier()


def unwrap_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def clean_state_dict_prefixes(state_dict):
    """Handle common wrappers without modifying normal Lightning keys."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_debat_vavae(config_path: str, ckpt_path: str, vavae_root: str, device):
    """
    Instantiate the DeBaT AutoencoderKL and load a training checkpoint.

    For the low-frequency model we force use_vf=None during extraction. DINO is
    only needed for the auxiliary training loss; AutoencoderKL.encode() itself
    only uses encoder + quant_conv. This avoids loading DINO during caching.
    """
    vavae_root = str(Path(vavae_root).resolve())
    if vavae_root not in sys.path:
        sys.path.insert(0, vavae_root)

    from ldm.util import instantiate_from_config

    cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))

    # Do not instantiate training-only networks just to encode latents.
    # - DINO is only used by the low-frequency auxiliary alignment loss.
    # - LPIPS/discriminator are only used while training the VAE.
    if "params" in model_cfg:
        model_cfg.params.use_vf = None
        model_cfg.params.lossconfig = OmegaConf.create({"target": "torch.nn.Identity"})

    model = instantiate_from_config(model_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = clean_state_dict_prefixes(unwrap_checkpoint(ckpt))
    incompatible = model.load_state_dict(state_dict, strict=False)

    # Expected for low checkpoint: foundation_model/linear_proj may be ignored
    # because use_vf was disabled above. Encoder/quant_conv MUST be present.
    critical_missing = [
        k for k in incompatible.missing_keys
        if k.startswith("encoder.") or k.startswith("quant_conv.")
    ]
    if critical_missing:
        raise RuntimeError(
            f"Critical encoder weights missing from {ckpt_path}: {critical_missing[:20]}"
        )

    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_dual_frequency(images, low_model, high_model, use_mode=False):
    """Return [B, 32, 16, 16] = cat([low_z, high_z], dim=1)."""
    # Low-frequency preprocessing copied from AutoencoderKL.training_step.
    ll1, _ = low_model.dwt(images)
    low_input = F.interpolate(
        ll1 / 2.0,
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    )

    # High-frequency preprocessing copied from AutoencoderKL.training_step.
    _, hs = high_model.dwt(images)
    h1 = hs[0] / 2.0                       # [B, 3, 3, 128, 128]
    high_input = h1.reshape(
        h1.shape[0], h1.shape[1] * h1.shape[2], h1.shape[3], h1.shape[4]
    )                                      # [B, 9, 128, 128]

    low_posterior = low_model.encode(low_input)
    high_posterior = high_model.encode(high_input)

    z_low = low_posterior.mode() if use_mode else low_posterior.sample()
    z_high = high_posterior.mode() if use_mode else high_posterior.sample()

    if z_low.shape[0] != z_high.shape[0] or z_low.shape[-2:] != z_high.shape[-2:]:
        raise RuntimeError(
            f"Low/high latent shapes cannot be concatenated: "
            f"low={tuple(z_low.shape)}, high={tuple(z_high.shape)}"
        )

    return torch.cat([z_low, z_high], dim=1)


def save_shard(output_dir, rank, shard_idx, latents, latents_flip, labels):
    save_filename = os.path.join(
        output_dir, f"latents_rank{rank:02d}_shard{shard_idx:03d}.safetensors"
    )
    save_dict = {
        "latents": latents.contiguous(),
        "latents_flip": latents_flip.contiguous(),
        "labels": labels.contiguous(),
    }
    save_file(
        save_dict,
        save_filename,
        metadata={
            "total_size": str(latents.shape[0]),
            "dtype": str(latents.dtype),
            "latent_layout": "[low_32ch,high_32ch]",
        },
    )
    return save_filename


def flush_exact_shards(buffers, output_dir, rank, shard_idx, shard_size, force=False):
    """Flush exact-size shards; on final force=True flush the remainder."""
    latents_list, flips_list, labels_list = buffers
    if not latents_list:
        return buffers, shard_idx

    latents = torch.cat(latents_list, dim=0)
    flips = torch.cat(flips_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    start = 0
    total = latents.shape[0]
    while total - start >= shard_size or (force and total - start > 0):
        end = min(start + shard_size, total)
        filename = save_shard(
            output_dir, rank, shard_idx,
            latents[start:end], flips[start:end], labels[start:end],
        )
        print(f"[rank {rank}] saved {filename} {tuple(latents[start:end].shape)}")
        shard_idx += 1
        start = end

    if start < total:
        buffers = ([latents[start:]], [flips[start:]], [labels[start:]])
    else:
        buffers = ([], [], [])
    return buffers, shard_idx


def compute_latent_stats(data_dir: str, max_samples: int = 10000, seed: int = 42):
    """
    Compute LightningDiT-style per-channel mean/std over normal (non-flipped)
    latents, but streaming to avoid holding ~650 MB for 10k x 32ch latents.
    """
    files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
    if not files:
        raise RuntimeError(f"No .safetensors shards found in {data_dir}")

    counts = []
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            counts.append(f.get_slice("labels").get_shape()[0])

    total_images = int(sum(counts))
    num_samples = min(max_samples, total_images)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total_images, num_samples, replace=False))

    cumulative = np.cumsum([0] + counts)
    per_file = [[] for _ in files]
    file_ids = np.searchsorted(cumulative[1:], selected, side="right")
    for global_idx, file_id in zip(selected, file_ids):
        local_idx = int(global_idx - cumulative[file_id])
        per_file[int(file_id)].append(local_idx)

    channel_sum = None
    channel_sumsq = None
    scalar_count = 0

    for file, indices in zip(files, per_file):
        if not indices:
            continue
        with safe_open(file, framework="pt", device="cpu") as f:
            features = f.get_slice("latents")
            for idx in indices:
                z = features[idx:idx + 1].to(torch.float64)  # [1,C,H,W]
                if channel_sum is None:
                    channels = z.shape[1]
                    channel_sum = torch.zeros(channels, dtype=torch.float64)
                    channel_sumsq = torch.zeros(channels, dtype=torch.float64)
                channel_sum += z.sum(dim=(0, 2, 3))
                channel_sumsq += (z * z).sum(dim=(0, 2, 3))
                scalar_count += z.shape[0] * z.shape[2] * z.shape[3]

    mean = channel_sum / scalar_count
    # Match torch.std(..., correction=1/default unbiased=True).
    variance = (channel_sumsq - channel_sum.square() / scalar_count) / max(scalar_count - 1, 1)
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    stats = {
        "mean": mean.float().view(1, -1, 1, 1),
        "std": std.float().view(1, -1, 1, 1),
    }
    stats_path = os.path.join(data_dir, "latents_stats.pt")
    torch.save(stats, stats_path)
    print(f"Saved latent stats: {stats_path}")
    print(f"mean shape={tuple(stats['mean'].shape)}, std shape={tuple(stats['std'].shape)}")
    return stats


def main(args):
    if args.image_size != 256:
        raise ValueError(
            "These DeBaT checkpoints were trained with the low branch at 256x256 "
            "and high branch at 128x128 after one Haar DWT level. Use --image_size 256."
        )

    distributed, rank, world_size, local_rank, device = setup_distributed(args.seed)

    if rank == 0:
        print(f"world_size={world_size}, device={device}")
        print("Loading low-frequency VA-VAE...")
    low_model = load_debat_vavae(
        args.low_config, args.low_ckpt, args.vavae_root, device
    )

    if rank == 0:
        print("Loading high-frequency VA-VAE...")
    high_model = load_debat_vavae(
        args.high_config, args.high_ckpt, args.vavae_root, device
    )

    dataset = ImageFolder(args.data_path, transform=image_transform(args.image_size))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed,
        drop_last=False,
    ) if distributed else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    output_dir = os.path.abspath(args.output_dir)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    barrier(distributed)

    latents_buf, flips_buf, labels_buf = [], [], []
    buffered = 0
    shard_idx = 0
    processed = 0

    save_dtype = torch.float16 if args.save_dtype == "fp16" else torch.float32

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        images_flip = torch.flip(images, dims=[-1])

        z = encode_dual_frequency(
            images, low_model, high_model, use_mode=args.posterior_mode
        )
        z_flip = encode_dual_frequency(
            images_flip, low_model, high_model, use_mode=args.posterior_mode
        )

        if batch_idx == 0:
            print(
                f"[rank {rank}] latent shape={tuple(z.shape)}, "
                f"low channels={z.shape[1] // 2}, high channels={z.shape[1] // 2}"
            )
            if z.shape[1:] != (32, 16, 16):
                raise RuntimeError(
                    f"Expected concatenated latent [B,32,16,16], got {tuple(z.shape)}"
                )

        z = z.to(dtype=save_dtype).cpu()
        z_flip = z_flip.to(dtype=save_dtype).cpu()
        labels = labels.cpu().to(torch.int64)

        latents_buf.append(z)
        flips_buf.append(z_flip)
        labels_buf.append(labels)
        buffered += z.shape[0]
        processed += z.shape[0]

        if buffered >= args.shard_size:
            (latents_buf, flips_buf, labels_buf), shard_idx = flush_exact_shards(
                (latents_buf, flips_buf, labels_buf),
                output_dir, rank, shard_idx, args.shard_size, force=False,
            )
            buffered = sum(x.shape[0] for x in latents_buf)

        if rank == 0 and processed % max(args.batch_size * 20, 100) < args.batch_size:
            print(f"processed ~{processed} samples on rank 0")

    (latents_buf, flips_buf, labels_buf), shard_idx = flush_exact_shards(
        (latents_buf, flips_buf, labels_buf),
        output_dir, rank, shard_idx, args.shard_size, force=True,
    )

    barrier(distributed)
    if rank == 0:
        compute_latent_stats(
            output_dir,
            max_samples=args.stats_samples,
            seed=args.seed,
        )
        print("\nDone.")
        print(f"LightningDiT data_path: {output_dir}")
        print("Set model.in_chans: 32 and keep vae.downsample_ratio: 16")

    barrier(distributed)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vavae_root", type=str, required=True,
                        help="Path to DeBaT/vavae (contains ldm/)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="ImageFolder root, e.g. ImageNet/train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--low_config", type=str, required=True)
    parser.add_argument("--high_config", type=str, required=True)
    parser.add_argument("--low_ckpt", type=str, required=True)
    parser.add_argument("--high_ckpt", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument("--stats_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--posterior_mode", action="store_true",
                        help="Use posterior.mode(); default matches LightningDiT and samples posterior")
    main(parser.parse_args())

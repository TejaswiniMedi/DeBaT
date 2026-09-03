import argparse
import os
from datetime import datetime

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
import torchvision
from PIL import Image
from safetensors.torch import save_file
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.dual_vavae import DualVA_VAE

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FlatImageDataset(Dataset):
    """Dataset for unlabeled flat image folders such as FFHQ."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        )
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(self.samples[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def resolve_data_root(data_path, data_split):
    split_root = os.path.join(data_path, data_split)
    return split_root if os.path.isdir(split_root) else data_path


def has_direct_images(root):
    root = Path(root)
    return any(
        path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        for path in root.iterdir()
    )


def build_dataset(root, transform, dataset_format):
    if dataset_format == "imagefolder":
        return ImageFolder(root, transform=transform)
    if dataset_format == "flat":
        return FlatImageDataset(root, transform=transform)
    if has_direct_images(root):
        return FlatImageDataset(root, transform=transform)
    return ImageFolder(root, transform=transform)


def setup_distributed(args):
    try:
        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            dist.init_process_group("nccl", init_method="env://", world_size=args.world_size, rank=rank)
        else:
            dist.init_process_group("nccl")
            rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        return rank, world_size, device, True
    except Exception:
        print("Failed to initialize DDP. Running in local mode.")
        return 0, 1, 0, False


def save_demo_grid(path, tokenizer, images, latents, max_images=8):
    images = images[:max_images].to(tokenizer.device)
    latents = latents[:max_images].to(tokenizer.device)
    recon_low = tokenizer.decode_low_to_tensor(latents)
    recon_full = tokenizer.decode_to_tensor(latents)

    input_vis = torch.clamp((images + 1) / 2, 0, 1)
    low_vis = torch.clamp((recon_low + 1) / 2, 0, 1)
    full_vis = torch.clamp((recon_full + 1) / 2, 0, 1)
    err_vis = torch.clamp(torch.abs(full_vis - input_vis) * 4, 0, 1)

    grid = torchvision.utils.make_grid(
        torch.cat([input_vis, low_vis, full_vis, err_vis], dim=0),
        nrow=images.shape[0],
        padding=2,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(grid, path)


def main(args):
    assert torch.cuda.is_available(), "Extracting dual VA-VAE latents requires at least one GPU."

    rank, world_size, device, distributed = setup_distributed(args)
    torch.manual_seed(args.seed + rank)
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"Starting rank={rank}, seed={args.seed + rank}, world_size={world_size}.")
        print(f"low_config={args.low_config}")
        print(f"high_config={args.high_config}")

    output_dir = os.path.join(args.output_path, args.exp_name, f"{args.data_split}_{args.image_size}")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer = DualVA_VAE(
        low_config=args.low_config,
        high_config=args.high_config,
        low_ckpt_path=args.low_ckpt_path,
        high_ckpt_path=args.high_ckpt_path,
        img_size=args.image_size,
        device=f"cuda:{device}",
    )

    data_root = resolve_data_root(args.data_path, args.data_split)
    datasets = [
        build_dataset(data_root, tokenizer.img_transform(p_hflip=0.0), args.dataset_format),
        build_dataset(data_root, tokenizer.img_transform(p_hflip=1.0), args.dataset_format),
    ]
    samplers = [
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        if distributed
        else None
        for dataset in datasets
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        for dataset, sampler in zip(datasets, samplers)
    ]

    if rank == 0:
        print(f"Total data in one loop: {len(loaders[0].dataset)}")

    run_images = 0
    saved_files = 0
    shard_images = 0
    demo_saved = False
    latents = []
    latents_flip = []
    labels = []

    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f"{datetime.now()} processing {run_images} of {len(loaders[0].dataset)} images")

        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]
            z = tokenizer.encode_images(x).detach().cpu()

            if batch_idx == 0 and rank == 0:
                print("latent shape", z.shape, "dtype", z.dtype)

            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
                if rank == 0 and not demo_saved:
                    demo_path = os.path.join(output_dir, "demo_grid.png")
                    save_demo_grid(demo_path, tokenizer, x, z, max_images=args.demo_images)
                    print(f"Saved demo grid to {demo_path}")
                    demo_saved = True
            else:
                latents_flip.append(z)

        shard_images += batch_data[0][0].shape[0]
        if shard_images >= args.shard_size:
            save_shard(output_dir, rank, saved_files, latents, latents_flip, labels)
            latents, latents_flip, labels = [], [], []
            shard_images = 0
            saved_files += 1

    if len(latents) > 0:
        save_shard(output_dir, rank, saved_files, latents, latents_flip, labels)

    if distributed:
        dist.barrier()
    if rank == 0:
        ImgLatentDataset(output_dir, latent_norm=True, latent_multiplier=args.latent_multiplier)
        print(f"Dual VA-VAE latents are ready at {output_dir}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def save_shard(output_dir, rank, shard_idx, latents, latents_flip, labels):
    latents = torch.cat(latents, dim=0)
    latents_flip = torch.cat(latents_flip, dim=0)
    labels = torch.cat(labels, dim=0)
    save_dict = {
        "latents": latents,
        "latents_flip": latents_flip,
        "labels": labels,
    }
    save_filename = os.path.join(output_dir, f"latents_rank{rank:02d}_shard{shard_idx:03d}.safetensors")
    save_file(
        save_dict,
        save_filename,
        metadata={"total_size": f"{latents.shape[0]}", "dtype": f"{latents.dtype}", "device": f"{latents.device}"},
    )
    print(f"Saved {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/ceph/mfatima/ImageNet/")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="/ceph/tmedi/Imagenet-1k-latent/")
    parser.add_argument("--exp_name", type=str, default="dual_vavae_f16d64")
    parser.add_argument("--low_config", type=str, default="vavae/configs/f16d32_vfdinov2_low.yaml")
    parser.add_argument("--high_config", type=str, default="vavae/configs/f16d32_vfdinov2_high.yaml")
    parser.add_argument("--low_ckpt_path", type=str, default=None)
    parser.add_argument("--high_ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "imagefolder", "flat"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--demo_images", type=int, default=8)
    parser.add_argument("--latent_multiplier", type=float, default=1.0)
    parser.add_argument("--world_size", default=1, type=int)
    args = parser.parse_args()
    main(args)

"""
Clean LightningDiT training entry point using Hugging Face Accelerate.

Supports:
  - single GPU: python train.py --config configs/...
  - single GPU via Accelerate: accelerate launch --num_processes 1 ...
  - multi GPU: accelerate launch --num_processes N ...
  - the repository's existing run_train.sh

Important design rule:
  Accelerate is the ONLY component that wraps the model for distributed
  training. Do not manually construct torch.nn.parallel.DistributedDataParallel.
"""

import argparse
import json
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.img_latent_dataset import ImgLatentDataset
from models.lightningdit import LightningDiT_models
from transport import create_transport


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_logger(logging_dir, is_main_process):
    """Create a real logger on rank 0 and a silent logger elsewhere."""
    logger = logging.getLogger(f"lightningdit.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    if is_main_process:
        formatter = logging.Formatter(
            "[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(
            os.path.join(logging_dir, "log.txt")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def requires_grad(model, flag=True):
    for param in model.parameters():
        param.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Move EMA parameters toward the current unwrapped model parameters."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    if ema_params.keys() != model_params.keys():
        missing = set(model_params) - set(ema_params)
        extra = set(ema_params) - set(model_params)
        raise RuntimeError(
            "EMA/model parameter names do not match. "
            f"Missing in EMA: {sorted(missing)[:5]}; "
            f"extra in EMA: {sorted(extra)[:5]}"
        )

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(
            param.detach(), alpha=1.0 - decay
        )


def strip_module_prefix(state_dict):
    """Allow checkpoints saved from old DDP code to load cleanly."""
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_weights_with_shape_check(model, checkpoint, verbose=False):
    """
    Load matching checkpoint tensors.

    Preserves LightningDiT's special handling for x_embedder.proj.weight
    when the latent input-channel count changes.
    """
    source_state = checkpoint.get("model", checkpoint)
    source_state = strip_module_prefix(source_state)
    target_state = model.state_dict()

    loaded = 0
    skipped = 0

    for name, source in source_state.items():
        if name not in target_state:
            skipped += 1
            if verbose:
                print(f"[weight_init] Parameter {name!r} not found; skipping.")
            continue

        target = target_state[name]

        if source.shape == target.shape:
            target.copy_(source)
            loaded += 1
            continue

        # LightningDiT special case: adapt only the input-channel dimension.
        if (
            name == "x_embedder.proj.weight"
            and source.ndim == target.ndim
            and source.shape[0] == target.shape[0]
            and source.shape[2:] == target.shape[2:]
        ):
            adapted = torch.zeros_like(target)
            channels = min(source.shape[1], target.shape[1])
            adapted[:, :channels].copy_(source[:, :channels])
            target.copy_(adapted)
            loaded += 1

            if verbose:
                print(
                    f"[weight_init] Adapted {name}: "
                    f"{tuple(source.shape)} -> {tuple(target.shape)}"
                )
            continue

        skipped += 1
        if verbose:
            print(
                f"[weight_init] Shape mismatch for {name!r}: "
                f"checkpoint={tuple(source.shape)}, "
                f"model={tuple(target.shape)}; skipping."
            )

    model.load_state_dict(target_state, strict=False)

    if verbose:
        print(f"[weight_init] Loaded {loaded} tensors; skipped {skipped}.")

    return model


def resolve_experiment_paths(train_config, accelerator):
    """
    Resolve one experiment name on rank 0, then broadcast it to every rank.

    This avoids the upstream bug where exp_name=None can cause different ranks
    to derive different output paths.
    """
    output_dir = train_config["train"]["output_dir"]

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    exp_name = train_config["train"].get("exp_name")

    if not exp_name:
        if accelerator.is_main_process:
            experiment_index = len(
                [p for p in glob(os.path.join(output_dir, "*")) if os.path.isdir(p)]
            )
            model_name = train_config["model"]["model_type"].replace("/", "-")
            exp_name = f"{experiment_index:03d}-{model_name}"
        else:
            exp_name = None

        payload = [exp_name]
        broadcast_object_list(payload, from_process=0)
        exp_name = payload[0]

    # Keep the resolved value in the config/checkpoint on every process.
    train_config["train"]["exp_name"] = exp_name

    experiment_dir = os.path.join(output_dir, exp_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    tensorboard_dir = os.path.join("tensorboard_logs", exp_name)

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    return experiment_dir, checkpoint_dir, tensorboard_dir


def checkpoint_step(path):
    try:
        return int(Path(path).stem)
    except ValueError:
        return -1


def resolve_resume_checkpoint(resume, checkpoint_dir):
    """
    resume may be:
      false / None -> no resume
      true         -> latest numeric *.pt in checkpoint_dir
      "/path/x.pt" -> explicit checkpoint
    """
    if resume is None or resume is False:
        return None

    if isinstance(resume, str):
        lowered = resume.strip().lower()
        if lowered in {"", "false", "no", "none", "0"}:
            return None
        if lowered not in {"true", "yes", "1"}:
            path = os.path.expanduser(resume)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Resume checkpoint not found: {path}")
            return path

    checkpoint_files = glob(os.path.join(checkpoint_dir, "*.pt"))
    checkpoint_files = [
        path for path in checkpoint_files if checkpoint_step(path) >= 0
    ]

    if not checkpoint_files:
        return None

    return max(checkpoint_files, key=checkpoint_step)


def build_model(train_config):
    downsample_ratio = train_config.get("vae", {}).get("downsample_ratio", 16)
    image_size = train_config["data"]["image_size"]

    if image_size % downsample_ratio != 0:
        raise ValueError(
            f"image_size={image_size} must be divisible by "
            f"downsample_ratio={downsample_ratio}."
        )

    latent_size = image_size // downsample_ratio
    model_cfg = train_config["model"]

    return LightningDiT_models[model_cfg["model_type"]](
        input_size=latent_size,
        num_classes=train_config["data"]["num_classes"],
        use_qknorm=model_cfg.get("use_qknorm", False),
        use_swiglu=model_cfg.get("use_swiglu", False),
        use_rope=model_cfg.get("use_rope", False),
        use_rmsnorm=model_cfg.get("use_rmsnorm", False),
        wo_shift=model_cfg.get("wo_shift", False),
        in_channels=model_cfg.get("in_chans", 4),
        use_checkpoint=model_cfg.get("use_checkpoint", False),
    )


def build_transport(train_config):
    cfg = train_config["transport"]
    return create_transport(
        cfg["path_type"],
        cfg["prediction"],
        cfg["loss_weight"],
        cfg["train_eps"],
        cfg["sample_eps"],
        use_cosine_loss=cfg.get("use_cosine_loss", False),
        use_lognorm=cfg.get("use_lognorm", False),
    )


def compute_loss(loss_dict):
    """
    Returns:
      objective: value used for backward()
      report:    primary diffusion loss used for logging/validation
    """
    primary = loss_dict["loss"].mean()

    if "cos_loss" in loss_dict:
        objective = primary + loss_dict["cos_loss"].mean()
    else:
        objective = primary

    return objective, primary


def move_batch_to_device(x, y, accelerator):
    # Prepared Accelerate DataLoaders normally already place batches on device,
    # but explicit placement keeps the function robust for custom datasets.
    if accelerator.mixed_precision == "no":
        x = x.to(accelerator.device, dtype=torch.float32, non_blocking=True)
    else:
        x = x.to(accelerator.device, non_blocking=True)

    y = y.to(accelerator.device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(model, loader, accelerator, transport):
    """Compute a globally reduced validation loss on one or many GPUs."""
    model.eval()

    local_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    local_sample_count = torch.zeros(
        (), device=accelerator.device, dtype=torch.float64
    )

    for x, y in loader:
        x, y = move_batch_to_device(x, y, accelerator)

        loss_dict = transport.training_losses(model, x, dict(y=y))
        _, report_loss = compute_loss(loss_dict)

        batch_size = x.shape[0]
        local_loss_sum += report_loss.detach().double() * batch_size
        local_sample_count += batch_size

    totals = torch.stack([local_loss_sum, local_sample_count])
    totals = accelerator.reduce(totals, reduction="sum")

    if totals[1].item() == 0:
        raise RuntimeError("Validation DataLoader produced zero samples.")

    model.train()
    return (totals[0] / totals[1]).float()


def save_checkpoint(
    model,
    ema,
    optimizer,
    train_config,
    train_steps,
    checkpoint_dir,
    accelerator,
    logger,
):
    """
    Save a checkpoint that is independent of DDP's `module.` wrapper.
    """
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        checkpoint = {
            "model": unwrapped_model.state_dict(),
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
            "config": train_config,
            "step": train_steps,
        }

        checkpoint_path = os.path.join(
            checkpoint_dir, f"{train_steps:07d}.pt"
        )
        accelerator.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    accelerator.wait_for_everyone()


def do_train(train_config, accelerator):
    device = accelerator.device

    # Rank-specific RNG streams are useful for diffusion noise while Accelerate
    # still coordinates distributed data loading.
    seed = int(train_config["train"].get("seed", 0))
    set_seed(seed, device_specific=True)

    experiment_dir, checkpoint_dir, tensorboard_dir = (
        resolve_experiment_paths(train_config, accelerator)
    )

    logger = create_logger(
        experiment_dir,
        is_main_process=accelerator.is_main_process,
    )

    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_text(
            "training configs",
            json.dumps(train_config, indent=4),
            global_step=0,
        )

    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(
        f"Processes: {accelerator.num_processes}; "
        f"device: {device}; mixed precision: {accelerator.mixed_precision}"
    )

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    model = build_model(train_config)

    weight_init = train_config["train"].get("weight_init")
    if weight_init:
        init_checkpoint = torch.load(
            os.path.expanduser(weight_init),
            map_location="cpu",
        )
        model = load_weights_with_shape_check(
            model,
            init_checkpoint,
            verbose=accelerator.is_main_process,
        )
        logger.info(f"Loaded pretrained initialization from {weight_init}")

    # Create EMA after weight initialization so both start identically.
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    num_param_tensors = len(list(model.parameters()))
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if num_trainable == 0:
        raise RuntimeError(
            "The training model has zero trainable parameters before "
            "accelerator.prepare()."
        )

    logger.info(
        f"LightningDiT parameters: {num_params / 1e6:.2f}M "
        f"({num_param_tensors} parameter tensors; "
        f"{num_trainable / 1e6:.2f}M trainable)"
    )

    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------
    optimizer_cfg = train_config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg["lr"],
        weight_decay=optimizer_cfg.get("weight_decay", 0.0),
        betas=(0.9, optimizer_cfg["beta2"]),
    )

    logger.info(
        f"Optimizer: AdamW, lr={optimizer_cfg['lr']}, "
        f"beta2={optimizer_cfg['beta2']}, "
        f"weight_decay={optimizer_cfg.get('weight_decay', 0.0)}"
    )

    transport = build_transport(train_config)
    logger.info(
        f"Transport: lognorm={train_config['transport'].get('use_lognorm', False)}, "
        f"cosine_loss={train_config['transport'].get('use_cosine_loss', False)}"
    )

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    grad_accum_steps = accelerator.gradient_accumulation_steps
    world_size = accelerator.num_processes
    requested_global_batch = int(train_config["train"]["global_batch_size"])
    batch_denominator = world_size * grad_accum_steps

    if requested_global_batch % batch_denominator != 0:
        raise ValueError(
            "train.global_batch_size must be divisible by "
            "(number of processes * gradient_accumulation_steps). "
            f"Got {requested_global_batch} / "
            f"({world_size} * {grad_accum_steps})."
        )

    batch_size_per_gpu = requested_global_batch // batch_denominator

    if batch_size_per_gpu < 1:
        raise ValueError(
            "Per-GPU micro-batch size became < 1. Increase global_batch_size "
            "or reduce gradient_accumulation_steps / GPU count."
        )

    data_cfg = train_config["data"]
    dataset = ImgLatentDataset(
        data_dir=data_cfg["data_path"],
        latent_norm=data_cfg.get("latent_norm", False),
        latent_multiplier=data_cfg.get("latent_multiplier", 0.18215),
    )

    if len(dataset) < batch_size_per_gpu:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but per-GPU batch size is "
            f"{batch_size_per_gpu}; with drop_last=True this would create "
            "an empty training loader."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(data_cfg["num_workers"] > 0),
    )

    valid_loader = None
    if data_cfg.get("valid_path"):
        valid_dataset = ImgLatentDataset(
            data_dir=data_cfg["valid_path"],
            latent_norm=data_cfg.get("latent_norm", False),
            latent_multiplier=data_cfg.get("latent_multiplier", 0.18215),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            persistent_workers=(data_cfg["num_workers"] > 0),
        )

        logger.info(
            f"Validation dataset: {len(valid_dataset):,} samples "
            f"from {data_cfg['valid_path']}"
        )

    logger.info(
        f"Training dataset: {len(dataset):,} samples from {data_cfg['data_path']}"
    )
    logger.info(
        f"Micro-batch/GPU: {batch_size_per_gpu}; "
        f"GPUs/processes: {world_size}; "
        f"gradient accumulation: {grad_accum_steps}; "
        f"effective global batch: "
        f"{batch_size_per_gpu * world_size * grad_accum_steps}"
    )

    # ---------------------------------------------------------------------
    # Resume BEFORE accelerator.prepare().
    # ---------------------------------------------------------------------
    train_steps = 0
    resume_checkpoint = resolve_resume_checkpoint(
        train_config["train"].get("resume", False),
        checkpoint_dir,
    )

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")

        model_state = strip_module_prefix(checkpoint["model"])
        model.load_state_dict(model_state, strict=True)

        if "ema" in checkpoint:
            ema.load_state_dict(strip_module_prefix(checkpoint["ema"]), strict=True)
        else:
            ema.load_state_dict(model_state, strict=True)

        if "opt" in checkpoint:
            optimizer.load_state_dict(checkpoint["opt"])

        train_steps = int(
            checkpoint.get("step", checkpoint_step(resume_checkpoint))
        )
        logger.info(
            f"Resuming from {resume_checkpoint} at step {train_steps}"
        )
    elif train_config["train"].get("resume", False):
        logger.info(
            f"Resume requested, but no checkpoint found in {checkpoint_dir}; "
            "starting from scratch."
        )

    # ---------------------------------------------------------------------
    # This is the ONLY distributed wrapping point.
    # ---------------------------------------------------------------------
    if valid_loader is None:
        model, optimizer, loader = accelerator.prepare(
            model, optimizer, loader
        )
    else:
        model, optimizer, loader, valid_loader = accelerator.prepare(
            model, optimizer, loader, valid_loader
        )

    # Under single GPU this is the same model; under DDP this removes the
    # DistributedDataParallel wrapper.
    unwrapped_model = accelerator.unwrap_model(model)

    model.train()
    ema.eval()

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    max_steps = int(train_config["train"]["max_steps"])
    log_every = int(train_config["train"]["log_every"])
    ckpt_every = int(train_config["train"]["ckpt_every"])
    max_grad_norm = optimizer_cfg.get("max_grad_norm")

    if log_every <= 0 or ckpt_every <= 0:
        raise ValueError("log_every and ckpt_every must both be > 0.")

    running_loss = 0.0
    running_microbatches = 0
    optimizer_steps_since_log = 0
    start_time = time()

    optimizer.zero_grad(set_to_none=True)

    if train_steps >= max_steps:
        logger.info(
            f"Checkpoint is already at step {train_steps} >= max_steps={max_steps}."
        )
    else:
        while train_steps < max_steps:
            for x, y in loader:
                with accelerator.accumulate(model):
                    x, y = move_batch_to_device(x, y, accelerator)
                    loss_dict = transport.training_losses(
                        model,
                        x,
                        dict(y=y),
                    )
                    loss, report_loss = compute_loss(loss_dict)

                    accelerator.backward(loss)

                    if (
                        max_grad_norm is not None
                        and accelerator.sync_gradients
                    ):
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            max_grad_norm,
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += float(report_loss.detach().float().item())
                running_microbatches += 1

                # One "train step" means one optimizer update, not one
                # gradient-accumulation micro-step.
                if not accelerator.sync_gradients:
                    continue

                update_ema(ema, unwrapped_model)
                train_steps += 1
                optimizer_steps_since_log += 1

                # ------------------------- logging -------------------------
                if train_steps % log_every == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                    elapsed = max(time() - start_time, 1e-12)
                    steps_per_sec = optimizer_steps_since_log / elapsed

                    stats = torch.tensor(
                        [running_loss, float(running_microbatches)],
                        device=device,
                        dtype=torch.float64,
                    )
                    stats = accelerator.reduce(stats, reduction="sum")
                    avg_loss = (stats[0] / stats[1]).item()

                    logger.info(
                        f"(step={train_steps:07d}) "
                        f"Train Loss: {avg_loss:.6f}, "
                        f"Optimizer Steps/Sec: {steps_per_sec:.3f}"
                    )

                    if writer is not None:
                        writer.add_scalar(
                            "Loss/train", avg_loss, train_steps
                        )

                    running_loss = 0.0
                    running_microbatches = 0
                    optimizer_steps_since_log = 0
                    start_time = time()

                # ------------------------ checkpoint -----------------------
                if train_steps % ckpt_every == 0:
                    save_checkpoint(
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        train_config=train_config,
                        train_steps=train_steps,
                        checkpoint_dir=checkpoint_dir,
                        accelerator=accelerator,
                        logger=logger,
                    )

                    if valid_loader is not None:
                        logger.info(
                            f"Starting validation at step {train_steps}"
                        )
                        val_loss = evaluate(
                            model,
                            valid_loader,
                            accelerator,
                            transport,
                        )

                        logger.info(
                            f"Validation Loss: {val_loss.item():.6f}"
                        )
                        if writer is not None:
                            writer.add_scalar(
                                "Loss/validation",
                                val_loss.item(),
                                train_steps,
                            )

                    if writer is not None:
                        writer.flush()

                if train_steps >= max_steps:
                    break

    # Save the exact final step if it was not already checkpointed.
    if train_steps > 0 and train_steps % ckpt_every != 0:
        save_checkpoint(
            model=model,
            ema=ema,
            optimizer=optimizer,
            train_config=train_config,
            train_steps=train_steps,
            checkpoint_dir=checkpoint_dir,
            accelerator=accelerator,
            logger=logger,
        )

    accelerator.wait_for_everyone()

    if writer is not None:
        writer.flush()
        writer.close()

    logger.info("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_positional",
        nargs="?",
        default=None,
        help="Optional positional config path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to LightningDiT YAML config.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default=None,
        help=(
            "Optional override. When omitted, Accelerate uses the value from "
            "the launcher/configuration."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config or args.config_positional or "configs/debug.yaml"
    train_config = load_config(config_path)

    grad_accum_steps = int(
        train_config["train"].get("gradient_accumulation_steps", 1)
    )
    if grad_accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=grad_accum_steps,
    )

    try:
        do_train(train_config, accelerator)
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from einops import rearrange

from taming.modules.losses.vqperceptual import *


class LPIPSWithDiscriminator(nn.Module):
    """
    Reconstruction + KL + adversarial loss.

    Supports both:
      - normal RGB VAE tensors: [B, 3, H, W]
      - high-frequency wavelet VAE tensors: [B, 9, H, W]

    IMPORTANT for the 9-channel HF VAE:
      - set perceptual_weight: 0.0
      - set disc_in_channels: 9

    Standard LPIPS/VGG is only used for 3-channel image tensors.
    """

    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        pp_style=False,
        vf_weight=1e2,
        adaptive_vf=False,
        cos_margin=0,
        distmat_margin=0,
        distmat_weight=1.0,
        cos_weight=1.0,
    ):
        super().__init__()

        assert disc_loss in ["hinge", "vanilla"]

        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight

        # --------------------------------------------------------------
        # LPIPS
        # --------------------------------------------------------------
        # Do not even instantiate LPIPS for the HF model when its weight is 0.
        # This avoids unnecessary VGG loading and prevents accidental use on
        # [B, 9, H, W] wavelet tensors.
        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = None

        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight

        # Learnable reconstruction log-variance.
        self.logvar = nn.Parameter(
            torch.ones(size=()) * logvar_init
        )

        # --------------------------------------------------------------
        # Discriminator
        # --------------------------------------------------------------
        # For HF VAE use disc_in_channels=9.
        self.disc_in_channels = disc_in_channels

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = (
            hinge_d_loss if disc_loss == "hinge"
            else vanilla_d_loss
        )

        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.pp_style = pp_style
        if pp_style:
            print("Using pp_style for nll loss")

        self.vf_weight = vf_weight
        self.adaptive_vf = adaptive_vf
        self.cos_margin = cos_margin
        self.distmat_margin = distmat_margin

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_shapes(self, inputs, reconstructions):
        if inputs.shape != reconstructions.shape:
            raise RuntimeError(
                "Reconstruction target and prediction must have identical shape. "
                f"inputs={tuple(inputs.shape)}, "
                f"reconstructions={tuple(reconstructions.shape)}"
            )

        if inputs.ndim != 4:
            raise RuntimeError(
                "Expected image/feature tensors with shape [B,C,H,W], "
                f"got {tuple(inputs.shape)}"
            )

    def _perceptual_term(self, inputs, reconstructions):
        """
        LPIPS is only meaningful/supported here for 3-channel image tensors.

        For the 9-channel HF VAE, configure:
            perceptual_weight: 0.0
        """
        if self.perceptual_weight <= 0:
            return None

        if self.perceptual_loss is None:
            return None

        if inputs.shape[1] != 3 or reconstructions.shape[1] != 3:
            raise ValueError(
                "LPIPS expects 3-channel tensors, but received "
                f"inputs={tuple(inputs.shape)} and "
                f"reconstructions={tuple(reconstructions.shape)}. "
                "For the 9-channel high-frequency VAE, set "
                "perceptual_weight: 0.0."
            )

        return self.perceptual_loss(
            inputs.contiguous(),
            reconstructions.contiguous(),
        )

    def _check_discriminator_channels(self, x):
        if x.shape[1] != self.disc_in_channels:
            raise RuntimeError(
                "Discriminator channel mismatch: "
                f"tensor has {x.shape[1]} channels but "
                f"disc_in_channels={self.disc_in_channels}. "
                "For the HF VAE use disc_in_channels: 9."
            )

    # ------------------------------------------------------------------
    # Adaptive weights
    # ------------------------------------------------------------------
    def calculate_adaptive_weight(
        self,
        nll_loss,
        g_loss,
        last_layer=None,
    ):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(
                nll_loss,
                last_layer,
                retain_graph=True,
            )[0]

            g_grads = torch.autograd.grad(
                g_loss,
                last_layer,
                retain_graph=True,
            )[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss,
                self.last_layer[0],
                retain_graph=True,
            )[0]

            g_grads = torch.autograd.grad(
                g_loss,
                self.last_layer[0],
                retain_graph=True,
            )[0]

        d_weight = (
            torch.norm(nll_grads)
            / (torch.norm(g_grads) + 1e-4)
        )

        d_weight = torch.clamp(
            d_weight,
            0.0,
            1e4,
        ).detach()

        d_weight = (
            d_weight
            * self.discriminator_weight
        )

        return d_weight

    def calculate_adaptive_weight_vf(
        self,
        nll_loss,
        vf_loss,
        last_layer=None,
    ):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(
                nll_loss,
                last_layer,
                retain_graph=True,
            )[0]

            vf_grads = torch.autograd.grad(
                vf_loss,
                last_layer,
                retain_graph=True,
            )[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss,
                self.last_layer[0],
                retain_graph=True,
            )[0]

            vf_grads = torch.autograd.grad(
                vf_loss,
                self.last_layer[0],
                retain_graph=True,
            )[0]

        vf_weight = (
            torch.norm(nll_grads)
            / (torch.norm(vf_grads) + 1e-4)
        )

        vf_weight = torch.clamp(
            vf_weight,
            0.0,
            1e8,
        ).detach()

        vf_weight = (
            vf_weight
            * self.vf_weight
        )

        return vf_weight

    # ------------------------------------------------------------------
    # Main loss
    # ------------------------------------------------------------------
    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
        z=None,
        aux_feature=None,
        enc_last_layer=None,
    ):
        self._check_shapes(
            inputs,
            reconstructions,
        )

        # --------------------------------------------------------------
        # Reconstruction + KL
        # --------------------------------------------------------------
        # Works directly for [B,9,H,W].
        rec_loss = torch.abs(
            inputs.contiguous()
            - reconstructions.contiguous()
        )

        # Optional LPIPS only for RGB tensors.
        p_loss = self._perceptual_term(
            inputs,
            reconstructions,
        )

        if p_loss is not None:
            rec_loss = (
                rec_loss
                + self.perceptual_weight * p_loss
            )

        if not self.pp_style:
            nll_loss = (
                rec_loss / torch.exp(self.logvar)
                + self.logvar
            )

            weighted_nll_loss = nll_loss

            if weights is not None:
                weighted_nll_loss = (
                    weights * nll_loss
                )

            weighted_nll_loss = (
                torch.sum(weighted_nll_loss)
                / weighted_nll_loss.shape[0]
            )

            nll_loss = (
                torch.sum(nll_loss)
                / nll_loss.shape[0]
            )

            kl_loss = posteriors.kl()

            kl_loss = (
                torch.sum(kl_loss)
                / kl_loss.shape[0]
            )

        else:
            nll_loss = rec_loss
            weighted_nll_loss = nll_loss

            if weights is not None:
                weighted_nll_loss = (
                    weights * nll_loss
                )

            weighted_nll_loss = torch.mean(
                weighted_nll_loss
            )

            nll_loss = torch.mean(
                nll_loss
            )

            kl_loss = posteriors.kl(
                no_sum=True
            )

            kl_loss = torch.mean(
                kl_loss
            )

        # --------------------------------------------------------------
        # Generator update
        # --------------------------------------------------------------
        if optimizer_idx == 0:
            self._check_discriminator_channels(
                reconstructions
            )

            if cond is None:
                assert not self.disc_conditional

                logits_fake = self.discriminator(
                    reconstructions.contiguous()
                )
            else:
                assert self.disc_conditional

                logits_fake = self.discriminator(
                    torch.cat(
                        (
                            reconstructions.contiguous(),
                            cond,
                        ),
                        dim=1,
                    )
                )

            g_loss = -torch.mean(
                logits_fake
            )

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss,
                        g_loss,
                        last_layer=last_layer,
                    )
                except RuntimeError:
                    # Validation/inference can legitimately lack the graph
                    # required for this adaptive gradient calculation.
                    if self.training:
                        raise

                    d_weight = torch.tensor(
                        0.0,
                        device=inputs.device,
                    )
            else:
                d_weight = torch.tensor(
                    0.0,
                    device=inputs.device,
                )

            # ----------------------------------------------------------
            # Optional visual-foundation feature loss
            # ----------------------------------------------------------
            # For the 9-channel HF VAE it is simplest to set use_vf=None.
            if (
                z is not None
                and aux_feature is not None
            ):
                z_flat = rearrange(
                    z,
                    "b c h w -> b c (h w)",
                )

                aux_feature_flat = rearrange(
                    aux_feature,
                    "b c h w -> b c (h w)",
                )

                z_norm = torch.nn.functional.normalize(
                    z_flat,
                    dim=1,
                )

                aux_feature_norm = torch.nn.functional.normalize(
                    aux_feature_flat,
                    dim=1,
                )

                z_cos_sim = torch.einsum(
                    "bci,bcj->bij",
                    z_norm,
                    z_norm,
                )

                aux_feature_cos_sim = torch.einsum(
                    "bci,bcj->bij",
                    aux_feature_norm,
                    aux_feature_norm,
                )

                diff = torch.abs(
                    z_cos_sim
                    - aux_feature_cos_sim
                )

                vf_loss_1 = torch.nn.functional.relu(
                    diff - self.distmat_margin
                ).mean()

                vf_loss_2 = torch.nn.functional.relu(
                    1
                    - self.cos_margin
                    - torch.nn.functional.cosine_similarity(
                        aux_feature,
                        z,
                    )
                ).mean()

                vf_loss = (
                    vf_loss_1 * self.distmat_weight
                    + vf_loss_2 * self.cos_weight
                )
            else:
                vf_loss = None

            disc_factor = adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
            )

            if vf_loss is not None:
                if self.adaptive_vf:
                    try:
                        vf_weight = self.calculate_adaptive_weight_vf(
                            nll_loss,
                            vf_loss,
                            last_layer=enc_last_layer,
                        )
                    except RuntimeError:
                        if self.training:
                            raise

                        vf_weight = torch.tensor(
                            0.0,
                            device=inputs.device,
                        )
                else:
                    vf_weight = self.vf_weight

                loss = (
                    weighted_nll_loss
                    + self.kl_weight * kl_loss
                    + d_weight * disc_factor * g_loss
                    + vf_weight * vf_loss
                )
            else:
                loss = (
                    weighted_nll_loss
                    + self.kl_weight * kl_loss
                    + d_weight * disc_factor * g_loss
                )

            log = {
                f"{split}/total_loss":
                    loss.clone().detach().mean(),

                f"{split}/logvar":
                    self.logvar.detach(),

                f"{split}/kl_loss":
                    kl_loss.detach().mean(),

                f"{split}/nll_loss":
                    nll_loss.detach().mean(),

                f"{split}/rec_loss":
                    rec_loss.detach().mean(),

                f"{split}/d_weight":
                    d_weight.detach(),

                f"{split}/disc_factor":
                    torch.tensor(
                        disc_factor,
                        device=inputs.device,
                    ),

                f"{split}/g_loss":
                    g_loss.detach().mean(),
            }

            if p_loss is not None:
                log[f"{split}/p_loss"] = (
                    p_loss.detach().mean()
                )

            if vf_loss is not None:
                log[f"{split}/vf_loss"] = (
                    vf_loss.detach().mean()
                )

                if not isinstance(
                    vf_weight,
                    float,
                ):
                    log[f"{split}/vf_weight"] = (
                        vf_weight.detach()
                    )
                else:
                    log[f"{split}/vf_weight"] = torch.tensor(
                        vf_weight,
                        device=inputs.device,
                    )

            return loss, log

        # --------------------------------------------------------------
        # Discriminator update
        # --------------------------------------------------------------
        if optimizer_idx == 1:
            self._check_discriminator_channels(
                inputs
            )
            self._check_discriminator_channels(
                reconstructions
            )

            if cond is None:
                assert not self.disc_conditional

                logits_real = self.discriminator(
                    inputs.contiguous().detach()
                )

                logits_fake = self.discriminator(
                    reconstructions.contiguous().detach()
                )
            else:
                assert self.disc_conditional

                logits_real = self.discriminator(
                    torch.cat(
                        (
                            inputs.contiguous().detach(),
                            cond,
                        ),
                        dim=1,
                    )
                )

                logits_fake = self.discriminator(
                    torch.cat(
                        (
                            reconstructions.contiguous().detach(),
                            cond,
                        ),
                        dim=1,
                    )
                )

            disc_factor = adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
            )

            d_loss = (
                disc_factor
                * self.disc_loss(
                    logits_real,
                    logits_fake,
                )
            )

            log = {
                f"{split}/disc_loss":
                    d_loss.clone().detach().mean(),

                f"{split}/logits_real":
                    logits_real.detach().mean(),

                f"{split}/logits_fake":
                    logits_fake.detach().mean(),
            }

            return d_loss, log

        raise ValueError(
            f"Unknown optimizer_idx={optimizer_idx}; expected 0 or 1."
        )


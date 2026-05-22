from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class DepthEncoder(nn.Module):
    """CNN + flatten + projection MLP for depth-image encoding."""

    def __init__(
        self,
        depth_height: int,
        depth_width: int,
        output_channels: list[int],
        kernel_size: list[int],
        stride: list[int],
        padding: str = "same",
        activation: str = "leakyrelu",
        max_pool: bool = True,
        global_pool: str = "none",
        flatten: bool = True,
        embedding_dim: int = 128,
        mlp_activation: str = "elu",
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        if not (len(output_channels) == len(kernel_size) == len(stride)):
            raise ValueError(
                "output_channels, kernel_size, and stride must have the same length, "
                f"got {len(output_channels)}, {len(kernel_size)}, {len(stride)}."
            )

        self.depth_height = depth_height
        self.depth_width = depth_width
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.cnn_encoder = self._build_cnn_encoder(
            in_channels=in_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
            max_pool=max_pool,
            global_pool=global_pool,
            flatten=flatten,
        )

        with torch.no_grad():
            dummy_depth = torch.zeros(1, in_channels, depth_height, depth_width)
            cnn_output = self.cnn_encoder(dummy_depth)
            if cnn_output.dim() != 2:
                raise RuntimeError(
                    f"Depth CNN encoder output must be 2D after flattening, got shape {cnn_output.shape}."
                )
            cnn_output_dim = cnn_output.shape[-1]

        self.output_dim = embedding_dim
        self.depth_embedding_mlp = nn.Sequential(
            nn.Linear(cnn_output_dim, embedding_dim),
            self._get_activation(mlp_activation),
        )

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        depth_image = self._format_depth_input(depth_image)
        cnn_features = self.cnn_encoder(depth_image)
        return self.depth_embedding_mlp(cnn_features)

    def _format_depth_input(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Convert input depth tensor to shape [B, C, H, W]."""
        if depth_image.dim() == 2:
            # [B, H*W]
            expected = self.depth_height * self.depth_width
            if depth_image.shape[-1] != expected:
                raise ValueError(
                    f"2D depth input must have shape [B, {expected}], got {depth_image.shape}."
                )
            depth_image = depth_image.view(-1, self.in_channels, self.depth_height, self.depth_width)

        elif depth_image.dim() == 3:
            # [B, H, W]
            if depth_image.shape[-2:] != (self.depth_height, self.depth_width):
                raise ValueError(
                    f"3D depth input must have spatial shape ({self.depth_height}, {self.depth_width}), "
                    f"got {depth_image.shape}."
                )
            depth_image = depth_image.unsqueeze(1)

        elif depth_image.dim() == 4:
            # [B, C, H, W]
            if depth_image.shape[1] != self.in_channels:
                raise ValueError(
                    f"Depth input channel mismatch: expected {self.in_channels}, got {depth_image.shape[1]}."
                )
            if depth_image.shape[-2:] != (self.depth_height, self.depth_width):
                raise ValueError(
                    f"4D depth input must have spatial shape ({self.depth_height}, {self.depth_width}), "
                    f"got {depth_image.shape}."
                )
        else:
            raise ValueError(
                f"Depth input must be 2D, 3D, or 4D, got tensor with shape {depth_image.shape}."
            )

        return depth_image

    def _build_cnn_encoder(
        self,
        in_channels: int,
        output_channels: list[int],
        kernel_size: list[int],
        stride: list[int],
        padding: str,
        activation: str,
        max_pool: bool,
        global_pool: str,
        flatten: bool,
    ) -> nn.Module:
        layers = []
        current_in_channels = in_channels

        for out_ch, k_size, s in zip(output_channels, kernel_size, stride):
            if padding.lower() == "same":
                pad = k_size // 2
            elif padding.lower() in ("valid", "none"):
                pad = 0
            else:
                raise ValueError(f"Unsupported padding mode: {padding}. Use 'same' or 'valid'.")

            layers.append(
                nn.Conv2d(
                    current_in_channels,
                    out_ch,
                    kernel_size=k_size,
                    stride=s,
                    padding=pad,
                )
            )
            layers.append(self._get_activation(activation))

            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            current_in_channels = out_ch

        if global_pool.lower() == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif global_pool.lower() == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        elif global_pool.lower() == "none":
            pass
        else:
            raise ValueError(f"Unsupported global_pool mode: {global_pool}.")

        if flatten:
            layers.append(nn.Flatten())

        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        key = name.lower()
        if key not in activations:
            raise ValueError(f"Unsupported activation: {name}.")
        return activations[key]

class PolicyHead(nn.Module):
    """Policy MLP head operating on concatenated latent features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    self._get_activation(activation),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.policy_net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.input_dim:
            raise ValueError(
                f"PolicyHead expected latent dim {self.input_dim}, got {latent.shape[-1]}."
            )
        return self.policy_net(latent)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(negative_slope=0.01),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        key = name.lower()
        if key not in activations:
            raise ValueError(f"Unsupported activation: {name}.")
        return activations[key]

class StudentCNNPolicy(nn.Module):
    """CNN-based student policy model for distillation.

    Architecture:
        depth_image -> DepthEncoder -> depth_embedding
        noise_policy -> optional normalization
        concat(noise_policy, depth_embedding) -> PolicyHead -> output/distribution
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        depth_encoder: DepthEncoder,
        policy_head: PolicyHead,
        proprio_group: str = "noise_policy",
        depth_group: str = "depth_image",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        self.obs_groups = obs_groups[obs_set]
        self.proprio_group = proprio_group
        self.depth_group = depth_group
        self.obs_normalization = obs_normalization

        self.proprio_dim, self.depth_shape = self._get_obs_specs(obs)

        self.depth_encoder = depth_encoder
        self.policy_head = policy_head

        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.proprio_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            dist_cfg = copy.deepcopy(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            required_head_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            required_head_output_dim = output_dim

        if self.policy_head.output_dim != required_head_output_dim:
            raise ValueError(
                f"Policy head output_dim ({self.policy_head.output_dim}) must match "
                f"required output dim ({required_head_output_dim})."
            )

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.policy_head.policy_net)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        head_output = self.policy_head(latent)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(head_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(head_output)

        return head_output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        proprio = obs[self.proprio_group]
        depth = obs[self.depth_group]

        if len(proprio.shape) != 2:
            raise ValueError(
                f"Expected proprio observation '{self.proprio_group}' to be 2D, got {proprio.shape}."
            )

        proprio = self.obs_normalizer(proprio)
        depth_embedding = self.depth_encoder(depth)
        latent = torch.cat([proprio, depth_embedding], dim=-1)
        return latent

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        return self.depth_encoder(depth)

    def policy_from_features(
        self,
        proprio: torch.Tensor,
        depth_embedding: torch.Tensor,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        proprio = self.obs_normalizer(proprio)
        latent = torch.cat([proprio, depth_embedding], dim=-1)
        head_output = self.policy_head(latent)

        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(head_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(head_output)

        return head_output

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("output_mean is only available when a distribution is configured.")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("output_std is only available when a distribution is configured.")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("output_entropy is only available when a distribution is configured.")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        if self.distribution is None:
            raise RuntimeError("output_distribution_params is only available when a distribution is configured.")
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("get_output_log_prob is only available when a distribution is configured.")
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("get_kl_divergence is only available when a distribution is configured.")
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _TorchStudentCNNPolicy(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxStudentCNNPolicy(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            proprio = obs[self.proprio_group]
            self.obs_normalizer.update(proprio)  # type: ignore

    def _get_obs_specs(self, obs: TensorDict) -> tuple[int, tuple[int, ...]]:
        proprio = obs[self.proprio_group]
        depth = obs[self.depth_group]

        if len(proprio.shape) != 2:
            raise ValueError(
                f"Expected proprio observation '{self.proprio_group}' to be 2D, got {proprio.shape}."
            )

        if depth.dim() not in (2, 3, 4):
            raise ValueError(
                f"Expected depth observation '{self.depth_group}' to be 2D/3D/4D, got {depth.shape}."
            )

        proprio_dim = proprio.shape[-1]
        depth_shape = tuple(depth.shape[1:]) if depth.dim() > 1 else tuple(depth.shape)
        return proprio_dim, depth_shape
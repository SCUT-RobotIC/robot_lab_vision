from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


# ======================================================================== #
#  Sub-network A: DepthEncoder                                             #
# ======================================================================== #

class DepthEncoder(nn.Module):
    """CNN + flatten + projection MLP encoder for depth images.

    Sub-network A in StudentCNNPolicy:
        a: depth_image -> DepthEncoder (A) -> c: depth_embedding

    After training StudentCNNPolicy (Z), this module can be independently
    extracted and deployed via its own forward() / as_jit() / as_onnx().
    """

    is_recurrent: bool = False
    """Whether the module contains a recurrent component."""

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
        """Initialize the depth encoder.

        Args:
            depth_height: Height of the input depth image.
            depth_width: Width of the input depth image.
            output_channels: Number of output channels for each Conv2d layer.
            kernel_size: Kernel sizes for each Conv2d layer.
            stride: Strides for each Conv2d layer.
            padding: Padding mode, ``'same'`` or ``'valid'``.
            activation: Activation function for CNN layers.
            max_pool: Whether to apply MaxPool2d (kernel=2, stride=2) after each conv layer.
            global_pool: Global pooling mode after all conv layers: ``'avg'``, ``'max'``, or ``'none'``.
            flatten: Whether to flatten the CNN output before the projection MLP.
            embedding_dim: Output dimension of the depth embedding (A's output_dim, exposed to Z).
            mlp_activation: Activation function for the projection MLP.
            in_channels: Number of input image channels.
        """
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
        # Exposed so StudentCNNPolicy (Z) can auto-compute PolicyHead input_dim
        self.output_dim = embedding_dim

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

        self.depth_embedding_mlp = nn.Sequential(
            nn.Linear(cnn_output_dim, embedding_dim),
            self._get_activation(mlp_activation),
        )

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the depth encoder (sub-network A).

        Serves dual roles:
        - **Standalone deployment**: call ``model(depth_image)`` directly after extraction from Z.
        - **Full-network inference**: called internally by ``StudentCNNPolicy.get_latent()``.

        Args:
            depth_image: Input depth image; accepts shape ``[B, H*W]``, ``[B, H, W]``,
                         or ``[B, C, H, W]``.

        Returns:
            Depth embedding ``c`` of shape ``[B, embedding_dim]``.
        """
        depth_image = self._format_depth_input(depth_image)
        cnn_features = self.cnn_encoder(depth_image)
        return self.depth_embedding_mlp(cnn_features)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset internal state (no-op for non-recurrent)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return hidden state (``None`` for non-recurrent)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden state for truncated backpropagation (no-op for non-recurrent)."""
        pass

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible version for standalone deployment of A."""
        return _TorchDepthEncoder(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible version for standalone deployment of A."""
        return _OnnxDepthEncoder(self, verbose)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _format_depth_input(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Normalize depth tensor to shape ``[B, C, H, W]``."""
        if depth_image.dim() == 2:
            expected = self.depth_height * self.depth_width
            if depth_image.shape[-1] != expected:
                raise ValueError(
                    f"2D depth input must have shape [B, {expected}], got {depth_image.shape}."
                )
            depth_image = depth_image.view(-1, self.in_channels, self.depth_height, self.depth_width)

        elif depth_image.dim() == 3:
            if depth_image.shape[-2:] != (self.depth_height, self.depth_width):
                raise ValueError(
                    f"3D depth input must have spatial shape ({self.depth_height}, {self.depth_width}), "
                    f"got {depth_image.shape}."
                )
            depth_image = depth_image.unsqueeze(1)

        elif depth_image.dim() == 4:
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
                f"Depth input must be 2D, 3D, or 4D, got shape {depth_image.shape}."
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

            layers.append(nn.Conv2d(current_in_channels, out_ch, kernel_size=k_size, stride=s, padding=pad))
            layers.append(self._get_activation(activation))

            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            current_in_channels = out_ch

        if global_pool.lower() == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif global_pool.lower() == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        elif global_pool.lower() != "none":
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


# ======================================================================== #
#  Sub-network B: PolicyHead                                               #
# ======================================================================== #

class PolicyHead(nn.Module):
    """MLP policy head operating on concatenated latent features.

    Sub-network B in StudentCNNPolicy:
        concat(b: proprio_normalized, c: depth_embedding) -> PolicyHead (B) -> output

    After training StudentCNNPolicy (Z), this module can be independently extracted
    and deployed. During standalone deployment it receives pre-computed depth embeddings
    from DepthEncoder (A) concatenated with proprioceptive observations.
    """

    is_recurrent: bool = False
    """Whether the module contains a recurrent component."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
    ) -> None:
        """Initialize the policy head.

        Args:
            input_dim: Input dimension. Equals ``proprio_dim + depth_encoder.output_dim``.
                       Auto-computed by StudentCNNPolicy (Z); set manually for standalone use.
            output_dim: Output dimension. Equals ``distribution.input_dim`` when using a
                        distribution, otherwise equals the final action dim.
                        Auto-computed by StudentCNNPolicy (Z).
            hidden_dims: Hidden layer dimensions of the MLP.
            activation: Activation function for the MLP.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), self._get_activation(activation)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.policy_net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass of the policy head (sub-network B).

        Serves dual roles:
        - **Standalone deployment**: call ``model(cat(proprio, depth_embedding))`` directly.
        - **Full-network inference**: called internally by ``StudentCNNPolicy.forward()``.

        Args:
            latent: Concatenated ``[proprio_normalized, depth_embedding]`` of shape
                    ``[B, input_dim]``.

        Returns:
            Policy output of shape ``[B, output_dim]``.
        """
        if latent.shape[-1] != self.input_dim:
            raise ValueError(
                f"PolicyHead expected latent dim {self.input_dim}, got {latent.shape[-1]}."
            )
        return self.policy_net(latent)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset internal state (no-op for non-recurrent)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return hidden state (``None`` for non-recurrent)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden state for truncated backpropagation (no-op for non-recurrent)."""
        pass

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible version for standalone deployment of B."""
        return _TorchPolicyHead(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible version for standalone deployment of B."""
        return _OnnxPolicyHead(self, verbose)

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


# ======================================================================== #
#  Full network Z: StudentCNNPolicy                                        #
# ======================================================================== #

class StudentCNNPolicy(nn.Module):
    """CNN-based student policy model for distillation.

    Full network Z integrating DepthEncoder (A) and PolicyHead (B):

        a: depth_image  ──> DepthEncoder  (A) ──> c: depth_embedding ──┐
                                                                         ├─ cat ──> PolicyHead (B) ──> output
        b: proprio      ─────────────────────────────────────────────────┘

    After training, sub-networks A and B can be independently deployed:

    .. code-block:: python

        # Extract and deploy A
        depth_encoder_jit = model.depth_encoder.as_jit()
        # Extract and deploy B
        policy_head_jit = model.policy_head.as_jit()
    """

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        depth_encoder_cfg: dict,
        policy_head_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        policy_head_activation: str = "elu",
        proprio_group: str = "noise_policy",
        depth_group: str = "depth_image",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize StudentCNNPolicy (Z).

        Builds DepthEncoder (A) and PolicyHead (B) internally. PolicyHead's
        ``input_dim`` is auto-computed as ``proprio_dim + depth_encoder.output_dim``,
        mirroring how MLPModel auto-computes the MLP input from ``_get_latent_dim()``.

        Args:
            obs: Observation TensorDict used to infer the proprioceptive input dimension.
            obs_groups: Dictionary mapping observation set names to lists of obs group keys.
            obs_set: Key into ``obs_groups`` selecting the active observation set.
            output_dim: Final output dimension (e.g., action space size).
            depth_encoder_cfg: Config dict forwarded verbatim to ``DepthEncoder.__init__``.
                               Must include ``depth_height``, ``depth_width``,
                               ``output_channels``, ``kernel_size``, ``stride``, etc.
            policy_head_hidden_dims: Hidden layer dims for the PolicyHead MLP.
            policy_head_activation: Activation function for the PolicyHead MLP.
            proprio_group: obs key for proprioceptive input ``b``.
            depth_group: obs key for depth image input ``a``.
            obs_normalization: Whether to apply EmpiricalNormalization to proprio ``b``.
            distribution_cfg: Config dict for the output distribution. If provided, the
                               model can output stochastic values sampled from the distribution.
        """
        super().__init__()

        self.obs_groups = obs_groups[obs_set]
        self.proprio_group = proprio_group
        self.depth_group = depth_group
        self.obs_normalization = obs_normalization

        # Infer proprio_dim (and depth_shape for bookkeeping) from obs
        self.proprio_dim, self.depth_shape = self._get_obs_specs(obs)

        # ---- Build sub-network A: DepthEncoder -------------------------
        self.depth_encoder = DepthEncoder(**depth_encoder_cfg)

        # ---- Proprio (b) normalization ---------------------------------
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.proprio_dim)
        else:
            self.obs_normalizer = nn.Identity()

        # ---- Distribution ----------------------------------------------
        if distribution_cfg is not None:
            dist_cfg = copy.deepcopy(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore
            distribution: Distribution = dist_class(output_dim, **dist_cfg)
            self.distribution = distribution
            policy_head_output_dim = distribution.input_dim
        else:
            self.distribution = None
            policy_head_output_dim = output_dim

        # ---- Build sub-network B: PolicyHead ---------------------------
        # input_dim is auto-computed: proprio_dim + depth_encoder.output_dim
        self.policy_head = PolicyHead(
            input_dim=self._get_latent_dim(),
            output_dim=policy_head_output_dim,
            hidden_dims=list(policy_head_hidden_dims),
            activation=policy_head_activation,
        )

        # Initialize distribution-specific weights in PolicyHead
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.policy_head.policy_net)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of StudentCNNPolicy (Z).

        Orchestrates A and B in three steps that mirror the MLPModel template:

        1. ``get_latent()``:
               ``a`` ──> ``A.forward()`` ──> ``c``
               ``latent = cat(obs_normalizer(b), c)``
        2. ``B.forward(latent)`` ──> ``head_output``
        3. Distribution handling (or direct return).

        Args:
            obs: TensorDict containing ``proprio_group`` (b) and ``depth_group`` (a).
            masks: Optional trajectory masks; used to unpad non-recurrent models
                   trained with recurrent padding.
            hidden_state: Recurrent hidden state (unused for non-recurrent model).
            stochastic_output: If ``True`` and a distribution is configured, sample
                               stochastically. Defaults to ``False`` (deterministic).

        Returns:
            Output tensor of shape ``[B, output_dim]``.
        """
        # Unpad padded trajectories when the model is non-recurrent
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Step 1: build combined latent (runs A internally)
        latent = self.get_latent(obs, masks, hidden_state)
        # Step 2: run B (PolicyHead)
        head_output = self.policy_head(latent)
        # Step 3: distribution handling
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
        """Build the combined latent fed into PolicyHead (B).

        Runs DepthEncoder (A) on depth input ``a``, normalizes proprio input ``b``,
        then concatenates:

        .. code-block::

            latent = cat(obs_normalizer(b), A(a))   # [B, proprio_dim + embedding_dim]
        """
        proprio = obs[self.proprio_group]  # b: [B, proprio_dim]
        depth = obs[self.depth_group]       # a: raw depth image

        if len(proprio.shape) != 2:
            raise ValueError(
                f"Expected proprio observation '{self.proprio_group}' to be 2D, got {proprio.shape}."
            )

        # Normalize b
        proprio = self.obs_normalizer(proprio)
        # Run A: a -> c = depth_embedding
        depth_embedding = self.depth_encoder(depth)
        # Concatenate b_normalized and c
        latent = torch.cat([proprio, depth_embedding], dim=-1)
        return latent

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Convenience: run only DepthEncoder (A) on a depth image."""
        return self.depth_encoder(depth)

    def policy_from_features(
        self,
        proprio: torch.Tensor,
        depth_embedding: torch.Tensor,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run PolicyHead (B) from pre-computed features.

        Useful when depth embeddings are cached or when A and B are pipelined
        separately in a deployment scenario.
        """
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
        """Reset internal state (no-op for non-recurrent)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return recurrent hidden state (``None`` for non-recurrent)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        if self.distribution is None:
            raise RuntimeError("output_mean is only available when a distribution is configured.")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        if self.distribution is None:
            raise RuntimeError("output_std is only available when a distribution is configured.")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        if self.distribution is None:
            raise RuntimeError("output_entropy is only available when a distribution is configured.")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        if self.distribution is None:
            raise RuntimeError("output_distribution_params is only available when a distribution is configured.")
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities under the current distribution."""
        if self.distribution is None:
            raise RuntimeError("get_output_log_prob is only available when a distribution is configured.")
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute KL divergence between two distribution parameterizations."""
        if self.distribution is None:
            raise RuntimeError("get_kl_divergence is only available when a distribution is configured.")
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible version of the full model Z."""
        return _TorchStudentCNNPolicy(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible version of the full model Z."""
        return _OnnxStudentCNNPolicy(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update EmpiricalNormalization statistics from a batch of proprio observations."""
        if self.obs_normalization:
            proprio = obs[self.proprio_group]
            self.obs_normalizer.update(proprio)  # type: ignore

    def _get_latent_dim(self) -> int:
        """Return the combined latent dimension fed into PolicyHead (B).

        Equals ``proprio_dim + depth_encoder.output_dim``.
        Mirrors ``MLPModel._get_latent_dim()`` and is used to auto-construct PolicyHead.
        """
        return self.proprio_dim + self.depth_encoder.output_dim

    def _get_obs_specs(self, obs: TensorDict) -> tuple[int, tuple[int, ...]]:
        """Infer proprio dimension and depth tensor shape from the obs TensorDict."""
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


class _TorchDepthEncoder(nn.Module):
    def __init__(self, model: DepthEncoder) -> None:
        super().__init__()
        self.cnn_encoder = copy.deepcopy(model.cnn_encoder)
        self.depth_embedding_mlp = copy.deepcopy(model.depth_embedding_mlp)
        self.depth_height = model.depth_height
        self.depth_width = model.depth_width
        self.in_channels = model.in_channels

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        if depth_image.dim() == 2:
            expected = self.depth_height * self.depth_width
            if depth_image.shape[-1] != expected:
                raise ValueError(
                    f"2D depth input must have shape [B, {expected}], got {depth_image.shape}."
                )
            depth_image = depth_image.view(-1, self.in_channels, self.depth_height, self.depth_width)
        elif depth_image.dim() == 3:
            if depth_image.shape[-2:] != (self.depth_height, self.depth_width):
                raise ValueError(
                    f"3D depth input must have spatial shape ({self.depth_height}, {self.depth_width}), got {depth_image.shape}."
                )
            depth_image = depth_image.unsqueeze(1)
        elif depth_image.dim() == 4:
            if depth_image.shape[1] != self.in_channels:
                raise ValueError(
                    f"Depth input channel mismatch: expected {self.in_channels}, got {depth_image.shape[1]}."
                )
            if depth_image.shape[-2:] != (self.depth_height, self.depth_width):
                raise ValueError(
                    f"4D depth input must have spatial shape ({self.depth_height}, {self.depth_width}), got {depth_image.shape}."
                )
        else:
            raise ValueError(f"Depth input must be 2D, 3D, or 4D, got shape {depth_image.shape}.")

        cnn_features = self.cnn_encoder(depth_image)
        return self.depth_embedding_mlp(cnn_features)


class _OnnxDepthEncoder(_TorchDepthEncoder):
    def __init__(self, model: DepthEncoder, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.in_channels, self.depth_height, self.depth_width),)

    @property
    def input_names(self) -> list[str]:
        return ["depth_image"]

    @property
    def output_names(self) -> list[str]:
        return ["depth_embedding"]


class _TorchPolicyHead(nn.Module):
    def __init__(self, model: PolicyHead) -> None:
        super().__init__()
        self.policy_net = copy.deepcopy(model.policy_net)
        self.input_dim = model.input_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.input_dim:
            raise ValueError(
                f"PolicyHead expected latent dim {self.input_dim}, got {latent.shape[-1]}."
            )
        return self.policy_net(latent)


class _OnnxPolicyHead(_TorchPolicyHead):
    def __init__(self, model: PolicyHead, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_dim),)

    @property
    def input_names(self) -> list[str]:
        return ["latent"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _TorchStudentCNNPolicy(nn.Module):
    def __init__(self, model: StudentCNNPolicy) -> None:
        super().__init__()
        self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.policy_head = copy.deepcopy(model.policy_head)
        self.proprio_group = model.proprio_group
        self.depth_group = model.depth_group
        self.proprio_dim = model.proprio_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        proprio = obs[self.proprio_group]
        depth = obs[self.depth_group]

        if proprio.dim() != 2:
            raise ValueError(
                f"Expected proprio observation '{self.proprio_group}' to be 2D, got {proprio.shape}."
            )

        proprio = self.obs_normalizer(proprio)
        depth_embedding = self.depth_encoder(depth)
        latent = torch.cat([proprio, depth_embedding], dim=-1)
        return self.deterministic_output(self.policy_head(latent))


class _OnnxStudentCNNPolicy(_TorchStudentCNNPolicy):
    def __init__(self, model: StudentCNNPolicy, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.proprio_dim),
            torch.zeros(1, self.depth_encoder.in_channels, self.depth_encoder.depth_height, self.depth_encoder.depth_width),
        )

    @property
    def input_names(self) -> list[str]:
        return ["proprio", "depth_image"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    def forward(self, proprio: torch.Tensor, depth_image: torch.Tensor) -> torch.Tensor:
        if proprio.dim() != 2:
            raise ValueError(f"Expected proprio input to be 2D, got {proprio.shape}.")
        proprio = self.obs_normalizer(proprio)
        depth_embedding = self.depth_encoder(depth_image)
        latent = torch.cat([proprio, depth_embedding], dim=-1)
        return self.deterministic_output(self.policy_head(latent))
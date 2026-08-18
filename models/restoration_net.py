"""
Severity-Gated NAF-Bottleneck restoration network.

Blind restoration of grayscale images degraded by an unknown combination of additive Gaussian
noise, multiplicative speckle noise, and 2x spatial downsampling. Built from NAFNet-style gated
conv blocks, an implicit severity embedding injected via FiLM at every stage, and a single
lightweight global-context block at the bottleneck resolution.

References (mechanisms reproduced, not literally copied):
  - NAFNet:    Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
  - Restormer: Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image
               Restoration", CVPR 2022 (channel-wise attention reproduced in LightweightGlobalContext).
  - AirNet / PromptIR: implicit, continuous degradation conditioning reproduced via
               SeverityEmbedding + FiLM instead of prompt banks / cross-attention.
  - ESPCN:     Shi et al., "Real-Time Single Image and Video Super-Resolution Using an
               Efficient Sub-Pixel CNN", CVPR 2016 (PixelShuffle + ICNR init).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over NCHW tensors. Reduction runs in fp32 even under autocast(fp16)."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x32 = x.float()
        mu = x32.mean(dim=1, keepdim=True)
        var = x32.var(dim=1, keepdim=True, unbiased=False)
        x32 = (x32 - mu) / torch.sqrt(var + self.eps)
        out = x32 * self.weight.float().view(1, -1, 1, 1) + self.bias.float().view(1, -1, 1, 1)
        return out.to(orig_dtype)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies them elementwise -- NAFNet's activation replacement."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global-pool + 1x1 conv channel gate (NAFNet's SCA)."""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class FiLM(nn.Module):
    """Feature-wise linear modulation: maps the severity embedding to a per-channel scale/shift.
    Zero-initialized so it starts as an identity transform."""

    def __init__(self, embed_dim, channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(embed_dim, channels * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, embedding):
        scale, shift = self.to_scale_shift(embedding).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift


class SeverityGate(nn.Module):
    """Predicts a per-channel residual-branch scale from the severity embedding, in place of a
    static learned constant. Zero-initialized, so it starts identical to the static-beta/gamma case."""

    def __init__(self, embed_dim, channels):
        super().__init__()
        self.to_scale = nn.Linear(embed_dim, channels)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)

    def forward(self, embedding):
        return self.to_scale(embedding).unsqueeze(-1).unsqueeze(-1)


class NAFBlock(nn.Module):
    """
    NAFNet-style block: LayerNorm -> 1x1 expand -> depthwise 3x3 -> SimpleGate -> channel
    attention -> 1x1 project (residual), followed by a LayerNorm -> 1x1 expand -> SimpleGate ->
    1x1 project feed-forward (residual). If embed_dim is given, a FiLM layer conditioned on the
    severity embedding is applied after the block.

    severity_gated_residuals: when true, the residual-branch scales (beta/gamma) are predicted
    from the severity embedding via SeverityGate instead of being static learned constants.
    """

    def __init__(self, channels, expand_ratio=2, ffn_expand_ratio=2, embed_dim=None, severity_gated_residuals=False):
        super().__init__()
        dw_channels = channels * expand_ratio
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        ffn_channels = channels * ffn_expand_ratio
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.severity_gated_residuals = severity_gated_residuals and embed_dim is not None
        if self.severity_gated_residuals:
            self.beta_gate = SeverityGate(embed_dim, channels)
            self.gamma_gate = SeverityGate(embed_dim, channels)
        else:
            self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.film = FiLM(embed_dim, channels) if embed_dim is not None else None

    def forward(self, x, embedding=None):
        beta = self.beta_gate(embedding) if self.severity_gated_residuals else self.beta
        gamma = self.gamma_gate(embedding) if self.severity_gated_residuals else self.gamma

        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        x = x + y * gamma

        if self.film is not None and embedding is not None:
            x = self.film(x, embedding)
        return x


class SeverityEmbedding(nn.Module):
    """Global-average-pools the stem features, then an MLP produces a compact embedding consumed
    by FiLM throughout the network. No explicit degradation label is ever provided as input."""

    def __init__(self, in_channels, embed_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        return self.mlp(pooled)


class LightweightGlobalContext(nn.Module):
    """
    The network's only attention op, placed at the bottleneck resolution. Reproduces Restormer's
    MDTA idea: attention computed across the channel dimension instead of spatial tokens, so cost
    is O(C^2) rather than O((HW)^2).

    The q/k normalize + attention matmul runs in fp32 regardless of the surrounding autocast
    context -- F.normalize's default eps underflows to 0 in fp16 on flat/uniform regions, which
    otherwise produces exploding gradients through that division's backward pass.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.norm = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)
        qkv = self.qkv_dw(self.qkv(y))
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t):
            return t.view(b, self.num_heads, c // self.num_heads, h * w)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q32 = F.normalize(q.float(), dim=-1, eps=1e-6)
            k32 = F.normalize(k.float(), dim=-1, eps=1e-6)
            attn = (q32 @ k32.transpose(-2, -1)) * self.temperature.float()
            attn = attn.softmax(dim=-1)
            out = attn @ v.float()

        out = out.to(x.dtype).reshape(b, c, h, w)
        return x + self.proj(out)


def icnr_init(conv_weight, scale=2):
    """ICNR initialization (Aitken et al., 2017) for sub-pixel-convolution weights -- removes the
    checkerboard-pattern bias PixelShuffle otherwise starts training with."""
    out_channels, in_channels, kh, kw = conv_weight.shape
    sub_channels = out_channels // (scale ** 2)
    sub_kernel = torch.empty(sub_channels, in_channels, kh, kw)
    nn.init.kaiming_normal_(sub_kernel, nonlinearity="relu")
    kernel = sub_kernel.repeat_interleave(scale ** 2, dim=0)
    conv_weight.data.copy_(kernel)


class PixelShuffleUpsample(nn.Module):
    """Sub-pixel-convolution upsampling block: 3x3 conv to scale^2 * channels, then PixelShuffle."""

    def __init__(self, channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale ** 2), kernel_size=3, padding=1)
        icnr_init(self.conv.weight, scale=scale)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class RestorationNet(nn.Module):
    """
    Severity-Gated NAF-Bottleneck restoration network (single-channel, fixed 2x SR).

    Pipeline:
      stem conv
        -> severity embedding estimated once from stem features
        -> shallow encoder (NAFBlocks, FiLM-modulated) at input resolution        [skip saved here]
        -> strided downsample (input resolution -> bottleneck resolution)
        -> bottleneck (NAFBlocks, FiLM-modulated) + one global-context block
        -> stage-1 PixelShuffle upsample (bottleneck resolution -> input resolution),
           fused with the encoder skip connection
        -> decoder (NAFBlocks, FiLM-modulated) at input resolution
        -> stage-2 PixelShuffle upsample (input resolution -> 2x input resolution): the required
           super-resolution step
        -> refinement conv -> sigmoid

    The sigmoid output lands directly in [0,1]. Only the input is clamped to a configurable
    robust range (input_clip_range) to absorb out-of-range NoisyLR values.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        num_encoder_blocks=2,
        num_bottleneck_blocks=6,
        num_decoder_blocks=2,
        embed_dim=64,
        scale_factor=2,
        use_global_context=True,
        global_context_heads=4,
        input_clip_range=(-0.5, 1.5),
        severity_gated_residuals=False,
    ):
        super().__init__()
        if scale_factor != 2:
            raise ValueError("Only the officially specified 2x scale factor is implemented.")
        self.input_clip_range = tuple(input_clip_range)
        self.scale_factor = scale_factor

        bottleneck_channels = base_channels * 2

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.severity_embed = SeverityEmbedding(base_channels, embed_dim)

        self.encoder = nn.ModuleList(
            [
                NAFBlock(base_channels, embed_dim=embed_dim, severity_gated_residuals=severity_gated_residuals)
                for _ in range(num_encoder_blocks)
            ]
        )
        self.downsample = nn.Conv2d(base_channels, bottleneck_channels, kernel_size=2, stride=2)

        self.bottleneck = nn.ModuleList(
            [
                NAFBlock(bottleneck_channels, embed_dim=embed_dim, severity_gated_residuals=severity_gated_residuals)
                for _ in range(num_bottleneck_blocks)
            ]
        )
        self.global_context = (
            LightweightGlobalContext(bottleneck_channels, num_heads=global_context_heads)
            if use_global_context
            else None
        )

        self.upsample_to_input_res = PixelShuffleUpsample(bottleneck_channels, scale=2)
        self.skip_fuse = nn.Conv2d(bottleneck_channels + base_channels, bottleneck_channels, kernel_size=1)

        self.decoder = nn.ModuleList(
            [
                NAFBlock(bottleneck_channels, embed_dim=embed_dim, severity_gated_residuals=severity_gated_residuals)
                for _ in range(num_decoder_blocks)
            ]
        )

        self.upsample_sr = PixelShuffleUpsample(bottleneck_channels, scale=2)
        self.refine = nn.Sequential(
            nn.Conv2d(bottleneck_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

        self.config = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            num_encoder_blocks=num_encoder_blocks,
            num_bottleneck_blocks=num_bottleneck_blocks,
            num_decoder_blocks=num_decoder_blocks,
            embed_dim=embed_dim,
            scale_factor=scale_factor,
            use_global_context=use_global_context,
            global_context_heads=global_context_heads,
            input_clip_range=list(self.input_clip_range),
            severity_gated_residuals=severity_gated_residuals,
        )

    def forward(self, x):
        x = torch.clamp(x, *self.input_clip_range)

        # Pad to even H/W so upsample_to_input_res's output matches the encoder skip exactly,
        # then crop back to scale_factor times the original input size.
        _, _, h, w = x.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        feat = self.stem(x)
        embedding = self.severity_embed(feat)

        skip = feat
        for block in self.encoder:
            skip = block(skip, embedding)

        b = self.downsample(skip)
        for block in self.bottleneck:
            b = block(b, embedding)
        if self.global_context is not None:
            b = self.global_context(b)

        b = self.upsample_to_input_res(b)
        b = self.skip_fuse(torch.cat([b, skip], dim=1))

        for block in self.decoder:
            b = block(b, embedding)

        b = self.upsample_sr(b)
        out = torch.sigmoid(self.refine(b))

        if pad_h or pad_w:
            out = out[:, :, : h * self.scale_factor, : w * self.scale_factor]
        return out


def build_model_from_config(config: dict) -> RestorationNet:
    """Builds RestorationNet from a plain dict (a full experiment config with a 'model' key, or a
    model-only dict), so checkpoints stay self-describing."""
    model_cfg = config.get("model", config)
    return RestorationNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        num_encoder_blocks=model_cfg.get("num_encoder_blocks", 2),
        num_bottleneck_blocks=model_cfg.get("num_bottleneck_blocks", 6),
        num_decoder_blocks=model_cfg.get("num_decoder_blocks", 2),
        embed_dim=model_cfg.get("embed_dim", 64),
        scale_factor=model_cfg.get("scale_factor", 2),
        use_global_context=model_cfg.get("use_global_context", True),
        global_context_heads=model_cfg.get("global_context_heads", 4),
        input_clip_range=tuple(model_cfg.get("input_clip_range", (-0.5, 1.5))),
        severity_gated_residuals=model_cfg.get("severity_gated_residuals", False),
    )

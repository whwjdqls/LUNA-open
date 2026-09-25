"""Identity/query fusion and image-conditioned neural animator.

This is an independent joint-attention implementation, not an imported LHM
checkpoint architecture. Token-producing image encoders have separate APIs.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .avatar import CanonicalAvatar, Gaussians, PosedAvatar
from .geometry import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    sincos_to_quaternion,
)


@dataclass
class ModelConfig:
    width: int = 1024
    depth: int = 5
    heads: int = 16
    decoder_width: int = 512
    body_dim: int = 1536
    face_dim: int = 1024
    motion_dim: int = 1024
    num_parts: int = 24
    initial_scale: float = 0.008


class JointAttention(nn.Module):
    """Two streams with distinct projections and joint self/cross attention."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by attention heads")
        self.heads = heads
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.qkv = nn.ModuleList([nn.Linear(width, 3 * width) for _ in range(2)])
        self.out = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
                for _ in range(2)
            ]
        )

    def forward(self, query: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        streams = (query, context)
        projections = []
        for i, stream in enumerate(streams):
            b, n, c = stream.shape
            qkv = self.qkv[i](self.norms[i](stream)).reshape(b, n, 3, self.heads, c // self.heads)
            projections.append(qkv.permute(2, 0, 3, 1, 4))
        q, k, v = torch.cat(projections, dim=3).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(1, 2).flatten(2)
        parts = attention.split((query.shape[1], context.shape[1]), dim=1)
        output = []
        for i, stream in enumerate(streams):
            stream = stream + self.out[i](parts[i])
            output.append(stream + self.mlp[i](self.norms[i + 2](stream)))
        return tuple(output)


def mlp(width: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out),
    )


class IdentityEncoder(nn.Module):
    def __init__(self, anchors: Tensor, semantic_labels: Tensor, config: ModelConfig):
        super().__init__()
        self.register_buffer("anchors", anchors.clone())
        self.register_buffer("semantic_labels", semantic_labels.clone())
        self.queries = nn.Parameter(torch.randn(len(anchors), config.width) * 0.02)
        self.parts = nn.Embedding(config.num_parts, config.width)
        self.body_projection = nn.Linear(config.body_dim, config.width)
        self.face_projection = nn.Linear(config.face_dim, config.width)
        self.face_layers = nn.ModuleList(
            [nn.Linear(config.face_dim, config.face_dim) for _ in range(4)]
        )
        self.face_fusion = nn.Linear(4 * config.face_dim, config.face_dim)
        self.type_embedding = nn.Parameter(torch.randn(2, config.width) * 0.02)
        self.blocks = nn.ModuleList(
            [JointAttention(config.width, config.heads) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.width)
        self.decoder = mlp(config.width, config.decoder_width, 14)
        self.initial_scale = config.initial_scale
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, body_tokens: Tensor, face_tokens: Tensor) -> CanonicalAvatar:
        # [B,Nviews,Npatches,C] -> unordered multiview token set; within-image
        # spatial encoding comes from pretrained backbones. No input pose required.
        b = body_tokens.shape[0]
        if face_tokens.ndim == 5:
            # Frozen DINO features from four depths; learned per-depth 1x1
            # projections and concatenation fusion, as motivated by LHM.
            layers = [layer(face_tokens[:, :, i]) for i, layer in enumerate(self.face_layers)]
            face_tokens = self.face_fusion(torch.cat(layers, dim=-1))
        body = self.body_projection(body_tokens.flatten(1, 2)) + self.type_embedding[0]
        face = self.face_projection(face_tokens.flatten(1, 2)) + self.type_embedding[1]
        context = torch.cat((body, face), dim=1)
        query = (self.queries + self.parts(self.semantic_labels))[None].expand(b, -1, -1)
        for block in self.blocks:
            query, context = block(query, context)
        query = self.norm(query)
        raw = self.decoder(query).float()
        offset, q, scale, alpha, color = raw.split((3, 4, 3, 1, 3), dim=-1)
        identity_q = raw.new_tensor([1, 0, 0, 0])
        gaussians = Gaussians(
            self.anchors + offset,
            normalize_quaternion(q + identity_q),
            self.initial_scale * torch.exp(scale.clamp(-7, 7)),
            alpha.squeeze(-1).sigmoid(),
            color.sigmoid(),
        )
        return CanonicalAvatar(gaussians, query)


class NeuralAnimator(nn.Module):
    def __init__(
        self,
        num_queries: int,
        config: ModelConfig,
        translation_mean: Tensor,
        translation_std: Tensor,
    ):
        super().__init__()
        self.register_buffer("translation_mean", translation_mean.float())
        self.register_buffer("translation_std", translation_std.float().clamp_min(1e-4))
        self.identity_projection = mlp(config.width, config.decoder_width, config.width // 2)
        self.motion_queries = nn.Parameter(torch.randn(num_queries, config.width // 2) * 0.02)
        self.motion_projection = nn.Linear(config.motion_dim, config.width)
        self.blocks = nn.ModuleList(
            [JointAttention(config.width, config.heads) for _ in range(config.depth)]
        )
        self.local = mlp(config.width, config.decoder_width, 10)
        self.rotation_head = mlp(config.motion_dim, config.decoder_width, 6)
        self.translation_head = mlp(config.motion_dim, config.decoder_width, 3)
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        for head in (self.rotation_head, self.translation_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            self.rotation_head[-1].bias.copy_(torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))

    def forward(
        self, canonical: CanonicalAvatar, driving_tokens: Tensor, global_only: bool = False
    ) -> PosedAvatar:
        descriptor = driving_tokens.mean(1)
        rotation_raw = self.rotation_head(descriptor).float()
        translation_raw = self.translation_head(descriptor).float()
        pairs = F.normalize(rotation_raw.tanh().reshape(-1, 3, 2), dim=-1)
        q_global = sincos_to_quaternion(pairs)
        translation = self.translation_mean + self.translation_std * translation_raw.tanh()
        base = canonical.gaussians
        if global_only:
            delta_mu = torch.zeros_like(base.means)
            delta_q = torch.zeros_like(base.quaternions)
            delta_c = torch.zeros_like(base.colors)
        else:
            query = torch.cat(
                (
                    self.identity_projection(canonical.tokens),
                    self.motion_queries[None].expand(len(driving_tokens), -1, -1),
                ),
                -1,
            )
            context = self.motion_projection(driving_tokens)
            for block in self.blocks:
                query, context = block(query, context)
            delta_mu, delta_q, delta_c = self.local(query).float().split((3, 4, 3), -1)
        with torch.autocast(device_type=delta_mu.device.type, enabled=False):
            identity_q = delta_q.new_tensor([1, 0, 0, 0])
            rotation = quaternion_to_matrix(q_global)
            means = (base.means.float() + delta_mu) @ rotation.transpose(-1, -2) + translation[
                :, None
            ]
            quats = quaternion_multiply(
                q_global[:, None],
                quaternion_multiply(
                    normalize_quaternion(delta_q + identity_q), base.quaternions.float()
                ),
            )
        posed = Gaussians(
            means, normalize_quaternion(quats), base.scales, base.opacities, base.colors + delta_c
        )
        return PosedAvatar(posed, q_global, translation, pairs)

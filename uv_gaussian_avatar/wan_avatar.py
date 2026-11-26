"""One-pass RGB-to-UV model using a frozen Wan2.2 TI2V transformer."""

import importlib
import json
import sys
from types import ModuleType

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from luna_open.lora import LoRALinear

from .adaptive_atlas import AdaptiveAtlas
from .canonical_geometry import activate
from .native_islands import decoder_stages
from .representation import pose_gaussians
from .wan_rope import ExplicitWanRoPE, install_explicit_rope, native_wan_rope


def dense_attention(q, k, v, *, k_lens=None, window_size=(-1, -1)):
    """Equal-size unpadded sequences use PyTorch's differentiable SDPA."""
    if window_size != (-1, -1):
        raise ValueError('Expected global Wan attention')
    if k_lens is not None and not torch.all(k_lens == k.shape[1]):
        raise ValueError('The UV adapter only passes unpadded token grids')
    dtype = q.dtype
    result = F.scaled_dot_product_attention(*(t.transpose(1, 2).to(v.dtype) for t in (q, k, v)))
    return result.transpose(1, 2).contiguous().to(dtype)


def load_wan(source, checkpoints, dependencies=None):
    if dependencies is not None:
        sys.path.append(str(dependencies))
    # Load only the upstream transformer and VAE, independent of movie and
    # audio pipeline imports. Relative imports still resolve to upstream Wan.
    for name, path in (('_wan_uv', source / 'wan'), ('_wan_uv.modules', source / 'wan' / 'modules')):
        if name in sys.modules and sys.modules[name].__path__ != [str(path)]:
            raise ValueError('A different Wan source is already loaded')
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    module = importlib.import_module('_wan_uv.modules.model')
    vae_module = importlib.import_module('_wan_uv.modules.vae2_2')
    module.flash_attention = dense_attention
    install_explicit_rope(module)
    transformer = load_transformer_weights(module, checkpoints)
    if transformer.model_type != 'ti2v' or transformer.in_dim != 48:
        raise ValueError('Expected the Wan2.2 TI2V checkpoint with 48 VAE channels')
    transformer.requires_grad_(False).eval()
    vae = vae_module.Wan2_2_VAE(vae_pth=str(checkpoints / 'Wan2.2_VAE.pth'), dtype=torch.bfloat16)
    return transformer, vae, module.sinusoidal_embedding_1d


def load_transformer_weights(module, checkpoints):
    config = json.loads((checkpoints / 'config.json').read_text())
    with torch.device('meta'):
        backbone = module.WanModel(**{key: value for key, value in config.items() if not key.startswith('_')})
    # Wan keeps RoPE frequencies outside the state dict to retain complex precision.
    d = backbone.dim // backbone.num_heads
    band = 2 * (d // 6)
    backbone.freqs = torch.cat([module.rope_params(1024, width) for width in (d - 2 * band, band, band)],
                               dim=1).cuda()
    index = json.loads((checkpoints / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    mapping = index['weight_map']
    weights = {}
    for shard in sorted(set(mapping.values())):
        with safe_open(checkpoints / shard, framework='pt', device='cpu') as storage:
            for name in storage.keys():
                if mapping.get(name) != shard or name in weights:
                    raise ValueError(f'Wan weight index differs for {name}')
                full_precision = name.startswith(('time_embedding.', 'time_projection.')) or '.norm3.' in name
                dtype = torch.float32 if full_precision else torch.bfloat16
                weights[name] = storage.get_tensor(name).to(device='cuda', dtype=dtype)
        print(f'Loaded Wan checkpoint shard {shard}', flush=True)
    if weights.keys() != mapping.keys():
        raise ValueError('Wan checkpoint index and loaded tensor names differ')
    backbone.load_state_dict(weights, strict=True, assign=True)
    return backbone


def attach_lora(backbone, rank):
    targets = []
    for i, block in enumerate(backbone.blocks):
        for prefix, group, keys in (
            ('self_attn', block.self_attn, ('q', 'k', 'v', 'o')),
            ('cross_attn', block.cross_attn, ('q', 'k', 'v', 'o')),
            ('ffn', block.ffn, ('0', '2')),
        ):
            for key in keys:
                adapter = LoRALinear(getattr(group, key), rank, rank)
                # Keep the small optimizer parameters and their updates in FP32.
                adapter.lora_A = nn.Parameter(adapter.lora_A.detach().float())
                adapter.lora_B = nn.Parameter(adapter.lora_B.detach().float())
                setattr(group, key, adapter)
                targets.append(f'blocks.{i}.{prefix}.{key}')
    return targets


class UVHead(nn.Module):
    def __init__(self, hidden, width=128):
        super().__init__()
        self.projection = nn.Linear(hidden, width)
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.attributes = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 14, 1))
        nn.init.normal_(self.attributes[-1].weight, std=0.002)
        nn.init.zeros_(self.attributes[-1].bias)

    def project(self, queries, side):
        return self.projection(queries).transpose(1, 2).unflatten(2, (side, side))

    def decode(self, x, side, atlas_size):
        intermediate = min(atlas_size, side * 4)
        x = F.interpolate(x, size=(intermediate, intermediate), mode='bilinear', align_corners=False)
        x = self.refine(x)
        x = F.interpolate(x, size=(atlas_size, atlas_size), mode='bilinear', align_corners=False)
        return self.attributes(x)

    def forward(self, queries, side, atlas):
        dense = self.decode(self.project(queries, side), side, atlas.size)
        return atlas.gather(dense.permute(0, 2, 3, 1).float())


class IslandUpsample(nn.Module):
    """Refine at the current resolution, then learn a convolution after 2x upsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.GroupNorm(8, in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))
        self.upsample_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.skip(x) + self.residual(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.upsample_conv(x)


class AdaptiveUVHead(UVHead):
    """Warm-started coarse atlas with an independent progressive CNN on each island."""

    def __init__(self, hidden, num_islands):
        super().__init__(hidden)
        self.island_embedding = nn.Parameter(torch.zeros(num_islands, 128))
        self.island_decoder = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            IslandUpsample(256, 256), IslandUpsample(256, 128), IslandUpsample(128, 64),
            nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 14, 1),
        )
        nn.init.normal_(self.island_decoder[-1].weight, std=0.0001)
        nn.init.zeros_(self.island_decoder[-1].bias)

    def forward(self, queries, side, atlas):
        features = self.project(queries, side)
        coarse_atlas = self.decode(features, side, atlas.size)
        base = F.grid_sample(coarse_atlas.float(), atlas.base_grid, padding_mode='border',
                             align_corners=False).squeeze(2)
        residuals = []
        for index, island in enumerate(atlas.islands):
            x = F.grid_sample(features.float(), island.grid, padding_mode='border', align_corners=False)
            x = x + self.island_embedding[index][None, :, None, None]
            residuals.append(self.island_decoder(x * island.coarse_mask).flatten(2)[..., island.indices])
        return (base + torch.cat(residuals, dim=-1)).transpose(1, 2).contiguous()


class NativeUVHead(nn.Module):
    """Shared chart CNN and optional token-level canonical displacement head."""

    def __init__(self, hidden, patch_size, *, coarse_to_fine=False, coarse_range=.135):
        super().__init__()
        self.projection = nn.Linear(hidden, 256)
        blocks = []
        width = 256
        for out_channels in decoder_stages(patch_size):
            blocks.append(IslandUpsample(width, out_channels))
            width = out_channels
        self.cnn = nn.Sequential(*blocks, nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 14, 1))
        nn.init.normal_(self.cnn[-1].weight, std=0.002)
        nn.init.zeros_(self.cnn[-1].bias)

        self.coarse_range = coarse_range
        self.coarse_head = None
        if coarse_to_fine:
            self.coarse_head = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 3))
            nn.init.normal_(self.coarse_head[-1].weight, std=.0002)
            nn.init.zeros_(self.coarse_head[-1].bias)

    def forward(self, queries, layout):
        projected = self.projection(queries)
        parts = []
        for j, island in enumerate(layout.islands):
            start, end = layout.token_offsets[j:j + 2]
            patch_grid = island.scatter(projected[:, start:end])
            parts.append(island.gather(self.cnn(patch_grid)))
        raw = torch.cat(parts, dim=1).float()
        if self.coarse_head is None:
            return raw
        return raw, self.coarse_range * self.coarse_head(queries).float().tanh()


class WanUVAvatar(nn.Module):
    def __init__(self, backbone, vae, time_embedding, atlas, base_log_scale,
                 side=16, rank=8, query_timestep=500, *, image_side=None, native_layout=None,
                 canonical_3d_pos=False, geometry=None, coarse_to_fine=False, coarse_range=.135, fine_range=.015):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.atlas = atlas
        self.register_buffer('base_log_scale', base_log_scale)
        # Wan's VAE wrapper is not an nn.Module. Register its frozen encoder
        # module separately so state and device management can see it.
        self.vae = vae
        self.vae_model = vae.model.requires_grad_(False)
        self.time_embedding = time_embedding
        self.side = side
        self.image_side = side if image_side is None else image_side
        self.native_layout = native_layout
        self.geometry = geometry
        self.fine_range = fine_range
        if coarse_to_fine and (native_layout is None or geometry is None):
            raise ValueError('Coarse-to-fine decoding requires native mesh correspondence')
        if native_layout is None and (self.image_side != side or canonical_3d_pos):
            raise ValueError('Legacy Wan UV requires a shared image/query side and no canonical encoding')
        self.query_timestep = query_timestep
        self.targets = attach_lora(backbone, rank)
        num_queries = side * side if native_layout is None else native_layout.num_tokens
        self.queries = nn.Parameter(torch.randn(num_queries, 256) * 0.02)
        self.query_projection = nn.Sequential(nn.Linear(256, backbone.dim), nn.LayerNorm(backbone.dim))
        self.image_context = nn.Sequential(nn.LayerNorm(backbone.dim), nn.Linear(backbone.dim, backbone.dim))
        if native_layout is not None:
            self.token_island_embedding = nn.Embedding(len(native_layout.islands), 256)
            self.token_uv_projection = nn.Linear(4, 256, bias=False)
            nn.init.normal_(self.token_island_embedding.weight, std=0.02)
            nn.init.normal_(self.token_uv_projection.weight, std=0.02)
            coords, phases = native_wan_rope(backbone.freqs, self.image_side, native_layout.global_uv)
            self.register_buffer('token_rope_coordinates', coords, persistent=False)
            self.register_buffer('token_rope_phases', phases, persistent=False)
            self.head = NativeUVHead(backbone.dim, native_layout.patch_size,
                                     coarse_to_fine=coarse_to_fine, coarse_range=coarse_range)
            self.token_xyz_projection = nn.Linear(3, 256, bias=False) if canonical_3d_pos else None
            if self.token_xyz_projection is not None:
                nn.init.zeros_(self.token_xyz_projection.weight)
        elif isinstance(atlas, AdaptiveAtlas):
            self.head = AdaptiveUVHead(backbone.dim, len(atlas.islands))
        else:
            self.head = UVHead(backbone.dim)

    @torch.no_grad()
    def encode_image(self, image):
        """One white-background RGB image [3,H,W] in [0,1] to a one-frame latent."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError('Expected one RGB image in CHW order')
        h, w = image.shape[-2:]
        square = max(h, w)
        x, y = (square - w) // 2, (square - h) // 2
        image = F.pad(image, (x, square - w - x, y, square - h - y), value=1)
        size = self.image_side * 32  # Wan2.2 VAE stride 16 and DiT patch stride 2
        image = F.interpolate(image[None], size=(size, size), mode='bilinear', align_corners=False, antialias=True)[0]
        latent = self.vae.encode([(image * 2 - 1).unsqueeze(1)])[0]
        if latent.shape != (48, 1, self.image_side * 2, self.image_side * 2):
            raise RuntimeError(f'Unexpected Wan VAE image shape {tuple(latent.shape)}')
        return latent

    def forward(self, latent, *, return_aux=False):
        single = latent.ndim == 4
        if single:
            latent = latent.unsqueeze(0)
        backbone = self.backbone
        n = self.image_side**2
        batch = len(latent)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=latent.is_cuda):
            image = backbone.patch_embedding(latent).flatten(2).transpose(1, 2)
            if image.shape[1] != n:
                raise ValueError('Wan image token geometry differs from the configured image lattice')
            seed = self.queries
            if self.native_layout is not None:
                layout = self.native_layout
                uv = torch.cat((layout.global_uv, layout.local_uv), dim=-1) * 2 - 1
                seed = seed + self.token_island_embedding(layout.island_ids) + self.token_uv_projection(uv)
                if self.token_xyz_projection is not None:
                    seed = seed + self.token_xyz_projection(layout.normalized_xyz)
            queries = self.query_projection(seed)[None].expand(batch, -1, -1)
            x = torch.cat([image, queries], dim=1)
            context = self.image_context(image)
        length = x.shape[1]
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            timestep = torch.full((length,), self.query_timestep, device=x.device)
            timestep[:n] = 0
            e = backbone.time_embedding(self.time_embedding(backbone.freq_dim, timestep).float()[None])
            modulation = backbone.time_projection(e).unflatten(2, (6, backbone.dim))
        grids = (ExplicitWanRoPE(self.token_rope_phases) if self.native_layout is not None
                 else torch.tensor([[2, self.side, self.side]]).expand(batch, -1))
        lengths = torch.full((batch,), length)
        freqs = backbone.freqs.to(x.device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=latent.is_cuda):
            for block in backbone.blocks:
                x = block(x, modulation, lengths, grids, freqs, context, None)
            tokens = backbone.head.norm(x[:, n:])
            raw = (self.head(tokens, self.native_layout) if self.native_layout is not None
                   else self.head(tokens, self.side, self.atlas))
        token_offsets = None
        if isinstance(raw, tuple):
            raw, token_offsets = raw
        if single:
            raw = raw[0]
            if token_offsets is not None:
                token_offsets = token_offsets[0]
        fields, aux = activate(raw, self.base_log_scale, token_offsets=token_offsets,
                               geometry=self.geometry, fine_range=self.fine_range)
        return (fields, aux) if return_aux else fields

    def trainable_state(self):
        return {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def load_trainable_state(self, state, *, add_island_decoder=False):
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        allowed = {name for name in parameters if name.startswith('head.island_')} if add_island_decoder else set()
        if parameters.keys() - state.keys() != allowed or state.keys() - parameters.keys():
            raise ValueError('Checkpoint trainable parameter names differ')
        for name, value in state.items():
            p = parameters[name]
            if p.shape != value.shape:
                raise ValueError(f'Checkpoint shape differs for {name}')
            p.copy_(value)

    @torch.no_grad()
    def predict_image(self, image):
        """Return one canonical set of Gaussian attributes from a single RGB image."""
        return self(self.encode_image(image))


def animate_prediction(fields, posed_surface):
    return pose_gaussians(posed_surface, **fields)

"""Native UV patch indexing on the exact adaptive Gaussian atlas."""

import torch
from torch import nn
from torch.nn import functional as F


def decoder_stages(patch_size):
    if patch_size < 1 or patch_size & (patch_size - 1):
        raise ValueError('UV patch size must be a positive power of two')
    stages = patch_size.bit_length() - 1
    return [max(32, 256 >> i) for i in range(stages)]


def add_uv_arguments(parser):
    parser.add_argument('--uv-representation', choices=('legacy', 'native-islands'), default='legacy')
    parser.add_argument('--uv-patch-size', type=int, default=16)
    parser.add_argument('--image-side', type=int, help='Image token side length, 16 for 512px input')
    parser.add_argument('--canonical-3d-pos', action='store_true')
    parser.add_argument('--adaptive-decoder', action='store_true')
    parser.add_argument('--adaptive-body-scale', type=float,
                        help='Body sampling multiplier, defaults to 1.2 for native islands')


def resolve_uv_arguments(parser, args):
    native = args.uv_representation == 'native-islands'
    if args.image_side is None:
        args.image_side = 16 if native else args.query_side
    if min(args.image_side, args.query_side, args.atlas_size, args.rank) < 1:
        parser.error('Image, query, atlas and LoRA dimensions must be positive')
    if args.adaptive_body_scale is None:
        args.adaptive_body_scale = 1.2 if native else 2.0
    if native:
        try:
            decoder_stages(args.uv_patch_size)
        except ValueError as error:
            parser.error(str(error))
        args.adaptive_decoder = True
    elif args.canonical_3d_pos or args.uv_patch_size != 16 or args.image_side != args.query_side:
        parser.error('Independent image/UV settings require --uv-representation native-islands')
    if args.adaptive_body_scale != 2.0 and not args.adaptive_decoder:
        parser.error('--adaptive-body-scale requires an adaptive atlas')
    return args


class IslandPatches(nn.Module):
    """One rectangular patch lattice and its compact valid samples."""

    def __init__(self, mask, bounds, resolution, anchors, patch_size):
        super().__init__()
        if mask.ndim != 2 or mask.dtype != torch.bool or not mask.any():
            raise ValueError('Expected a nonempty two-dimensional island mask')
        self.bounds = tuple(bounds)
        self.resolution = resolution
        height, width = mask.shape
        self.shape = (height, width)
        self.dense_shape = (height + (-height) % patch_size, width + (-width) % patch_size)
        padded_height, padded_width = self.dense_shape
        rows, cols = padded_height // patch_size, padded_width // patch_size
        self.patch_shape = (rows, cols)

        mask = F.pad(mask.cpu(), (0, padded_width - width, 0, padded_height - height))
        y, x = mask.nonzero(as_tuple=True)
        dense_indices = y * padded_width + x
        patch_for_texel = (y // patch_size) * cols + (x // patch_size)
        patch_indices, dense_to_patch, counts = patch_for_texel.unique(
            sorted=True, return_inverse=True, return_counts=True,
        )
        self.register_buffer('patch_indices', patch_indices)
        self.register_buffer('dense_indices', dense_indices)
        self.register_buffer('dense_to_patch', dense_to_patch)
        self.register_buffer('patch_row_col', torch.stack((patch_indices // cols, patch_indices % cols), -1))
        self.register_buffer('valid_counts', counts)
        if anchors.shape != (len(dense_indices), 3):
            raise ValueError('Dense UV texels and Gaussian anchors are misaligned')

        # Use only covered texel centers so a partial patch is positioned on its chart.
        left, top, _, _ = bounds
        x, y = x.to(torch.float64) + 0.5, y.to(torch.float64) + 0.5
        values = torch.stack(((left + x) / resolution, 1 - (top + y) / resolution,
                              x / width, 1 - y / height), dim=1)
        values = torch.cat((values, anchors.detach().cpu().to(torch.float64)), dim=1)
        average = values.new_zeros((len(patch_indices), 7)).index_add_(0, dense_to_patch, values)
        uv, local, canonical = (average / counts[:, None]).float().split((2, 2, 3), dim=1)
        self.register_buffer('global_uv', uv.contiguous())
        self.register_buffer('local_uv', local.contiguous())
        self.register_buffer('canonical_xyz', canonical.contiguous())

    def scatter(self, tokens):
        """[B, n_valid, C] → [B, C, h, w], with empty positions set to zero."""
        batch, valid, channels = tokens.shape
        if valid != len(self.patch_indices):
            raise ValueError('Token count differs from the island layout')
        rows, cols = self.patch_shape
        dense = tokens.new_zeros((batch, channels, rows * cols))
        return dense.index_copy(2, self.patch_indices, tokens.transpose(1, 2)).unflatten(2, (rows, cols))

    def gather(self, fields):
        """[B, C, p*h, p*w] → [B, n_gaussians, C], in original anchor order."""
        if fields.shape[-2:] != self.dense_shape:
            raise ValueError('Decoder resolution differs from the island lattice')
        return fields.flatten(2).index_select(2, self.dense_indices).transpose(1, 2)


class NativeUVLayout(nn.Module):
    def __init__(self, atlas, neutral_vertices, patch_size):
        super().__init__()
        decoder_stages(patch_size)
        self.patch_size = patch_size
        self.islands = nn.ModuleList()
        self.token_offsets = [0]
        self.gaussian_offsets = [0]
        self.stats = []
        anchors = atlas.sample(neutral_vertices).detach().cpu()
        for index, (chart, stats) in enumerate(zip(atlas.islands, atlas.island_stats, strict=True)):
            start = self.gaussian_offsets[-1]
            end = start + stats['gaussians']
            island = IslandPatches(chart.valid_texels, stats['bounds'], stats['resolution'],
                                   anchors[start:end], patch_size)
            self.islands.append(island)
            tokens = len(island.patch_indices)
            slots = island.patch_shape[0] * island.patch_shape[1]
            self.token_offsets.append(self.token_offsets[-1] + tokens)
            self.gaussian_offsets.append(end)
            self.stats.append(dict(
                island_id=index, **stats, bounds_hw=list(island.shape), padded_hw=list(island.dense_shape),
                coarse_hw=list(island.patch_shape), valid_tokens=tokens, invalid_patches=slots - tokens,
                token_utilization=tokens / slots, token_start=self.token_offsets[-2],
                token_end=self.token_offsets[-1], gaussian_start=start, gaussian_end=end,
            ))
        if self.gaussian_offsets[-1] != len(anchors):
            raise ValueError('The patch layout did not cover the complete atlas')
        self.register_buffer('island_ids', torch.repeat_interleave(
            torch.arange(len(self.islands)), torch.diff(torch.tensor(self.token_offsets)),
        ))
        for name in ('global_uv', 'local_uv', 'canonical_xyz', 'patch_row_col'):
            self.register_buffer(name, torch.cat([getattr(island, name) for island in self.islands]))
        neutral_vertices = neutral_vertices.detach().cpu()
        low, high = neutral_vertices.amin(0), neutral_vertices.amax(0)
        half_extent = ((high - low).amax() * 0.5).clamp_min(1e-6)
        self.register_buffer('normalized_xyz', (self.canonical_xyz - (low + high) * 0.5) / half_extent)

    @property
    def num_tokens(self):
        return self.token_offsets[-1]

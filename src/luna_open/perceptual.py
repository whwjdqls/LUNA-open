"""LPIPS-Alex with explicitly prepared, verified backbone weights.

The filename's hash prefix comes from torchvision 0.23.0. The complete SHA256
was verified against the official download and pinned during setup.
"""

from pathlib import Path

from .provenance import file_sha256

ALEXNET_FILENAME = "alexnet-owt-7be5be79.pth"
ALEXNET_URL = "https://download.pytorch.org/models/" + ALEXNET_FILENAME
ALEXNET_BYTES = 244408911
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def verify_alexnet(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"LPIPS backbone missing at {path}; run scripts/download_lpips.py in an allocation"
        )
    if path.stat().st_size != ALEXNET_BYTES or file_sha256(path) != ALEXNET_SHA256:
        raise ValueError(f"LPIPS AlexNet backbone does not match the pinned weights: {path}")


def build_lpips(backbone: str = "alex", device: str = "cuda"):
    if backbone != "alex":
        raise ValueError("The initial benchmark protocol requires LPIPS-Alex")
    import lpips
    import torch

    verify_alexnet(Path(torch.hub.get_dir()) / "checkpoints" / ALEXNET_FILENAME)
    return lpips.LPIPS(net=backbone).to(device).eval().requires_grad_(False)

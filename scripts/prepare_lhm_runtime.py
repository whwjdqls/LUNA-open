"""Link acquired LHM assets into an isolated copy of its pinned source tree."""

import json
import os
import shutil
import socket
from pathlib import Path


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    work = Path("/scratch2/whwjdqls99/LUNA-open")
    runtime = work / "baselines/lhm-20260928/source"
    if not runtime.exists():
        shutil.copytree(work / "references/LHM", runtime)
    prior = work / "assets/lhm-priors"
    sapiens = (
        prior
        / "pretrained_models/sapiens/pretrained/checkpoints/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2"
    )
    links = {
        runtime / "pretrained_models": prior / "pretrained_models",
        runtime / "gfpgan": prior / "gfpgan",
        sapiens: work / "assets/sapiens_body/sapiens_1b_epoch_173_torchscript.pt2",
    }
    for destination, source in links.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"Existing path has another target: {destination}")
        else:
            destination.symlink_to(source, target_is_directory=source.is_dir())
    (runtime.parent / "asset-links.json").write_text(
        json.dumps({str(key): str(value) for key, value in links.items()}, indent=2) + "\n"
    )
    print(f"Prepared {runtime}", flush=True)


if __name__ == "__main__":
    main()

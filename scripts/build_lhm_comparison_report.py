"""Package measured LHM/local-identity results and all qualitative comparisons."""

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import socket
import zipfile
from pathlib import Path

import markdown2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def font(size):
    return ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", size)


def panel(images, labels, title, subtitle, tile=512):
    width = len(images) * (tile + 12) + 20
    result = Image.new("RGB", (width, tile + 104), "#eef1f5")
    draw = ImageDraw.Draw(result)
    draw.text((16, 10), title, fill="#17212b", font=font(20))
    for index, (image, label) in enumerate(zip(images, labels)):
        left = 16 + index * (tile + 12)
        draw.text((left, 40), label, fill="#17212b", font=font(18))
        result.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (left, 68))
    draw.text((16, tile + 76), subtitle, fill="#465465", font=font(14))
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--ours-metrics", type=Path, required=True)
    parser.add_argument("--lhm", type=Path, required=True)
    parser.add_argument("--lhm-metrics", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads((args.protocol / "protocol.json").read_text())
    ours = json.loads(args.ours_metrics.read_text())
    baseline = json.loads(args.lhm_metrics.read_text())
    fits = json.loads(args.fits.read_text())
    for result in (ours, baseline):
        if (
            result["method"]["manifest_sha256"] != protocol["manifest_sha256"]
            or result["split"] != "test"
        ):
            raise ValueError("Comparison protocol mismatch")
        if len(result["frames"]) != 41:
            raise ValueError("Require every official test frame")
    if not fits["all_passed"]:
        raise ValueError("Geometry conversion has unresolved failures")
    canonical_digest = hashlib.sha256(
        (args.ours / "canonical-camera.json").read_bytes()
    ).hexdigest()
    if baseline["method"].get("canonical_camera_sha256") != canonical_digest:
        raise ValueError("Canonical comparison cameras differ")
    update = ours["method"]["update"]
    args.output.mkdir(parents=True)
    for subdir in ("quantitative", "qualitative", "evidence"):
        (args.output / subdir).mkdir()
    for source, name in (
        (args.ours_metrics, "ours-metrics.json"),
        (args.lhm_metrics, "lhm-metrics.json"),
        (args.fits, "conversion-fits.json"),
        (args.protocol / "protocol.json", "protocol.json"),
        (args.ours / "method.json", "ours-method.json"),
        (args.lhm / "method.json", "lhm-method.json"),
        (args.lhm / "model-load.json", "lhm-model-load.json"),
        (args.lhm / "projection-check.json", "projection-check.json"),
        (args.ours / "canonical-camera.json", "canonical-camera.json"),
    ):
        shutil.copy2(source, args.output / "evidence" / name)
    baseline_work = args.lhm.parent
    for name in (
        "requirements-resolved.txt",
        "asset-links.json",
        "copied-torch-packages.json",
        "gfpgan-cache-audit.json",
    ):
        source = baseline_work / name
        if source.exists():
            shutil.copy2(source, args.output / "evidence" / name)
    receipt = Path(ours["method"]["checkpoint"]).with_suffix(".json")
    if receipt.exists():
        shutil.copy2(receipt, args.output / "evidence/identity-snapshot.json")
    scripts = args.output / "evidence/scripts"
    scripts.mkdir()
    for name in (
        "evaluate_lhm_neuman.py",
        "evaluate_renders.py",
        "fit_neuman_smplx.py",
        "export_identity_baseline.py",
        "freeze_identity_checkpoint.py",
        "run_lhm_neuman_yonsei.sh",
        "setup_lhm_yonsei.sh",
    ):
        shutil.copy2(Path(__file__).parent / name, scripts / name)
    keys = ["psnr", "l1", "foreground_l1", "background_l1", "ssim", "lpips", "mask_iou"]
    with (args.output / "quantitative/metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "scene", *keys])
        writer.writeheader()
        for label, data in (
            ("LHM-500M / 1 ref", baseline),
            (f"Ours / 4 refs / step {update}", ours),
        ):
            for scene, metrics in {**data["per_scene"], "macro": data["mean_over_scenes"]}.items():
                writer.writerow(
                    dict(method=label, scene=scene, **{key: metrics[key] for key in keys})
                )
    scenes = list(protocol["scenes"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, title) in zip(
        axes,
        (
            ("psnr", "PSNR (dB) ↑"),
            ("l1", "Full-crop L1 ↓"),
            ("foreground_l1", "Foreground L1 ↓"),
            ("lpips", "LPIPS-Alex ↓"),
        ),
    ):
        x = np.arange(len(scenes))
        ax.bar(
            x - 0.19,
            [baseline["per_scene"][scene][key] for scene in scenes],
            0.38,
            label="LHM-500M · 1 ref",
            color="#e0803c",
        )
        ax.bar(
            x + 0.19,
            [ours["per_scene"][scene][key] for scene in scenes],
            0.38,
            label=f"Ours · 4 refs · {update}",
            color="#2467ac",
        )
        ax.set_xticks(x, scenes, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / "quantitative" / f"per-scene.{suffix}", dpi=180)
    plt.close(fig)
    qualitative_links = []
    selected_panels = []
    for scene, info in protocol["scenes"].items():
        folder = args.output / "qualitative" / scene
        folder.mkdir()
        for label, source in (("lhm", args.lhm / scene), ("ours", args.ours / scene)):
            for kind in ("rgb", "alpha", "references"):
                shutil.copytree(source / kind, folder / label / kind)
        shutil.copytree(args.lhm / scene / "canonical", folder / "lhm/canonical")
        (folder / "ground-truth").mkdir()
        for name in info["targets"]:
            shutil.copy2(args.protocol / scene / "rgb" / name, folder / "ground-truth" / name)
        shutil.copy2(
            args.fits.parent / scene / info["targets"][0],
            folder / "body-conversion-overlay.png",
        )
        frames = []
        for name in info["targets"]:
            images = [
                Image.open(path).convert("RGB")
                for path in (
                    args.protocol / scene / "rgb" / name,
                    args.lhm / scene / "rgb" / name,
                    args.ours / scene / "rgb" / name,
                )
            ]
            comparison = panel(
                images,
                (
                    "Ground truth",
                    "LHM-500M · 1 reference",
                    f"Our identity · 4 references · {update}",
                ),
                f"{scene} | {name} | Official NeuMan test",
                "Both posed using supplied body annotations; LHM uses audited SMPL-X conversion",
            )
            comparison.save(folder / name)
            frames.append(comparison)
        frames[0].save(
            folder / "comparison.gif", save_all=True, append_images=frames[1:], duration=300, loop=0
        )
        middle = len(frames) // 2
        selected_panels.append(frames[middle])
        qualitative_links.append(
            f"### {scene}\n\n![{scene} comparison](qualitative/{scene}/comparison.gif)\n\n"
            f"[Selected frame](qualitative/{scene}/{info['targets'][middle]}) · "
            f"[All frames](qualitative/{scene}/index.html)\n"
        )
        page = "<!doctype html><meta charset='utf-8'><style>img{max-width:100%;margin-bottom:24px}body{font:16px sans-serif}</style>"
        page += f"<h1>{html.escape(scene)} — every official test frame</h1>"
        page += "".join(
            f"<h2>{html.escape(name)}</h2><img src='{html.escape(name)}'>"
            for name in info["targets"]
        )
        (folder / "index.html").write_text(page)
        refs = [
            Image.open(args.protocol / scene / "rgb" / name).convert("RGB")
            for name in info["references"]
        ]
        panel(
            refs,
            tuple(info["references"]),
            f"{scene} | Fixed training references",
            "LHM takes reference 1; our identity takes all four",
            tile=256,
        ).save(folder / "references.png")
        canonical = [
            Image.open(args.lhm / scene / "canonical/front.png").convert("RGB"),
            Image.open(args.ours / scene / "qualitative/canonical-front.png").convert("RGB"),
        ]
        panel(
            canonical,
            ("LHM canonical", "Our identity canonical"),
            f"{scene} | Canonical views",
            "Shared virtual front camera; canonical ground-truth images are unavailable",
        ).save(folder / "canonical.png")
        shutil.copy2(
            args.ours / scene / "qualitative/gt-canonical-lbs.gif",
            folder / "ours-gt-canonical-lbs.gif",
        )
        qualitative_links[-1] += (
            f"[Canonical comparison](qualitative/{scene}/canonical.png) · "
            f"[Our GT | canonical | LBS GIF](qualitative/{scene}/ours-gt-canonical-lbs.gif) · "
            f"[Reference inputs](qualitative/{scene}/references.png)\n"
        )
        qualitative_links[-1] += (
            f"[SMPL-to-SMPL-X geometry overlay](qualitative/{scene}/body-conversion-overlay.png)\n"
        )
    for index in range(0, len(selected_panels), 3):
        rows = [
            image.resize((1176, round(image.height * 1176 / image.width)))
            for image in selected_panels[index : index + 3]
        ]
        row_height = rows[0].height
        sheet = Image.new("RGB", (1176, row_height * len(rows)), "white")
        for row, image in enumerate(rows):
            sheet.paste(image, (0, row_height * row))
        sheet.save(args.output / "qualitative" / f"overview-{index // 3 + 1}.png")
    table = "| Model | Refs | PSNR ↑ | L1 ↓ | FG L1 ↓ | SSIM ↑ | LPIPS ↓ | IoU ↑ |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for name, count, data in (
        ("LHM-500M released", 1, baseline),
        (f"Our identity, step {update}", 4, ours),
    ):
        m = data["mean_over_scenes"]
        table += f"| {name} | {count} | {m['psnr']:.4f} | {m['l1']:.6f} | {m['foreground_l1']:.6f} | {m['ssim']:.6f} | {m['lpips']:.6f} | {m['mask_iou']:.6f} |\n"
    max_mean = max(row["mean_mm"] for row in fits["frames"])
    max_pixel = max(row["mean_pixels"] for row in fits["frames"])
    report = f"""# LHM versus our identity encoder on NeuMan

## Results

All 41 official test frames, six scenes, 512×512 white person crops. Each scene
is averaged independently, then the six scenes are weighted equally. L1 is RGB
absolute error on [0,1]. FG L1 measures error only over the ground-truth foreground;
predictions are not cleaned with the ground-truth mask.
NeuMan's supplied `segmentations/` masks define the person crops and white-background
reference/target images for both methods. Thus the protocol uses ground-truth masks
for preprocessing, including test cropping. Predicted images retain predicted alpha.

{table}

[Full metrics CSV](quantitative/metrics.csv) · [Our per-frame scores](evidence/ours-metrics.json) ·
[LHM per-frame scores](evidence/lhm-metrics.json)

![Per-scene errors](quantitative/per-scene.png)

## How to interpret the results

Our model has lower full-crop L1, foreground L1 and LPIPS in all six scenes under
this protocol. It was trained on these identities and uses four references;
released LHM uses one reference and has no NeuMan fine-tuning here. These results
do not establish a general ranking on unseen people or under equal training data.

The fixed middle-frame overviews also show a visual trade-off: LHM generally has
smoother surfaces, with more coherent face/clothing detail in examples such as
bike and citron. Our renders show visible splat artifacts, especially around
faces, arms and clothing edges, while matching the held-out silhouette and colors
more closely in these examples. Numerical error and visual smoothness should both
be presented. The overview frame is the middle test frame in each scene; every
test frame is included below for inspection.

![Bike, citron, jogging](qualitative/overview-1.png)
![Lab, parkinglot, seattle](qualitative/overview-2.png)

## What was compared

- Released LHM-500M, upstream commit `4f88aaeb3629249fbbddb4d0784a06962d9e1338`,
  with one fixed training reference per scene and no local fine-tuning.
- Our identity encoder, checkpoint **{update}**, four fixed training references,
  trained on the 344 training frames of these six identities.
- Both use annotation-driven body posing. Our model uses SMPL; LHM uses SMPL-X
  fitted to those SMPL meshes. Our neural animator is not involved.
- The differing reference counts and training exposure are part of this experiment.
  LHM's pretraining overlap with NeuMan is unknown. These are local measurements,
  not a reproduction of the LUNA paper's benchmark protocol or its unpublished MV-LHM.

## Geometry and implementation

All {len(fits["frames"])} conversion fits passed the predefined quality checks.
Worst per-frame mean correspondence error: **{max_mean:.3f} mm**; worst per-frame
mean reprojection difference: **{max_pixel:.3f} pixels**. Conversion fits body meshes;
it does not optimize held-out RGB. [Full conversion audit](evidence/conversion-fits.json).

LHM uses its released reconstruction, face restoration and skinning code. Its
provided gsplat renderer receives the full crop intrinsics and explicit 512×512
dimensions. [Projection check](evidence/projection-check.json) ·
[Model loading record](evidence/lhm-model-load.json).

Reference body crops follow our common person-crop protocol. Face crops use
the supplied NeuMan keypoints, followed by native LHM restoration and encoding.
This adapts the upstream input preprocessing, which normally uses a face detector.
The pinned LHM skinning function's stored-weight behavior is preserved.

[Our checkpoint/configuration](evidence/ours-method.json) ·
[LHM checkpoint/configuration](evidence/lhm-method.json) ·
[Exact frames and cameras](evidence/protocol.json).

Method sources: [LUNA, §3.3 and §4](https://arxiv.org/html/2606.31981v2),
[LHM paper](https://arxiv.org/html/2503.10625v1),
[pinned LHM source](https://github.com/aigc3d/LHM/tree/4f88aaeb3629249fbbddb4d0784a06962d9e1338),
and [released LHM-500M weights](https://huggingface.co/3DAIGC/LHM-500M/tree/7b9c7036404f9c21d8c4891ee4053e58c6ca2427).
LHM source is Apache-2.0; body assets, model weights and NeuMan have separate terms.

## Qualitative results

Each GIF shows **ground truth | LHM | our identity** on every held-out frame of
that scene. GIFs use 300 ms per frame for inspection; the sparse test frames do
not represent the original video timing. PNGs, references and canonical views
are in the same scene folders. Canonical views have no ground-truth counterpart.

""" + "\n".join(qualitative_links)
    (args.output / "REPORT.md").write_text(report)
    css = "body{max-width:1280px;margin:32px auto;padding:0 20px;font:17px/1.5 sans-serif;color:#17212b}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd3dc}"
    (args.output / "REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><style>"
        + css
        + "</style>"
        + markdown2.markdown(report, extras=["tables"])
    )
    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                out.write(path, Path(args.output.name) / path.relative_to(args.output))
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(
        json.dumps(
            dict(
                report=str(args.output / "REPORT.html"),
                archive=str(archive),
                bytes=archive.stat().st_size,
                sha256=digest,
                identity_update=update,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

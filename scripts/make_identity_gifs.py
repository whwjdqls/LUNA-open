"""Assemble ground-truth/canonical/LBS GIFs from verified identity render PNGs.

PIL-only display preparation, run on a Slurm compute node. Source renders are
checked against their recorded inventory and are never overwritten. The middle
panel is the fixed canonical front view; the final panel is the rendered identity
after the existing SMPL teacher applies the supplied fitted pose via LBS.
"""

import argparse
import hashlib
import html
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size)


def frame_panel(pictures, title, frame_label, duration_ms, tile):
    margin, gap, heading, footer = 16, 12, 78, 34
    canvas = Image.new("RGB", (3 * tile + 2 * gap + 2 * margin, heading + tile + footer), "#eef1f5")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), title, fill="#17212b", font=font(21))
    labels = ("Ground truth", "Canonical identity render", "Fitted SMPL pose (LBS)")
    for column, (label, picture) in enumerate(zip(labels, pictures, strict=True)):
        left = margin + column * (tile + gap)
        draw.text((left, 47), label, fill="#17212b", font=font(18))
        canvas.paste(picture.resize((tile, tile), Image.Resampling.LANCZOS), (left, heading))
    draw.text((margin, heading + tile + 8), frame_label, fill="#465465", font=font(14))
    timing = f"Display: {1000 / duration_ms:g} fps"
    draw.text((canvas.width - 160, heading + tile + 8), timing, fill="#465465", font=font(14))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=200)
    parser.add_argument("--tile", type=int, default=384)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Run image preparation on a Slurm compute node")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.duration_ms < 10 or args.duration_ms % 10 or args.tile < 320:
        raise ValueError("GIF duration must be a multiple of 10 ms; tile must be at least 320")
    started = time.perf_counter()
    source_report = json.loads((args.source / "report.json").read_text())
    inventory = source_report["artifact_inventory"]
    checkpoint = Path(source_report["checkpoint"])
    if digest(checkpoint) != source_report["checkpoint_sha256"]:
        raise ValueError("Latest identity checkpoint changed; refresh source visualizations")
    last_update = json.loads((checkpoint.parent / "train.jsonl").read_text().splitlines()[-1])[
        "update"
    ]
    if last_update != source_report["identity_update"]:
        raise ValueError("A newer identity update exists")
    checked = set()

    def open_source(relative):
        path = args.source / relative
        expected = inventory[relative]
        if relative not in checked:
            if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError(f"Source image changed: {relative}")
            checked.add(relative)
        with Image.open(path) as image:
            if image.size != (512, 512):
                raise ValueError(f"Unexpected raw render size: {relative}")
            return image.convert("RGB")

    args.output.mkdir(parents=True)
    for split in ("train", "test"):
        (args.output / split).mkdir()
    records, sections = {"train": {}, "test": {}}, []
    for scene in source_report["canonical"]:
        canonical = open_source(f"canonical/{scene}/front.png")
        section = f"<section><h2>{html.escape(scene)}</h2><div class=pair>"
        for split in ("train", "test"):
            rows = source_report["splits"][split][scene]["records"]
            panels = []
            for index, row in enumerate(rows):
                name = row["frame"]
                truth = open_source(f"{split}/{scene}/target-{name}")
                posed = open_source(f"{split}/{scene}/identity-{name}")
                tag = " | reference input" if row["reference_input"] else ""
                panels.append(
                    frame_panel(
                        (truth, canonical, posed),
                        f"{scene} | {split.upper()} | identity step {last_update:,}",
                        f"Frame {name} | {index + 1}/{len(rows)}{tag}",
                        args.duration_ms,
                        args.tile,
                    )
                )
            # One palette per sequence avoids independent frame color remapping.
            palette_sheet = Image.new("RGB", (128, 128 * len(panels)))
            for index, picture in enumerate(panels):
                palette_sheet.paste(picture.resize((128, 128)), (0, 128 * index))
            palette = palette_sheet.quantize(colors=256, dither=Image.Dither.NONE)
            frames = [
                picture.quantize(palette=palette, dither=Image.Dither.NONE) for picture in panels
            ]
            relative = f"{split}/{scene}.gif"
            target = args.output / relative
            frames[0].save(
                target,
                save_all=True,
                append_images=frames[1:],
                duration=args.duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
            with Image.open(target) as animation:
                if animation.n_frames != len(rows) or animation.info["loop"] != 0:
                    raise ValueError(f"Incomplete GIF: {relative}")
                for index in range(animation.n_frames):
                    animation.seek(index)
                    animation.load()
                    if animation.info["duration"] != args.duration_ms:
                        raise ValueError(f"GIF timing differs: {relative}")
                dimensions = list(animation.size)
            poster = args.output / split / f"{scene}-preview.jpg"
            panels[len(panels) // 2].save(poster, quality=95)
            records[split][scene] = {
                "path": relative,
                "frames": [row["frame"] for row in rows],
                "reference_targets": sum(row["reference_input"] for row in rows),
                "dimensions": dimensions,
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "preview_sha256": digest(poster),
                "verified_frame_count": len(rows),
                "verified_duration_ms": args.duration_ms,
            }
            section += f'<figure><figcaption>{split.title()} — {len(rows)} frames · <a href="{relative}">Open GIF</a></figcaption><a href="{relative}"><img loading="lazy" src="{relative}" alt="{html.escape(scene + " " + split)}: ground truth, canonical identity, fitted SMPL LBS"></a></figure>'
            print(
                json.dumps(
                    {
                        "event": "gif_complete",
                        "split": split,
                        "scene": scene,
                        "frames": len(rows),
                        "bytes": target.stat().st_size,
                    }
                ),
                flush=True,
            )
        sections.append(section + "</div></section>")
    counts = {
        split: sum(len(row["frames"]) for row in groups.values())
        for split, groups in records.items()
    }
    if counts != source_report["counts"]:
        raise ValueError("Incomplete split coverage")
    gallery = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Identity {last_update}: train/test GIFs</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#f4f6f9;color:#17212b;max-width:1440px;margin:32px auto;padding:0 20px}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}img{width:100%;height:auto}"
        "figcaption,p{line-height:1.7}section{margin:32px 0}a{color:#174ba0}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style>"
        f"<h1>Identity encoder {last_update:,}: train and test GIFs</h1>"
        "<p>Ground truth | canonical identity render | fitted SMPL pose (LBS).</p>"
        "<p>The canonical identity stays in a fixed front view. The third panel applies each supplied SMPL fit to that identity using the LBS teacher. "
        f"All {counts['train']} train and {counts['test']} test frames are included. Display is {1000 / args.duration_ms:g} fps, not original capture timing; no interpolated frames. "
        "Training frames used as reference inputs are labeled. Click a GIF for full size.</p>"
        + "".join(sections)
        + "</html>\n"
    )
    (args.output / "index.html").write_text(gallery)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "seconds": time.perf_counter() - started,
        "identity_update": last_update,
        "source": str(args.source),
        "source_report_sha256": digest(args.source / "report.json"),
        "checkpoint_sha256": source_report["checkpoint_sha256"],
        "latest_checkpoint_hash_rechecked": True,
        "script_sha256": digest(Path(__file__)),
        "source_images_verified": len(checked),
        "counts": counts,
        "gif_count": sum(len(groups) for groups in records.values()),
        "columns": ["ground truth", "canonical identity render", "fitted SMPL pose (LBS)"],
        "canonical": "Fixed front view of the same identity output for every frame in a scene",
        "posed": source_report["reconstruction"],
        "duration_ms": args.duration_ms,
        "playback": "Frame order preserved; display timing only; no interpolation; GIF uses one 256-color palette per sequence",
        "source_renders_modified": False,
        "gifs": records,
        "gallery_sha256": digest(args.output / "index.html"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "gif_count": report["gif_count"],
                "counts": counts,
                "source_images_verified": len(checked),
                "seconds": report["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

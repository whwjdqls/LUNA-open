"""Compare only completed evaluation artifacts with the same controlled protocol."""

import argparse
import json
import math
from pathlib import Path


def compare(paths):
    results = [json.loads(path.read_text()) for path in paths]
    methods = [result["provenance"]["method"] for result in results]
    if sorted(methods) != ["lhm", "lhmpp", "luna"]:
        raise ValueError("Require exactly one evaluation for each of LHM, LHM++ and LUNA")
    baseline = results[0]
    if not baseline["frames"]:
        raise ValueError("Evaluation has no targets")
    keys = set(baseline["mean_over_scenes"])
    if keys != {"psnr", "l1", "ssim", "mask_iou", "lpips"}:
        raise ValueError("Controlled evaluation metric set differs")
    for result in results:
        if result.get("split") not in {"val", "test"}:
            raise ValueError("Evaluation split is missing")
        if set(result["mean_over_scenes"]) != keys or any(
            not math.isfinite(value) for value in result["mean_over_scenes"].values()
        ):
            raise ValueError("Evaluation metrics are missing or nonfinite")
        for row in result["frames"]:
            if set(row["metrics"]) != keys or any(
                not math.isfinite(value) for value in row["metrics"].values()
            ):
                raise ValueError("Frame metrics are missing or nonfinite")
    for result in results[1:]:
        if result["split"] != baseline["split"]:
            raise ValueError("Evaluation splits differ")
        for key in (
            "contract_sha256",
            "corpus_sha256",
            "phase",
            "objective",
            "runner_source_sha256",
        ):
            if result["provenance"][key] != baseline["provenance"][key]:
                raise ValueError(f"Controlled comparison differs: {key}")

        def rows(data):
            return [(row["scene"], row["frame"]) for row in data["frames"]]

        if rows(result) != rows(baseline):
            raise ValueError("Evaluated targets or their order differ")
    return dict(
        protocol="shared_photometric; native architecture; LUNA supervision ablation",
        methods={
            method: result["mean_over_scenes"]
            for method, result in zip(methods, results, strict=True)
        },
        phase=baseline["provenance"]["phase"],
        split=baseline["split"],
        frames=len(baseline["frames"]),
        contract_sha256=baseline["provenance"]["contract_sha256"],
        corpus_sha256=baseline["provenance"]["corpus_sha256"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs=3, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.metrics), indent=2))


if __name__ == "__main__":
    main()

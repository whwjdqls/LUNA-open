"""Choose dataset semantics from the versioned training manifest."""

import json
from pathlib import Path


def dataset_from_manifest(
    root,
    manifest,
    split="train",
    size=512,
    random_references=False,
    require_smpl=False,
    verify=True,
):
    document = json.loads(Path(manifest).read_text())
    kind = document.get("dataset", "neuman")
    if kind == "dna_rendering":
        from .dna import DNADataset

        return DNADataset(root, manifest, split, size, random_references, require_smpl, verify)
    if kind != "neuman":
        raise ValueError(f"Unknown dataset kind: {kind}")
    from ..provenance import verify_sources
    from .neuman import NeuManDataset

    if verify:
        verify_sources(Path(root), document)
    return NeuManDataset(root, manifest, split, size, random_references=random_references)

#!/usr/bin/env python3
"""Download the author's pinned checkpoint; install the m2t2 extra first."""
from pathlib import Path
import hashlib
from huggingface_hub import hf_hub_download

from gr3_grasping.m2t2_backend import WEIGHT_REPO, WEIGHT_REVISION, WEIGHT_SHA256

destination = Path(__file__).resolve().parents[1] / "third_party/M2T2Models"
path = Path(hf_hub_download(WEIGHT_REPO, "m2t2.pth", revision=WEIGHT_REVISION, local_dir=destination))
if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHT_SHA256:
    raise RuntimeError("Downloaded M2T2 checkpoint failed SHA-256 verification")
print(path)

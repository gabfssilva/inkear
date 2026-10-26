"""Converts the OmniVoice step of one bucket to a Core AI model, for `coreai-aot` to compile.

The bucket comes from the `omnivoice.json` that `omnivoice-convert` writes, which the app reads
too: the function is named as the config names it. The step returns its tokens as Float16
(`Step(float_tokens=True)`), because the Neural Engine mislabels integer results; see coreai-aot.
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import TypedDict

import torch
from coreai_torch import TorchConverter, get_decomp_table
from omnivoice_coreml import convert
from omnivoice_coreml.modules import CODEBOOKS, Bucket, Step

BUILD = Path(__file__).resolve().parents[2] / "build"


def convert_step(bucket: Bucket, path: Path) -> None:
    """Writes the Float16 step of `bucket` to `path`, an .aimodel with one function."""
    model = Step(convert.load(), bucket, float_tokens=True).eval().half()
    examples = (
        torch.zeros(1, CODEBOOKS, bucket.sequence, dtype=torch.int32),
        torch.ones(1, bucket.sequence, dtype=torch.float16),
        torch.ones(1, bucket.sequence, dtype=torch.float16),
    )
    with torch.inference_mode():
        exported = torch.export.export(model, args=examples).run_decompositions(get_decomp_table())
    if path.exists():
        shutil.rmtree(path)
    TorchConverter().add_exported_program(
        exported,
        input_names=["input_ids", "audio_mask", "valid"],
        output_names=["tokens", "scores"],
        entrypoint_name=bucket.name,
    ).to_coreai().save_asset(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target", type=int, default=320, help="the bucket's target frames (the app runs 320)"
    )
    args = parser.parse_args()

    config: _Config = json.loads((convert.BUILD / "omnivoice.json").read_text())
    buckets = {b["target"]: Bucket(cond=b["cond"], target=b["target"]) for b in config["step_buckets"]}
    if args.target not in buckets:
        raise SystemExit(f"no bucket of {args.target} frames in omnivoice.json; there are {sorted(buckets)}")
    path = BUILD / "step-ane.aimodel"
    BUILD.mkdir(exist_ok=True)
    started = time.monotonic()
    convert_step(buckets[args.target], path)
    print(f"{path}: {buckets[args.target].name} in {time.monotonic() - started:.0f} s")


class _BucketEntry(TypedDict):
    function: str
    cond: int
    target: int


class _Config(TypedDict):
    step_buckets: list[_BucketEntry]

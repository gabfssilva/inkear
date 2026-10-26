import argparse
import json
import math
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download
from omnivoice import OmniVoice
from omnivoice.utils.duration import RuleDurationEstimator

from omnivoice_coreml import pipeline, voices
from omnivoice_coreml.modules import CODEBOOKS, GUIDANCE, MASK_ID, VOCAB, Bucket, Decoder, Step

MODEL = "k2-fsa/OmniVoice"
REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
ROOT = Path(__file__).resolve().parents[2] / "build"
BUILD = ROOT / "omnivoice"
SOURCES = ROOT / "sources"

# Target frames (25 per second) the step and the decoder are specialised for, 1.28 s apart up to
# 15.4 s: padding costs a full step position per frame twice (conditional and unconditional), so
# the steps are fine. Functions share the weights; each costs only its compile on first use.
TARGETS = tuple(range(32, 385, 32))
# Text tokens per target frame run ~0.16 (English) to ~0.23 (Portuguese); a request that does
# not fit its bucket moves to the next one.
TEXT_TOKENS_PER_FRAME = 0.3
QUALITY = {"low": 8, "normal": 12, "high": 16, "highest": 32}


def load() -> OmniVoice:
    path = snapshot_download(MODEL, revision=REVISION)
    return OmniVoice.from_pretrained(path, device_map="cpu", dtype=torch.float32).eval()


def buckets(prompt: int) -> tuple[Bucket, ...]:
    """One bucket per target size; `prompt` is what every request carries besides its own text."""
    return tuple(
        Bucket(cond=_round(prompt + target + math.ceil(TEXT_TOKENS_PER_FRAME * target) + 8), target=target)
        for target in TARGETS
    )


def convert_step(model: OmniVoice, bucket: Bucket, path: Path) -> None:
    length = bucket.sequence
    examples = (
        torch.zeros(1, CODEBOOKS, length, dtype=torch.int32),
        torch.ones(1, length),
        torch.ones(1, length),
    )
    step = Step(model, bucket).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(step, examples)
    _convert(
        traced,
        [
            ct.TensorType(name="input_ids", shape=(1, CODEBOOKS, length), dtype=np.int32),
            ct.TensorType(name="audio_mask", shape=(1, length), dtype=np.float16),
            ct.TensorType(name="valid", shape=(1, length), dtype=np.float16),
        ],
        [ct.TensorType(name="tokens", dtype=np.int32), ct.TensorType(name="scores", dtype=np.float16)],
        path,
    )


def convert_decoder(model: OmniVoice, frames: int, path: Path) -> None:
    examples = (torch.zeros(1, CODEBOOKS, frames, dtype=torch.int32), torch.ones(1, frames))
    # Built outside inference mode: its polyphase weights are new tensors that tracing must version.
    decoder = Decoder(model.audio_tokenizer, frames).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(decoder, examples)
    _convert(
        traced,
        [
            ct.TensorType(name="codes", shape=(1, CODEBOOKS, frames), dtype=np.int32),
            ct.TensorType(name="frame_mask", shape=(1, frames), dtype=np.float16),
        ],
        [ct.TensorType(name="audio", dtype=np.float32)],
        path,
    )


def package(path: Path, functions: Sequence[tuple[str, Callable[[Path], None]]]) -> None:
    """Convert each function into a scratch package, then merge them into one sharing weights."""
    descriptor = ct.utils.MultiFunctionDescriptor()
    with tempfile.TemporaryDirectory() as scratch:
        for index, (name, convert) in enumerate(functions, 1):
            started = time.perf_counter()
            single = Path(scratch) / f"{name}.mlpackage"
            convert(single)
            descriptor.add_function(str(single), src_function_name="main", target_function_name=name)
            print(
                f"  {path.stem} {index}/{len(functions)} {name}: {time.perf_counter() - started:.0f}s",
                flush=True,
            )
        descriptor.default_function_name = functions[0][0]
        shutil.rmtree(path, ignore_errors=True)
        ct.utils.save_multifunction(descriptor, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Convert OmniVoice to Core ML packages, with their assets, in {BUILD}"
    )
    parser.add_argument("--only", choices=("voices", "step", "decoder", "config"), action="append")
    args = parser.parse_args()
    parts = set(args.only or ("voices", "step", "decoder", "config"))
    model = load()
    BUILD.mkdir(parents=True, exist_ok=True)
    if "voices" in parts:
        for source in voices.VOICES:
            print(
                f"voice {source.name}: {voices.write(model, source, SOURCES, BUILD / 'voices')}", flush=True
            )
    voice_set = [pipeline.Voice.load(BUILD / "voices" / f"{v.name}.json") for v in voices.VOICES]
    plan = buckets(_prompt(model, voice_set))
    if "step" in parts:
        package(BUILD / "step.mlpackage", [(b.name, lambda p, b=b: convert_step(model, b, p)) for b in plan])
    if "decoder" in parts:
        package(
            BUILD / "decoder.mlpackage",
            [(f"f{f}", lambda p, f=f: convert_decoder(model, f, p)) for f in TARGETS],
        )
    if "config" in parts:
        write_config(model, plan, voice_set)
    print(f"done: {BUILD}", flush=True)


def write_config(model: OmniVoice, plan: Sequence[Bucket], voice_set: Sequence[pipeline.Voice]) -> None:
    snapshot = Path(snapshot_download(MODEL, revision=REVISION))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(snapshot / name, BUILD / name)
    tokenizer = model.text_tokenizer
    estimator = RuleDurationEstimator()
    config = {
        "model": f"https://huggingface.co/{MODEL}/tree/{REVISION}",
        "license": "OmniVoice weights CC-BY-NC-4.0; audio tokenizer: Boson Higgs Audio 2 Community License",
        "sample_rate": pipeline.SAMPLE_RATE,
        "samples_per_frame": pipeline.SAMPLES_PER_FRAME,
        "codebooks": CODEBOOKS,
        "audio_vocab": VOCAB,
        "mask_id": MASK_ID,
        "guidance_scale": GUIDANCE,
        "t_shift": pipeline.T_SHIFT,
        "layer_penalty": pipeline.LAYER_PENALTY,
        "position_temperature": pipeline.POSITION_TEMPERATURE,
        "quality_steps": QUALITY,
        "step_buckets": [{"function": b.name, "cond": b.cond, "target": b.target} for b in plan],
        "decoder_frames": list(TARGETS),
        "special_tokens": {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in (
                "<|denoise|>",
                "<|lang_start|>",
                "<|lang_end|>",
                "<|instruct_start|>",
                "<|instruct_end|>",
                "<|text_start|>",
                "<|text_end|>",
            )
        },
        "duration": {
            "weights": estimator.weights,
            "ranges": [[end, script] for end, script in estimator.ranges],
            "fallback_reference": {"text": "Nice to meet you.", "frames": 25},
            "low_threshold": 50,
            "boost_strength": 3,
        },
        "voices": [v.name for v in voice_set],
    }
    (BUILD / "omnivoice.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))


def _prompt(model: OmniVoice, voice_set: Sequence[pipeline.Voice]) -> int:
    """Positions the longest voice adds to every request: style tokens, transcript and codes."""
    style = len(
        model.text_tokenizer(
            f"{pipeline.DENOISE}<|lang_start|>pt<|lang_end|><|instruct_start|>None<|instruct_end|>"
        ).input_ids
    )
    return style + max(len(model.text_tokenizer(v.text).input_ids) + v.tokens.shape[1] for v in voice_set)


def _round(positions: int) -> int:
    return -(-positions // 32) * 32


def _convert(
    traced: torch.jit.ScriptModule, inputs: list[ct.TensorType], outputs: list[ct.TensorType], path: Path
) -> None:
    ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    ).save(str(path))

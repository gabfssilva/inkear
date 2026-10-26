"""Placement, latency, real-time factor, parity and samples of the packages in build/qwen3."""

import argparse
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile
import torch
from coremltools.models.compute_plan import MLComputePlan
from qwen_tts import Qwen3TTSModel

from qwen3_coreml import assets as assets_module
from qwen3_coreml import metrics
from qwen3_coreml.convert import BUILD, TEMPERATURE, TOP_K
from qwen3_coreml.pipeline import Assets, CoreMLStages, Speech, TorchStages, synthesize

SAMPLES = BUILD / "samples"
PACKAGES = (
    ("TextProjector", None),
    ("CodeEmbedder", None),
    ("CodeDecoder", "prefill"),
    ("CodeDecoder", "decode"),
    ("MultiCodeDecoder", None),
    ("SpeechDecoder", None),
)


@dataclass(frozen=True)
class Placement:
    """Share of a function's estimated cost on the Neural Engine, and the ops placed elsewhere."""

    neural_engine: float
    elsewhere: Counter[str]


def placement(package: Path, function: str | None) -> Placement:
    # MLComputePlan plans the default function only: plan a copy where `function` is the default.
    units = ct.ComputeUnit.CPU_AND_NE
    with tempfile.TemporaryDirectory() as scratch:
        path = package
        if function is not None:
            descriptor = ct.utils.MultiFunctionDescriptor(str(package))
            descriptor.default_function_name = function
            path = Path(scratch) / package.name
            ct.utils.save_multifunction(descriptor, str(path))
        model = ct.models.MLModel(str(path), compute_units=units, function_name=function)
        plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(), compute_units=units)
        name = function or "main"

        def operations() -> Iterator[tuple[str, str, float]]:
            for operation in plan.model_structure.program.functions[name].block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
                if usage is not None:
                    device = type(usage.preferred_compute_device).__name__
                    yield device, operation.operator_name, cost.weight if cost else 0.0

        placed = list(operations())
    total = sum(weight for _, _, weight in placed) or 1.0
    on_ane = sum(weight for device, _, weight in placed if device == "MLNeuralEngineComputeDevice")
    return Placement(
        neural_engine=on_ane / total,
        elsewhere=Counter(op for device, op, _ in placed if device != "MLNeuralEngineComputeDevice"),
    )


def reference(
    model: Qwen3TTSModel, assets: Assets, text: str, language: str, voice: str, seed: int
) -> np.ndarray:
    """The checkpoint's own generate API, in the non-streaming layout the packages implement."""
    torch.manual_seed(seed)
    name = assets_module.LANGUAGES[language]
    prompt = {
        "ref_code": [None],
        "ref_spk_embedding": [torch.from_numpy(assets.speaker(voice))],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }
    wavs, _ = model.generate_voice_clone(
        text=text, language=name, voice_clone_prompt=prompt, non_streaming_mode=True
    )
    return np.asarray(wavs[0], dtype=np.float32)


def score(label: str, audio: np.ndarray, text: str, language: str) -> None:
    hypothesis = metrics.transcribe(audio, assets_module.SAMPLE_RATE, language)
    print(
        f"  {label:22s} {len(audio) / assets_module.SAMPLE_RATE:5.2f} s  "
        f"wer {metrics.wer(hypothesis, text):.3f}  utmos {metrics.mos(audio, assets_module.SAMPLE_RATE):.2f}  "
        f"| {hypothesis}",
        flush=True,
    )


def talker_parity(
    assets: Assets,
    coreml: CoreMLStages,
    reference_stages: TorchStages,
    speech: Speech,
    text: str,
    language: str,
    voice: str,
) -> None:
    """The talker fed the same inputs, the prompt then the frames of `speech`."""
    coreml.reset()
    reference_stages.reset()
    prompt = assets.prompt(text, language, voice, reference_stages)
    pairs = [(coreml.prefill(prompt), reference_stages.prefill(prompt))]
    frames = speech.codes.T
    for index, frame in enumerate(frames[:-1]):
        embed = reference_stages.frame_embedding(frame) + assets.tts_pad
        pairs.append(
            (coreml.step(embed, len(prompt) + index), reference_stages.step(embed, len(prompt) + index))
        )
    errors, agree, divergence = [], 0, []
    for (logits, hidden), (expected_logits, expected_hidden) in pairs:
        errors.append(np.abs(hidden - expected_hidden).max() / np.abs(expected_hidden).max())
        agree += int(np.argmax(logits[:2048]) == np.argmax(expected_logits[:2048]))
        p = np.exp(expected_logits - expected_logits.max())
        q = np.exp(logits - logits.max())
        p, q = p / p.sum(), q / q.sum()
        divergence.append(float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum()))
    print(
        f"  CodeDecoder (teacher-forced, {len(pairs)} positions): hidden rel err median {np.median(errors):.2e} "
        f"max {np.max(errors):.2e}; logits top-1 agreement {agree / len(pairs):.1%}; "
        f"KL median {np.median(divergence):.2e} max {np.max(divergence):.2e}",
        flush=True,
    )
    # The predictor on the reference talker's hidden states, with shared noise.
    rng = np.random.default_rng(0)
    same, total, frames_same = 0, 0, 0
    for (_, (_, hidden)), frame in zip(pairs, frames, strict=True):
        first = reference_stages.code(int(frame[0]))
        noise = rng.gumbel(size=(15, 2048)).astype(np.float32)
        codes, _ = coreml.predict(hidden, first, noise)
        expected_codes, _ = reference_stages.predict(hidden, first, noise)
        same += int((codes == expected_codes).sum())
        total += len(codes)
        frames_same += int((codes == expected_codes).all())
    print(
        f"  MultiCodeDecoder (shared noise, {len(frames)} frames): codes equal {same / total:.1%}, "
        f"frames equal {frames_same / len(frames):.1%}",
        flush=True,
    )
    audio = coreml.speech(speech.codes)
    expected = reference_stages.speech(speech.codes)
    noise_power = np.mean((audio - expected) ** 2)
    print(
        f"  SpeechDecoder ({speech.codes.shape[1]} frames): SNR {10 * np.log10(np.mean(expected**2) / noise_power):.1f} dB, "
        f"LSD {metrics.log_spectral_distance(audio, expected):.2f} dB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Report on the packages in {BUILD}")
    parser.add_argument("--runs", type=int, default=5, help="utterances timed for the real-time factor")
    args = parser.parse_args()
    assets = Assets.read(BUILD)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    print("Neural Engine share of estimated cost (CPU_AND_NE)", flush=True)
    for name, function in PACKAGES:
        share = placement(BUILD / f"{name}.mlpackage", function)
        label = f"{name}.{function}" if function else name
        print(f"  {label:22s} {share.neural_engine:6.1%}  elsewhere: {dict(share.elsewhere)}", flush=True)

    coreml = CoreMLStages(BUILD, ct.ComputeUnit.CPU_AND_NE)
    text, voice = assets_module.TEXTS["pt"], assets.config.default_voices["pt"]
    print(f"real-time factor, pt sentence, voice {voice}, CPU_AND_NE, {args.runs} runs", flush=True)
    synthesize(assets, coreml, text, "pt", voice, np.random.default_rng(100))
    for seed in range(args.runs):
        coreml.reset()
        speech = synthesize(assets, coreml, text, "pt", voice, np.random.default_rng(seed))
        stages = "  ".join(
            f"{stage} {len(calls)}x{np.median(calls) * 1e3:.2f}={sum(calls) * 1e3:.0f}ms"
            for stage, calls in speech.timings.calls.items()
        )
        print(
            f"  seed {seed}: {speech.seconds:5.2f} s audio, rtf {speech.timings.total() / speech.seconds:.3f}  | {stages}",
            flush=True,
        )

    model = assets_module.load(assets.config.model)
    reference_stages = TorchStages(model, TEMPERATURE, TOP_K)
    print("samples (seed 0), Whisper large-v3-turbo", flush=True)
    for language, text in assets_module.TEXTS.items():
        voice = assets.config.default_voices[language]
        expected = reference(model, assets, text, language, voice, 0)
        coreml.reset()
        speech = synthesize(assets, coreml, text, language, voice, np.random.default_rng(0))
        for label, audio in (("reference", expected), ("coreml", speech.audio)):
            soundfile.write(SAMPLES / f"{language}_{label}.wav", audio, assets_module.SAMPLE_RATE)
            score(f"{language} {label}", audio, text, language)
        print(f"parity, {language}", flush=True)
        talker_parity(assets, coreml, reference_stages, speech, text, language, voice)
        reference_stages.reset()
        coreml.reset()
        rng_seed = 7
        mirrored = synthesize(
            assets, reference_stages, text, language, voice, np.random.default_rng(rng_seed)
        )
        ours = synthesize(assets, coreml, text, language, voice, np.random.default_rng(rng_seed))
        frames = min(mirrored.codes.shape[1], ours.codes.shape[1])
        equal = (mirrored.codes[:, :frames] == ours.codes[:, :frames]).all(axis=0)
        first_difference = int(np.argmin(equal)) if not equal.all() else frames
        print(
            f"  same draws, PyTorch modules vs Core ML: {mirrored.codes.shape[1]} vs {ours.codes.shape[1]} frames, "
            f"identical up to frame {first_difference}",
            flush=True,
        )
        score(f"{language} torch-loop", mirrored.audio, text, language)

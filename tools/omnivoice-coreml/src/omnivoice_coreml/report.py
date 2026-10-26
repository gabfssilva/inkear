import argparse
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from coremltools.models.compute_plan import MLComputePlan
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig, VoiceClonePrompt

from omnivoice_coreml import convert, metrics, pipeline
from omnivoice_coreml.modules import CODEBOOKS, MASK_ID
from omnivoice_coreml.pipeline import Conditioning, Engine, Layout, Voice

TEXTS = {
    "pt": (
        "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central um rapaz "
        "aqui do bairro, que eu conheço de vista e de chapéu."
    ),
    "en": "The quick brown fox jumps over the lazy dog, and then it runs back into the forest before anyone notices.",
}
SAMPLES = convert.ROOT / "samples"
NEURAL_ENGINE = "MLNeuralEngineComputeDevice"


@dataclass(frozen=True)
class Placement:
    """Share of a function's estimated cost per device, and the op types left off the Neural Engine."""

    cost: Mapping[str, float]
    elsewhere: Mapping[str, int]


def placement(package: Path, function: str) -> Placement:
    # MLComputePlan only plans the default function, so plan a copy where `function` is the default.
    descriptor = ct.utils.MultiFunctionDescriptor(str(package))
    descriptor.default_function_name = function
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / package.name
        ct.utils.save_multifunction(descriptor, str(copy))
        model = ct.models.MLModel(str(copy), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=function)
        plan = MLComputePlan.load_from_path(
            path=model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
    cost, elsewhere = Counter[str](), Counter[str]()
    for operation in plan.model_structure.program.functions[function].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            continue
        device = type(usage.preferred_compute_device).__name__
        estimate = plan.get_estimated_cost_for_mlprogram_operation(operation)
        cost[device] += estimate.weight if estimate else 0.0
        if device != NEURAL_ENGINE:
            elsewhere[operation.operator_name.split(".")[-1]] += 1
    total = sum(cost.values()) or 1.0
    return Placement(cost={d: w / total for d, w in cost.items()}, elsewhere=dict(elsewhere))


def median_seconds(run: Callable[[], object], runs: int) -> float:
    run()
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        run()
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def reference_scores(
    model: OmniVoice, conditioning: Conditioning, tokens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The PyTorch fp32 prediction for one step, batched as `OmniVoice._generate_iterative` does."""
    length, frames = conditioning.length, conditioning.target
    ids = conditioning.ids.copy()
    ids[:, length - frames :] = tokens
    batch = torch.full((2, CODEBOOKS, length), MASK_ID, dtype=torch.long)
    audio = torch.zeros((2, length), dtype=torch.bool)
    attention = torch.zeros((2, 1, length, length), dtype=torch.bool)
    batch[0], audio[0], attention[0] = (
        torch.from_numpy(ids),
        torch.from_numpy(conditioning.audio_mask > 0),
        True,
    )
    batch[1, :, :frames], audio[1, :frames] = torch.from_numpy(ids[:, -frames:]), True
    attention[1, :, :frames, :frames] = True
    pad = torch.arange(frames, length)
    attention[1, :, pad, pad] = True
    with torch.inference_mode():
        logits = model(input_ids=batch, audio_mask=audio, attention_mask=attention).logits.float()
        predicted, scores = model._predict_tokens_with_scoring(
            logits[0:1, :, length - frames :], logits[1:2, :, :frames], OmniVoiceGenerationConfig()
        )
    return predicted[0].numpy(), scores[0].numpy()


def reference_speech(
    model: OmniVoice, text: str, language: str, voice: Voice, steps: int, seed: int
) -> np.ndarray:
    torch.manual_seed(seed)
    prompt = VoiceClonePrompt(
        ref_audio_tokens=torch.from_numpy(voice.tokens).long(), ref_text=voice.text, ref_rms=voice.rms
    )
    return model.generate(text=text, language=language, voice_clone_prompt=prompt, num_step=steps)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Placement, latency, parity, RTF and samples of {convert.BUILD}"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--voice", default="f1")
    args = parser.parse_args()
    engine = Engine(convert.BUILD)
    model = convert.load()
    voice = engine.voice(args.voice)
    default = convert.QUALITY["normal"]
    conditioning = pipeline.condition(engine.tokenizer, TEXTS["pt"], "pt", voice)
    bucket = engine.bucket(conditioning)
    decoder_frames = next(f for f in engine.decoder_frames if f >= conditioning.target)
    print(
        f"pt sentence, voice {voice.name}: {conditioning.length} conditional positions, "
        f"{conditioning.target} frames ({conditioning.target / 25:.2f} s) -> step {bucket.name}, decoder f{decoder_frames}",
        flush=True,
    )

    print("neural engine share of estimated cost, CPU_AND_NE (ops left on the CPU)", flush=True)
    for package, function in (("step", bucket.name), ("decoder", f"f{decoder_frames}")):
        plan = placement(convert.BUILD / f"{package}.mlpackage", function)
        share = plan.cost.get(NEURAL_ENGINE, 0.0)
        print(f"  {package:8s} {function:12s} {share:6.1%}  {plan.elsewhere}", flush=True)

    step = engine.step(bucket)
    layout = Layout.place(conditioning, bucket)
    decode = engine.decoder(decoder_frames)
    tokens = np.zeros((CODEBOOKS, conditioning.target), np.int32)
    print(f"latency on CPU_AND_NE, median of {args.runs} (ms)", flush=True)
    print(
        f"  step    {bucket.name:12s} {1e3 * median_seconds(lambda: step(layout.inputs()), args.runs):7.1f}"
    )
    print(
        f"  decoder f{decoder_frames:<11d} "
        f"{1e3 * median_seconds(lambda: decode(tokens, conditioning.target), args.runs):7.1f}",
        flush=True,
    )

    print(f"parity with PyTorch fp32 ({default} steps, positions picked greedily)", flush=True)
    states: dict[int, np.ndarray] = {}
    pipeline.unmask(
        step, Layout.place(conditioning, bucket), default, None, lambda i, t: states.__setitem__(i, t.copy())
    )
    frames = conditioning.target
    for index in (0, default // 2, default - 1):
        current = states[index - 1] if index else np.full((CODEBOOKS, frames), MASK_ID, np.int32)
        probe = Layout.place(conditioning, bucket)
        probe.write(current)
        predicted, scores = step(probe.inputs())
        predicted, scores = predicted[0, :, -frames:], scores[0, :, -frames:].astype(np.float32)
        expected_tokens, expected_scores = reference_scores(model, conditioning, current)
        masked = current == MASK_ID
        print(
            f"  step {index:2d}: argmax agreement on masked codes {np.mean((predicted == expected_tokens)[masked]):.3f}, "
            f"score |err| mean {np.abs(scores - expected_scores)[masked].mean():.3f}",
            flush=True,
        )
    final = states[max(states)]
    with torch.inference_mode():
        expected = (
            model.audio_tokenizer.decode(torch.from_numpy(final).long()[None]).audio_values[0, 0].numpy()
        )
    decoded = engine.decode(final)
    error = np.linalg.norm(decoded - expected) / np.linalg.norm(expected)
    print(
        f"  decoder: relative error {error:.4f} (SNR {-20 * np.log10(error):.1f} dB), "
        f"LSD {metrics.log_spectral_distance(decoded, expected):.2f} dB",
        flush=True,
    )

    print("quality levels: rtf = wall time (front end + steps + decoder + postprocess) / audio", flush=True)
    for level, steps in convert.QUALITY.items():
        seconds = median_seconds(lambda n=steps: engine.synthesize(TEXTS["pt"], "pt", voice, steps=n), 3)
        print(
            f"  {level:8s} {steps:2d} steps: {seconds:.3f} s for {frames / 25:.2f} s, rtf {seconds * 25 / frames:.3f}"
        )

    print(f"samples at {default} steps in {SAMPLES} (Whisper large-v3-turbo)", flush=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for language, text in TEXTS.items():
        for backend in ("reference", "coreml"):
            audio = (
                reference_speech(model, text, language, voice, default, 0)
                if backend == "reference"
                else engine.synthesize(text, language, voice, steps=default).audio
            )
            path = SAMPLES / f"{voice.name}_{language}_{backend}.wav"
            sf.write(path, audio, pipeline.SAMPLE_RATE)
            transcript = metrics.transcribe(path, language)
            accent = f" pt-BR {metrics.brazilian(path):.2f}" if language == "pt" else ""
            print(
                f"  {path.name:24s} {len(audio) / pipeline.SAMPLE_RATE:5.2f} s  WER {metrics.wer(text, transcript):.3f} "
                f"UTMOS {metrics.utmos(path):.2f}{accent} | {transcript}",
                flush=True,
            )

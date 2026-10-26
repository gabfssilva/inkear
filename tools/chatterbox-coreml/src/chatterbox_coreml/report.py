"""chatterbox-report: Neural Engine placement, latency, parity against the PyTorch reference,
real-time factor per quality level, and the WAV samples with their Whisper transcripts."""

import argparse
import json
import re
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import coremltools as ct
import mlx_whisper
import numpy as np
import soundfile
import torch
from coremltools.models.compute_plan import MLComputePlan

from chatterbox_coreml import assets, pipeline, reference, s3gen, t3
from chatterbox_coreml.metrics import log_spectral_distance

SAMPLES = assets.ROOT / "build" / "samples"
TEXTS = {
    "pt": "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central um rapaz aqui do "
    "bairro, que eu conheço de vista e de chapéu.",
    "en": "The quick brown fox jumps over the lazy dog, and then it runs back into the forest before anyone notices.",
}
WHISPER = "mlx-community/whisper-large-v3-turbo"
QUALITY = ("low", "normal", "high", "highest")
# One function per package, plus T3's smallest and largest decode windows; the other buckets
# share their graphs.
PLANNED = (
    ("t3", "prefill_256"),
    ("t3", "decode_256"),
    ("t3", "decode_1024"),
    ("flow_encoder", "n512"),
    ("flow_step", "m1024"),
    ("f0", "t128"),
    ("hift", "t128"),
)


@dataclass(frozen=True)
class Placement:
    neural_engine: float
    other_ops: Mapping[str, int]


def placement(package: Path, function: str) -> Placement:
    """Share of the estimated cost MLComputePlan puts on the Neural Engine, and what runs elsewhere."""
    # MLComputePlan only plans the default function, so plan a copy where `function` is the default.
    descriptor = ct.utils.MultiFunctionDescriptor(str(package))
    descriptor.default_function_name = function
    with tempfile.TemporaryDirectory(dir=assets.CACHE) as scratch:
        copy = Path(scratch) / package.name
        ct.utils.save_multifunction(descriptor, str(copy))
        model = ct.models.MLModel(str(copy), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=function)
        plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)

    def operations() -> Iterator[tuple[str, str, float]]:
        for operation in plan.model_structure.program.functions[function].block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
            if usage is not None:
                yield operation.operator_name, type(usage.preferred_compute_device).__name__, cost.weight if cost else 0.0

    placed = list(operations())
    total = sum(weight for _, _, weight in placed) or 1.0
    ane = sum(weight for _, device, weight in placed if device == "MLNeuralEngineComputeDevice")
    other = Counter(f"{name}@{device[2:-13]}" for name, device, _ in placed if device != "MLNeuralEngineComputeDevice")
    return Placement(ane / total, dict(other))


def timed(run: Callable[[], object], runs: int) -> tuple[float, float]:
    """(median, min) seconds over `runs` calls after one warm-up."""
    run()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times), min(times)


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def word_error_rate(hypothesis: str, truth: str) -> float:
    h, r = words(hypothesis), words(truth)
    distance = list(range(len(h) + 1))
    for i, word in enumerate(r, 1):
        previous, distance[0] = distance[0], i
        for j, guess in enumerate(h, 1):
            previous, distance[j] = distance[j], min(distance[j] + 1, distance[j - 1] + 1, previous + (word != guess))
    return distance[-1] / len(r)


def transcribe(path: Path, language: str) -> str:
    return str(mlx_whisper.transcribe(str(path), path_or_hf_repo=WHISPER, language=language)["text"]).strip()


def reference_logits(text: str, tokens: np.ndarray) -> np.ndarray:
    """PyTorch T3 logits (2, steps, vocab), teacher-forced on `tokens`, one causal pass."""
    model = reference.model()
    bundle = pipeline.Bundle()
    ids = pipeline.text_ids(bundle, text, "pt")
    hp = model.t3.hp
    text_tokens = torch.tensor([ids, ids])
    with torch.inference_mode():
        embeds, _ = model.t3.prepare_input_embeds(
            t3_cond=model.conds.t3, text_tokens=text_tokens, speech_tokens=torch.full((2, 1), hp.start_speech_token)
        )
        bos = embeds[:, -1:]
        speech = model.t3.speech_emb(torch.from_numpy(tokens)[None]) + model.t3.speech_pos_emb.emb.weight[1 : len(tokens) + 1]
        sequence = torch.cat([embeds, bos, speech.expand(2, -1, -1)], dim=1)
        hidden = model.t3.tfmr(inputs_embeds=sequence).last_hidden_state
        start = embeds.shape[1]
        return model.t3.speech_head(hidden[:, start : start + len(tokens) + 1]).numpy()


def coreml_logits(models: pipeline.Models, text: str, tokens: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Core ML T3 logits teacher-forced on `tokens`, with prefill and per-token decode seconds."""
    bundle = models.bundle
    embeds = pipeline.prefill_embeds(bundle, pipeline.text_ids(bundle, text, "pt"))
    length = embeds.shape[1]
    bucket = next(n for n in bundle.config["t3"]["prefill_lengths"] if n >= length)
    padded = np.zeros((2, bucket, t3.HIDDEN), np.float16)
    padded[:, :length] = embeds
    cos, sin = t3.rope(np.arange(bucket))
    last = np.zeros((1, bucket), np.float16)
    last[0, length - 1] = 1
    prefill = models("t3", f"prefill_{bucket}")
    state = prefill.make_state()
    start = time.perf_counter()
    logits = [pipeline.predict(prefill, {"embeds": padded, "cos": cos.astype(np.float16), "sin": sin.astype(np.float16), "last": last}, state)]
    prefill_seconds = time.perf_counter() - start
    tables = bundle.tables
    start = time.perf_counter()
    for step, token in enumerate(tokens):
        position = length + step
        window = next(w for w in bundle.config["t3"]["decode_windows"] if w > position)
        embed = tables["speech_emb"][token].astype(np.float32) + tables["speech_pos"][step + 1].astype(np.float32)
        cos, sin = t3.rope(np.array([position]))
        mask = np.full((1, window), t3.MASKED, np.float16)
        mask[0, : position + 1] = 0
        inputs = {
            "embed": embed[None].astype(np.float16),
            "cos": cos.astype(np.float16),
            "sin": sin.astype(np.float16),
            "position": np.array([position], np.int32),
            "mask": mask,
        }
        logits.append(pipeline.predict(models("t3", f"decode_{window}"), inputs, state))
    decode_seconds = (time.perf_counter() - start) / len(tokens)
    return np.stack(logits, axis=1), prefill_seconds, decode_seconds


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def t3_parity(models: pipeline.Models, text: str, tokens: np.ndarray) -> dict[str, float]:
    expected = reference_logits(text, tokens)
    actual, prefill_seconds, decode_seconds = coreml_logits(models, text, tokens)
    guided = [x[0] + 0.5 * (x[0] - x[1]) for x in (expected, actual)]
    p, q = softmax(guided[0] / 0.8), softmax(guided[1] / 0.8)
    return {
        "steps": len(tokens) + 1,
        "logit_max_abs_error": float(np.abs(guided[0] - guided[1]).max()),
        "logit_range": float(np.abs(guided[0]).max()),
        "top1_agreement": float((guided[0].argmax(-1) == guided[1].argmax(-1)).mean()),
        "mean_kl": float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1).mean()),
        "prefill_ms": 1e3 * prefill_seconds,
        "decode_ms_per_token": 1e3 * decode_seconds,
    }


def reference_mel(tokens: np.ndarray, steps: int = 10, noise: np.ndarray | None = None) -> np.ndarray:
    model = reference.model()
    decoder = model.s3gen.flow.decoder
    original = decoder.rand_noise
    if noise is not None:
        decoder.rand_noise = torch.from_numpy(noise)[None]
    forward = decoder.forward

    def stepped(mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        return forward(mu, mask, steps, temperature, spks, cond)

    decoder.forward = stepped
    try:
        with torch.inference_mode():
            mel = model.s3gen.flow_inference(torch.from_numpy(tokens)[None], ref_dict=model.conds.gen, finalize=True)
    finally:
        decoder.forward, decoder.rand_noise = forward, original
    return mel[0].numpy()


def reference_source(mel: np.ndarray, seed: int) -> np.ndarray:
    hift = reference.model().s3gen.mel2wav
    with torch.inference_mode():
        f0 = hift.f0_predictor(torch.from_numpy(mel)[None])[0].numpy()
    linear = hift.m_source.l_linear
    weight, bias = linear.weight.detach()[0].numpy(), float(linear.bias.detach()[0])
    return s3gen.source(f0, weight, bias, np.random.default_rng(seed))


def reference_hift(mel: np.ndarray, excitation: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        mel_t, source = torch.from_numpy(mel)[None], torch.from_numpy(excitation)[None, None]
        return reference.model().s3gen.mel2wav.decode(x=mel_t, s=source)[0].numpy()


def reference_audio(mel: np.ndarray, tokens: int, seed: int) -> np.ndarray:
    """HiFT on `mel` with the source built as the pipeline builds it, then the pipeline's trim."""
    return pipeline.finish(reference_hift(mel, reference_source(mel, seed)), tokens)


def flow_parity(models: pipeline.Models, tokens: np.ndarray) -> dict[str, float]:
    expected = reference_mel(tokens)
    actual = pipeline.flow(models, list(tokens), 10)
    other_seed = reference_mel(tokens, noise=np.random.default_rng(1).standard_normal(models.bundle.noise.shape).astype(np.float32))
    audio = {name: reference_audio(mel, len(tokens), 0) for name, mel in (("ref", expected), ("coreml", actual), ("seed", other_seed))}
    return {
        "mel_max_abs_error": float(np.abs(expected - actual).max()),
        "mel_rms_error": float(np.sqrt(((expected - actual) ** 2).mean())),
        "mel_range": float(np.abs(expected).max()),
        "lsd_db_coreml_vs_reference": log_spectral_distance(audio["coreml"], audio["ref"]),
        "lsd_db_other_noise_vs_reference": log_spectral_distance(audio["seed"], audio["ref"]),
    }


def vocoder_parity(models: pipeline.Models, tokens: np.ndarray) -> dict[str, float]:
    """f0 alone; HiFT alone on an identical source; then the whole vocoder by spectrum.

    The whole chain cannot be compared sample by sample: the source integrates f0 into phase,
    so an fp16 f0 drifts the harmonics' phase by the end of the utterance.
    """
    mel = reference_mel(tokens)
    hift = reference.model().s3gen.mel2wav
    with torch.inference_mode():
        expected_f0 = hift.f0_predictor(torch.from_numpy(mel)[None])[0].numpy()
    actual_f0 = pipeline.f0(models, mel)
    excitation = reference_source(mel, 0)
    expected = reference_hift(mel, excitation)
    error = expected - pipeline.hift(models, mel, excitation)
    whole = pipeline.finish(pipeline.vocode(models, mel, np.random.default_rng(0)), len(tokens))
    return {
        "f0_max_relative_error": float(np.abs(actual_f0 - expected_f0).max() / np.abs(expected_f0).max()),
        "hift_snr_db": float(10 * np.log10((expected**2).sum() / (error**2).sum())),
        "hift_max_abs_error": float(np.abs(error).max()),
        "vocoder_lsd_db": log_spectral_distance(whole, pipeline.finish(expected, len(tokens))),
    }


def stage_latency(models: pipeline.Models, runs: int) -> dict[str, dict[str, float]]:
    """Median and best ms per call of the planned functions, on placeholder inputs."""
    rows = {}
    for package, function in PLANNED:
        model = models(package, function)
        state = model.make_state() if package == "t3" else None
        description = {f.name: f for f in model.get_spec().description.functions}[function]
        inputs = {}
        for feature in description.input:
            shape = tuple(feature.type.multiArrayType.shape)
            integer = feature.type.multiArrayType.dataType == ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32
            inputs[feature.name] = np.zeros(shape, np.int32) if integer else np.ones(shape, np.float16)
        median, best = timed(lambda model=model, inputs=inputs, state=state: pipeline.predict(model, inputs, state), runs)
        rows[f"{package}/{function}"] = {"median_ms": 1e3 * median, "min_ms": 1e3 * best}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Report on the packages in {assets.BUILD}")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-placement", action="store_true")
    args = parser.parse_args()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    bundle = pipeline.Bundle()
    models = pipeline.Models(bundle)
    summary: dict[str, object] = {}

    print("neural engine share of estimated cost (CPU_AND_NE)", flush=True)
    shares = {}
    for package, function in () if args.skip_placement else PLANNED:
        share = placement(bundle.root / f"{package}.mlpackage", function)
        shares[f"{package}/{function}"] = asdict(share)
        print(f"  {package}/{function}: {share.neural_engine:.1%} {share.other_ops or ''}", flush=True)
    summary["placement"] = shares

    print("stage latency on CPU_AND_NE (ms)", flush=True)
    summary["latency"] = stage_latency(models, 10)
    for name, row in summary["latency"].items():
        print(f"  {name}: median {row['median_ms']:.1f} min {row['min_ms']:.1f}", flush=True)

    print("reference samples (PyTorch, CPU fp32)", flush=True)
    references = {}
    for language, text in TEXTS.items():
        start = time.time()
        synthesis = reference.synthesize(text, language, args.seed)
        references[language] = synthesis
        soundfile.write(SAMPLES / f"{language}_reference.wav", synthesis.wav, reference.SAMPLE_RATE)
        print(f"  {language}: {len(synthesis.tokens)} tokens, {synthesis.seconds:.2f}s audio, {time.time() - start:.1f}s wall", flush=True)

    tokens = references["pt"].tokens
    print("parity on the pt reference tokens", flush=True)
    summary["parity"] = {
        "t3_teacher_forced": t3_parity(models, TEXTS["pt"], tokens),
        "flow_10_steps": flow_parity(models, tokens),
        "vocoder": vocoder_parity(models, tokens),
    }
    print(json.dumps(summary["parity"], indent=2), flush=True)

    print("Core ML samples and real-time factor (warm, CPU_AND_NE)", flush=True)
    steps = bundle.config["flow"]["steps"]
    rtf: dict[str, object] = {}
    for language, text in TEXTS.items():
        pipeline.synthesize(models, text, language, steps["normal"], args.seed)  # loads its functions
        for quality in QUALITY:
            runs = [pipeline.synthesize(models, text, language, steps[quality], args.seed) for _ in range(args.runs)]
            best = min(runs, key=lambda run: run.timings.total)
            row = {
                "steps": steps[quality],
                "audio_s": best.seconds,
                "tokens": len(best.tokens),
                "rtf_median": statistics.median(run.timings.total / run.seconds for run in runs),
                "rtf_best": best.timings.total / best.seconds,
                **{f"{k}_s": v for k, v in asdict(best.timings).items()},
            }
            rtf[f"{language}/{quality}"] = row
            print(f"  {language}/{quality}: {json.dumps(row)}", flush=True)
            if quality == "normal":
                soundfile.write(SAMPLES / f"{language}_coreml.wav", best.wav, reference.SAMPLE_RATE)
    summary["rtf"] = rtf

    print("whisper large-v3-turbo", flush=True)
    transcripts = {}
    for language, text in TEXTS.items():
        for kind in ("reference", "coreml"):
            heard = transcribe(SAMPLES / f"{language}_{kind}.wav", language)
            transcripts[f"{language}_{kind}"] = {"text": heard, "wer": word_error_rate(heard, text)}
            print(f"  {language}_{kind} (WER {transcripts[f'{language}_{kind}']['wer']:.2f}): {heard}", flush=True)
    summary["whisper"] = transcripts
    summary["size_mb"] = sum(f.stat().st_size for f in bundle.root.rglob("*") if f.is_file()) / 2**20
    (assets.ROOT / "build" / "report.json").write_text(json.dumps(summary, indent=2))
    print(f"size {summary['size_mb']:.0f} MB; summary in build/report.json", flush=True)

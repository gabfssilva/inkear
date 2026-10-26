"""Placement, parity, native timing and samples for the packages in build/moss-nano."""

import argparse
import json
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from coremltools.models.compute_plan import MLComputePlan
from huggingface_hub import snapshot_download

from moss_nano_coreml import convert, graphs, metrics, pipeline, reference, voices

ROOT = convert.BUILD.parent
TEXTS = {
    "pt": "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central um rapaz aqui do "
    "bairro, que eu conheço de vista e de chapéu.",
    "en": "The quick brown fox jumps over the lazy dog, and then it runs back into the forest before anyone notices.",
}
VOICE = {"pt": "rachel", "en": "ava"}
EXISTING_REPO = "aufklarer/MOSS-TTS-Nano-100M-CoreML"
EXISTING_REVISION = "fdef5dc64de5b68b126d6ff726ff32a02a783e42"
UNITS = {
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "ALL": ct.ComputeUnit.ALL,
}


def device_shares(plan: MLComputePlan, function: str) -> Mapping[str, object]:
    cost: Counter[str] = Counter()
    ops: Counter[str] = Counter()
    cpu_ops: Counter[str] = Counter()
    for operation in plan.model_structure.program.functions[function].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None or operation.operator_name == "const":
            continue
        device = (
            type(usage.preferred_compute_device).__name__.removeprefix("ML").removesuffix("ComputeDevice")
        )
        estimate = plan.get_estimated_cost_for_mlprogram_operation(operation)
        cost[device] += estimate.weight if estimate else 0.0
        ops[device] += 1
        if device == "CPU":
            cpu_ops[operation.operator_name.split(".")[-1]] += 1
    total = sum(cost.values()) or 1.0
    return {
        "cost": {d: round(w / total, 4) for d, w in cost.items()},
        "ops": dict(ops),
        "cpu_ops": dict(cpu_ops),
    }


def existing() -> Mapping[str, object]:
    """Step 1: the speech-swift graphs (MLState KV caches, flexible prefill length) under each compute unit."""
    root = Path(snapshot_download(EXISTING_REPO, revision=EXISTING_REVISION))
    result: dict[str, object] = {"repo": f"{EXISTING_REPO}@{EXISTING_REVISION}"}
    for graph in ("GlobalDecoder", "LocalDecoder", "CodecDecoder", "CodecEncoder"):
        rows: dict[str, object] = {}
        for label, units in UNITS.items():
            try:
                plan = MLComputePlan.load_from_path(path=str(root / f"{graph}.mlmodelc"), compute_units=units)
                rows[label] = device_shares(plan, "main")["cost"]
            except RuntimeError as error:
                rows[label] = f"load failed: {str(error).split('with error code:')[-1].strip(' .";}\n')}"
        result[graph] = rows
        print(f"  {graph}: {rows}", flush=True)
    return result


def placement() -> Mapping[str, object]:
    """Share of estimated cost per device for every function, under CPU_AND_NE."""
    result: dict[str, object] = {}
    for name in ("GlobalDecoder", "LocalDecoder", "CodecDecoder"):
        package = convert.BUILD / f"{name}.mlpackage"
        for function in (f.name for f in ct.models.utils.load_spec(str(package)).description.functions):
            # MLComputePlan plans the default function only, so plan a copy where it is the default.
            descriptor = ct.utils.MultiFunctionDescriptor(str(package))
            descriptor.default_function_name = function
            with tempfile.TemporaryDirectory() as scratch:
                copy = Path(scratch) / package.name
                ct.utils.save_multifunction(descriptor, str(copy))
                path = ct.models.utils.compile_model(
                    str(copy), destination_path=str(Path(scratch) / "m.mlmodelc")
                )
                plan = MLComputePlan.load_from_path(path=path, compute_units=ct.ComputeUnit.CPU_AND_NE)
                result[f"{name}.{function}"] = device_shares(plan, function)
            print(f"  {name}.{function}: {result[f'{name}.{function}']}", flush=True)
    return result


def parity(source: reference.Reference, pipe: pipeline.Pipeline) -> Mapping[str, object]:
    """Each stage against the PyTorch reference (fp32) on the pt sentence."""
    voice = voices.load(convert.BUILD / "voices", VOICE["pt"])
    rows = pipe.tokenizer.prompt(TEXTS["pt"], voice)
    codes = source.generate(TEXTS["pt"], voice, seed=0, sampling=reference.Sampling())
    sequence = np.concatenate([rows, np.concatenate([np.full((len(codes), 1), 9, np.int32), codes], 1)])
    model = source.model
    with torch.no_grad():
        embeds = model._build_inputs_embeds(torch.from_numpy(sequence).long()[None])
        expected = model.transformer(
            inputs_embeds=embeds, use_cache=False, return_dict=True
        ).last_hidden_state[0]
    expected = expected.numpy()
    timings = pipeline.Timings()
    hidden, length = pipe._prefill(rows, timings)
    result: dict[str, object] = {
        "prompt_rows_match_upstream": bool(np.array_equal(rows, source.prompt(TEXTS["pt"], voice))),
        "prefill_hidden": {
            "relative": metrics.relative(hidden[0], expected[length - 1]),
            "cosine": metrics.cosine(hidden[0], expected[length - 1]),
        },
    }
    steps = [pipe._step(frame, length + index, timings)[0] for index, frame in enumerate(codes[:60])]
    errors = [metrics.relative(h, expected[length + i]) for i, h in enumerate(steps)]
    cosines = [metrics.cosine(h, expected[length + i]) for i, h in enumerate(steps)]
    result["step_hidden_60_frames"] = {
        "relative_median": float(np.median(errors)),
        "relative_max": max(errors),
        "cosine_min": min(cosines),
    }

    sampler = graphs.LocalSampler(model).eval()
    settings = pipeline.Settings()
    rng = np.random.default_rng(0)
    seen = np.zeros((graphs.CODEBOOKS, graphs.CODEBOOK_SIZE), np.float32)
    first = every = ends = 0
    for index in range(60):
        state = expected[length - 1 + index][None].astype(np.float32)
        noise = -np.log(-np.log(np.clip(rng.random((17, graphs.CANDIDATES)), 1e-12, 1 - 1e-12))).astype(
            np.float32
        )
        with torch.no_grad():
            torch_end, torch_codes = sampler(
                *(
                    torch.from_numpy(a)
                    for a in (state, seen, noise[1:], noise[:1, :2], settings.row().astype(np.float32))
                )
            )
        out = pipe.local[16].predict(
            pipeline._pin(
                hidden=state.astype(np.float16),
                seen=seen.astype(np.float16),
                gumbel=noise[1:].astype(np.float16),
                text_gumbel=noise[:1, :2].astype(np.float16),
                settings=settings.row(),
            )
        )
        drawn = np.rint(out["codes"][0]).astype(int)
        first += int(drawn[0] == int(torch_codes[0, 0]))
        every += int(np.array_equal(drawn, torch_codes[0].numpy().astype(int)))
        ends += int((out["end"][0, 0] > 0.5) == (float(torch_end[0, 0]) > 0.5))
        seen[np.arange(graphs.CODEBOOKS), codes[index]] = 1
    greedy = pipeline.Settings(top_k=1)
    out = pipe.local[16].predict(
        pipeline._pin(
            hidden=expected[length - 1][None].astype(np.float16),
            seen=np.zeros((graphs.CODEBOOKS, graphs.CODEBOOK_SIZE), np.float16),
            gumbel=np.zeros((graphs.CODEBOOKS, graphs.CANDIDATES), np.float16),
            text_gumbel=np.zeros((1, 2), np.float16),
            settings=greedy.row(),
        )
    )
    with torch.no_grad():
        _, greedy_codes = sampler(
            torch.from_numpy(expected[length - 1][None]),
            torch.zeros(16, 1024),
            torch.zeros(16, 32),
            torch.zeros(1, 2),
            torch.from_numpy(greedy.row().astype(np.float32)),
        )
    result["local_same_noise_60_frames"] = {
        "first_codebook_agreement": first / 60,
        "whole_frame_agreement": every / 60,
        "end_flag_agreement": ends / 60,
        "greedy_frame_equal": bool(
            np.array_equal(np.rint(out["codes"][0]).astype(int), greedy_codes[0].numpy().astype(int))
        ),
    }

    long = np.concatenate([voices.load(convert.BUILD / "voices", name) for name in ("rachel", "thiago")])
    stream = pipeline._CodecStream(pipe, timings)
    for frame in long:
        stream.push(frame)
    result["codec_206_frames_snr_db"] = metrics.snr(stream.finish(), source.decode(long))
    print(f"  {result}", flush=True)
    return result


def native(
    compiled: Path,
    runner: Path,
    tokenizer: pipeline.Tokenizer,
    depth: int,
    language: str,
    seed: int,
    runs: int,
    output: Path,
) -> Mapping[str, object]:
    """One synthesis in the Swift runner (Core ML from Swift, CPU_AND_NE), timed over `runs` after a warm-up."""
    rows = tokenizer.prompt(TEXTS[language], voices.load(convert.BUILD / "voices", VOICE[language]))
    request = {
        "rows": rows.tolist(),
        "depth": depth,
        "settings": pipeline.Settings(depth=depth).row().ravel().astype(float).tolist(),
        "seed": seed,
        "max_frames": pipeline.Settings().max_frames,
    }
    request_path = output.with_suffix(".request.json")
    request_path.write_text(json.dumps(request))
    subprocess.run(
        [str(runner), str(compiled), str(request_path), str(output), str(runs)],
        check=True,
        capture_output=True,
    )
    request_path.unlink()
    result = json.loads(output.with_suffix(".json").read_text())
    audio = np.fromfile(output.with_suffix(".f32"), dtype=np.float32).reshape(2, -1)
    output.with_suffix(".f32").unlink()
    output.with_suffix(".json").unlink()
    sf.write(output.with_suffix(".wav"), audio.T, reference.SAMPLE_RATE, subtype="PCM_16")
    result.pop("codes")
    return result | {"audio": audio}


def build_native() -> tuple[Path, Path]:
    compiled = ROOT / "compiled"
    compiled.mkdir(exist_ok=True)
    for name in ("GlobalDecoder", "LocalDecoder", "CodecDecoder"):
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(convert.BUILD / f"{name}.mlpackage"), str(compiled)],
            check=True,
            capture_output=True,
        )
    (compiled / "config.json").write_text((convert.BUILD / "config.json").read_text())
    runner = ROOT / "bin" / "MossNanoRun"
    runner.parent.mkdir(exist_ok=True)
    source = Path(__file__).resolve().parents[2] / "swift" / "MossNanoRun.swift"
    subprocess.run(["swiftc", "-O", str(source), "-o", str(runner)], check=True)
    return compiled, runner


def judge(audio: np.ndarray, language: str) -> Mapping[str, object]:
    text = metrics.transcribe(audio, reference.SAMPLE_RATE, language)
    judged: dict[str, object] = {
        "audio_seconds": audio.shape[1] / reference.SAMPLE_RATE,
        "whisper": text,
        "wer": metrics.wer(text, TEXTS[language]),
    }
    if language == "pt":
        judged["p_brazilian"] = metrics.brazilian(audio, reference.SAMPLE_RATE)
    return judged


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Report on the packages in {convert.BUILD}")
    parser.add_argument("--seeds", type=int, default=6, help="seeds per language for the WER statistics")
    parser.add_argument("--quality-seeds", type=int, default=16, help="seeds per language and quality level")
    parser.add_argument("--runs", type=int, default=3, help="timed native runs after a warm-up")
    args = parser.parse_args()
    report: dict[str, object] = {}
    samples = ROOT / "samples"
    samples.mkdir(exist_ok=True)

    print("existing speech-swift graphs (aufklarer)", flush=True)
    report["existing"] = existing()
    print("placement, CPU_AND_NE", flush=True)
    compiled, runner = build_native()
    report["placement"] = placement()

    source = reference.load()
    pipe = pipeline.Pipeline(convert.BUILD, ct.ComputeUnit.CPU_AND_NE)
    print("parity", flush=True)
    report["parity"] = parity(source, pipe)

    print("native timing (Swift, CPU_AND_NE)", flush=True)
    timing: dict[str, object] = {}
    for depth in convert.CODEBOOK_DEPTHS:
        for language in TEXTS:
            out = native(
                compiled,
                runner,
                pipe.tokenizer,
                depth,
                language,
                0,
                args.runs,
                samples / f"{language}-swift-nq{depth}",
            )
            audio = out.pop("audio")
            assert isinstance(audio, np.ndarray)
            timing[f"{language}-nq{depth}"] = out | judge(audio, language)
            print(f"  {language} nq{depth}: {timing[f'{language}-nq{depth}']}", flush=True)
    report["native"] = timing

    print("samples: PyTorch and Core ML, one take per seed", flush=True)
    sampled: dict[str, object] = {}
    voice_wav, voice_rate = sf.read(
        convert.BUILD / "voices" / f"{VOICE['pt']}.wav", dtype="float32", always_2d=True
    )
    sampled["pt-voice-p_brazilian"] = metrics.brazilian(voice_wav.T, voice_rate)
    for language, text in TEXTS.items():
        voice = voices.load(convert.BUILD / "voices", VOICE[language])
        for seed in range(args.seeds):
            torch_audio = source.decode(
                source.generate(text, voice, seed=seed, sampling=reference.Sampling())
            )
            coreml = pipe.synthesize(text, voice, pipeline.Settings(), seed=seed)
            for label, audio in (("pytorch", torch_audio), ("coreml", coreml.audio)):
                if seed == 0:
                    sf.write(
                        samples / f"{language}-{label}.wav", audio.T, reference.SAMPLE_RATE, subtype="PCM_16"
                    )
                sampled[f"{language}-{label}-seed{seed}"] = judge(audio, language)
                print(
                    f"  {language} {label} seed {seed}: {sampled[f'{language}-{label}-seed{seed}']}",
                    flush=True,
                )
    report["samples"] = sampled

    print("quality levels, Core ML", flush=True)
    levels: dict[str, object] = {}
    for name, quality in pipeline.QUALITY.items():
        for language, text in TEXTS.items():
            voice = voices.load(convert.BUILD / "voices", VOICE[language])
            runs = []
            for seed in range(100, 100 + args.quality_seeds):
                takes = pipe.speak(text, voice, quality, seed)
                if name == "normal" and not runs:
                    sf.write(
                        samples / f"{language}-coreml-normal.wav",
                        takes[-1].audio.T,
                        reference.SAMPLE_RATE,
                        subtype="PCM_16",
                    )
                runs.append(
                    judge(takes[-1].audio, language)
                    | {
                        "attempts": len(takes),
                        "frames_generated": sum(len(take.codes) for take in takes),
                        "frames_kept": len(takes[-1].codes),
                    }
                )
            levels[f"{name}-{language}"] = {
                "wer_mean": float(np.mean([run["wer"] for run in runs])),
                "attempts_mean": float(np.mean([run["attempts"] for run in runs])),
                "generated_per_kept_frame": sum(run["frames_generated"] for run in runs)
                / sum(run["frames_kept"] for run in runs),
                "runs": runs,
            }
            print(
                f"  {name} {language}: "
                + str({k: v for k, v in levels[f"{name}-{language}"].items() if k != "runs"}),
                flush=True,
            )
    report["quality"] = levels
    (ROOT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {ROOT / 'report.json'}", flush=True)

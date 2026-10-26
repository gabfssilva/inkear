"""chatterbox-convert: every Core ML package and asset the Swift pipeline needs, in build/chatterbox."""

import argparse
import json
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.optimize import coreml as cto
from torch import nn

from chatterbox_coreml import assets, reference, s3gen, t3

PREFILL_LENGTHS = (128, 256, 384)
DECODE_WINDOWS = (256, 384, 512, 768, 1024)
FLOW_TOKENS = (384, 448, 512, 640, 768)
VOCODER_FRAMES = 128
VOCODER_CONTEXT = 16
STAGES = ("t3", "flow", "vocoder", "assets")

F16 = np.float16
_KV = ct.TensorType(shape=(t3.LAYERS, t3.ROWS * t3.HEADS, t3.CONTEXT, t3.HEAD_DIM), dtype=F16)


def _tensor(name: str, shape: Sequence[int], dtype: type[np.generic] = F16) -> ct.TensorType:
    return ct.TensorType(name=name, shape=tuple(shape), dtype=dtype)


def _convert(
    module: nn.Module,
    inputs: Sequence[ct.TensorType],
    output: str,
    states: Sequence[ct.StateType] = (),
) -> ct.models.MLModel:
    examples = tuple(
        torch.zeros(spec.shape.shape, dtype=torch.int32 if spec.dtype == ct.converters.mil.mil.types.int32 else torch.float32)
        for spec in inputs
    )
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), examples)
    return ct.convert(
        traced,
        inputs=list(inputs),
        outputs=[ct.TensorType(name=output, dtype=F16)],
        states=list(states) or None,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )


def _package(name: str, functions: Mapping[str, Callable[[], ct.models.MLModel]]) -> Path:
    """Convert each function in turn and save them as one package that shares identical weights."""
    descriptor = ct.utils.MultiFunctionDescriptor()
    path = assets.BUILD / f"{name}.mlpackage"
    with tempfile.TemporaryDirectory(dir=assets.CACHE) as scratch:
        for index, (function, build) in enumerate(functions.items(), 1):
            start = time.time()
            single = Path(scratch) / f"{function}.mlpackage"
            build().save(str(single))
            descriptor.add_function(str(single), src_function_name="main", target_function_name=function)
            print(f"  {name}: {function} ({index}/{len(functions)}) {time.time() - start:.0f}s", flush=True)
        descriptor.default_function_name = next(iter(functions))
        shutil.rmtree(path, ignore_errors=True)
        ct.utils.save_multifunction(descriptor, str(path))
    return path


def _palettized(model: ct.models.MLModel) -> ct.models.MLModel:
    """4-bit weights, one lookup table per 16 output channels.

    On the Neural Engine this is the fastest weight format for T3's memory-bound decode step
    (MLP-only step: fp16 5.8 ms, int8 4.5 ms, 6-bit 3.8 ms, 4-bit 3.3 ms, 2-bit 3.7 ms). k-means
    is deterministic, so every function gets identical tables and the package stores them once.
    """
    config = cto.OpPalettizerConfig(nbits=4, granularity="per_grouped_channel", group_size=16)
    return cto.palettize_weights(model, cto.OptimizationConfig(global_config=config))


def convert_t3() -> Path:
    model = reference.model().t3
    states = [ct.StateType(wrapped_type=_KV, name="k_cache"), ct.StateType(wrapped_type=_KV, name="v_cache")]

    def prefill(length: int) -> ct.models.MLModel:
        inputs = [
            _tensor("embeds", (t3.ROWS, length, t3.HIDDEN)),
            _tensor("cos", (length, t3.HEAD_DIM)),
            _tensor("sin", (length, t3.HEAD_DIM)),
            _tensor("last", (1, length)),
        ]
        return _palettized(_convert(t3.Prefill(model, length), inputs, "logits", states))

    def decode(window: int) -> ct.models.MLModel:
        inputs = [
            _tensor("embed", (1, t3.HIDDEN)),
            _tensor("cos", (1, t3.HEAD_DIM)),
            _tensor("sin", (1, t3.HEAD_DIM)),
            _tensor("position", (1,), np.int32),
            _tensor("mask", (1, window)),
        ]
        return _palettized(_convert(t3.Decode(model, window), inputs, "logits", states))

    return _package(
        "t3",
        {f"prefill_{n}": (lambda n=n: prefill(n)) for n in PREFILL_LENGTHS}
        | {f"decode_{w}": (lambda w=w: decode(w)) for w in DECODE_WINDOWS},
    )


def convert_flow() -> tuple[Path, Path]:
    flow = reference.model().s3gen.flow

    def encoder(tokens: int) -> ct.models.MLModel:
        inputs = [_tensor("tokens", (1, tokens), np.int32), _tensor("mask", (1, tokens))]
        return _convert(s3gen.FlowEncoder(flow, tokens), inputs, "mu")

    def step(frames: int) -> ct.models.MLModel:
        mel = (1, s3gen.MEL_CHANNELS, frames)
        inputs = [
            _tensor("x", mel),
            _tensor("mu", mel),
            _tensor("cond", mel),
            _tensor("spks", (1, s3gen.MEL_CHANNELS)),
            _tensor("mask", (1, 1, frames)),
            _tensor("t_emb", (1, s3gen.TIME_EMBEDDING_DIM)),
            _tensor("dt", (1,)),
        ]
        return _convert(s3gen.FlowStep(flow.decoder.estimator), inputs, "x_next")

    return (
        _package("flow_encoder", {f"n{n}": (lambda n=n: encoder(n)) for n in FLOW_TOKENS}),
        _package("flow_step", {f"m{2 * n}": (lambda n=n: step(2 * n)) for n in FLOW_TOKENS}),
    )


def convert_vocoder() -> tuple[Path, Path]:
    hift = reference.model().s3gen.mel2wav
    mel = _tensor("mel", (1, s3gen.MEL_CHANNELS, VOCODER_FRAMES))
    source = _tensor("source", (1, 1, s3gen.SAMPLES_PER_FRAME * VOCODER_FRAMES))
    return (
        _package("f0", {f"t{VOCODER_FRAMES}": lambda: _convert(s3gen.F0(hift), [mel], "f0")}),
        _package("hift", {f"t{VOCODER_FRAMES}": lambda: _convert(s3gen.HiFT(hift), [mel, source], "audio")}),
    )


def export_assets() -> None:
    """Embedding tables, the default voice, the flow noise, the tokenizer and config.json."""
    model = reference.model()
    checkpoint = assets.checkpoint()
    gen = model.conds.gen
    tables = assets.BUILD / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, weight in {
        "text_emb": model.t3.text_emb.weight,
        "text_pos": model.t3.text_pos_emb.emb.weight,
        "speech_emb": model.t3.speech_emb.weight,
        "speech_pos": model.t3.speech_pos_emb.emb.weight,
    }.items():
        weight.detach().numpy().astype("<f2").tofile(tables / f"{name}.f16")

    voice = assets.BUILD / "voices" / assets.VOICE_NAME
    voice.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        cond = model.t3.prepare_conditioning(model.conds.t3)[0]
        spks = model.s3gen.flow.spk_embed_affine_layer(nn.functional.normalize(gen["embedding"], dim=1))[0]
    cond.numpy().astype("<f4").tofile(voice / "t3_cond.f32")
    gen["prompt_token"][0].numpy().astype("<i4").tofile(voice / "prompt_tokens.i32")
    gen["prompt_feat"][0].numpy().astype("<f4").tofile(voice / "prompt_mel.f32")
    spks.numpy().astype("<f4").tofile(voice / "spks.f32")
    (voice / "voice.json").write_text(
        json.dumps(
            {
                "name": assets.VOICE_NAME,
                "source": assets.VOICE_URL,
                "license": "Unstated. Default reference clip of the pack's demo Space "
                "(ResembleAI/Chatterbox-Multilingual-TTS-pt-br, app.py DEFAULT_CONFIG); the model card "
                "says its prompts come from freely available data on the internet.",
                "built_with": "ChatterboxTTS.prepare_conditionals(wav, exaggeration=0.5) from the Space",
                "exaggeration": reference.EXAGGERATION,
                "t3_cond": [int(cond.shape[0]), int(cond.shape[1])],
                "prompt_tokens": int(gen["prompt_token"].shape[1]),
                "prompt_mel": [int(gen["prompt_feat"].shape[1]), s3gen.MEL_CHANNELS],
                "spks": s3gen.MEL_CHANNELS,
            },
            indent=2,
        )
    )
    frames = 2 * FLOW_TOKENS[-1]
    model.s3gen.flow.decoder.rand_noise[0, :, :frames].numpy().astype("<f4").tofile(assets.BUILD / "flow_noise.f32")
    shutil.copy(checkpoint.tokenizer, assets.BUILD / "tokenizer.json")
    linear = model.s3gen.mel2wav.m_source.l_linear
    (assets.BUILD / "config.json").write_text(json.dumps(config(linear), indent=2))


def config(source_linear: nn.Linear) -> dict[str, object]:
    hp = reference.model().t3.hp
    return {
        "model": "ResembleAI/Chatterbox-Multilingual-pt-br (Chatterbox Multilingual V3 single-language pack)",
        "revisions": {"pack": assets.PACK[1], "base": assets.BASE[1], "space": assets.SPACE[1]},
        "license": "MIT (model); tokenizer from ResembleAI/chatterbox (MIT)",
        "sample_rate": reference.SAMPLE_RATE,
        "language_token": "[pt]",
        "text": {"start": hp.start_text_token, "stop": hp.stop_text_token, "vocab": hp.text_tokens_dict_size},
        "speech": {
            "start": hp.start_speech_token,
            "stop": hp.stop_speech_token,
            "vocab": hp.speech_tokens_dict_size,
            "flow_vocab": 6561,
        },
        "t3": {
            "hidden": t3.HIDDEN,
            "layers": t3.LAYERS,
            "heads": t3.HEADS,
            "head_dim": t3.HEAD_DIM,
            "context": t3.CONTEXT,
            "masked": t3.MASKED,
            "rope_inverse_frequencies": t3.rope_inverse_frequencies().tolist(),
            "prefill_lengths": list(PREFILL_LENGTHS),
            "decode_windows": list(DECODE_WINDOWS),
        },
        "sampling": {
            "cfg_weight": 0.5,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
            "min_p": 0.05,
            "top_p": 0.95,
            "max_new_tokens": 1000,
        },
        "flow": {
            "tokens": list(FLOW_TOKENS),
            "token_mel_ratio": s3gen.TOKEN_MEL_RATIO,
            "cfg_rate": s3gen.CFG_RATE,
            "steps": {"low": 4, "normal": 6, "high": 8, "highest": 10},
            "time_embedding_dim": s3gen.TIME_EMBEDDING_DIM,
        },
        "vocoder": {
            "frames": VOCODER_FRAMES,
            "context": VOCODER_CONTEXT,
            "samples_per_frame": s3gen.SAMPLES_PER_FRAME,
            "source_weight": source_linear.weight.detach()[0].tolist(),
            "source_bias": float(source_linear.bias.detach()[0]),
            "harmonics": s3gen.HARMONICS,
            "voiced_threshold": 10.0,
            "sine_amplitude": 0.1,
            "noise_std": 0.003,
        },
        "samples_per_token": reference.SAMPLES_PER_TOKEN,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Convert the pt-BR Chatterbox pack to Core ML in {assets.BUILD}")
    parser.add_argument("--stage", choices=STAGES, action="append")
    args = parser.parse_args()
    assets.BUILD.mkdir(parents=True, exist_ok=True)
    assets.CACHE.mkdir(parents=True, exist_ok=True)
    steps: Mapping[str, Callable[[], object]] = {
        "assets": export_assets,
        "vocoder": convert_vocoder,
        "flow": convert_flow,
        "t3": convert_t3,
    }
    for stage in args.stage or STAGES:
        start = time.time()
        print(f"{stage}: converting", flush=True)
        steps[stage]()
        print(f"{stage}: done in {time.time() - start:.0f}s", flush=True)

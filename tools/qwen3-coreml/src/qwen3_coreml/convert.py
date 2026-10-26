"""Converts Qwen3-TTS 12Hz 0.6B to Core ML packages for the Neural Engine, with the assets Swift needs."""

import argparse
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from coremltools.converters.mil.mil import types
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel
from torch import Tensor, nn
from torch.nn import functional as F

from qwen3_coreml import assets
from qwen3_coreml.modules import KV_WIDTH, Predictor, SpeechDecoder, TalkerPrefill, TalkerStep

BUILD = assets.ROOT / "build" / "qwen3"
# Base with x-vectors of LibriVox readers: closer to native pt-BR pronunciation than any of
# CustomVoice's nine speakers.
CHECKPOINT = assets.BASE
SLOTS = 256
PREFILL = 128
FRAMES = 125
# The checkpoints' generation_config.json.
TEMPERATURE = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.05
MODELS = ("TextProjector", "CodeEmbedder", "CodeDecoder", "MultiCodeDecoder", "SpeechDecoder")


class Lookup(nn.Module):
    """A row of a fixed table: `input_ids [1]` int32 to `input_embeds [1024, 1, 1]`."""

    def __init__(self, table: Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, input_ids: Tensor) -> Tensor:
        return F.embedding(input_ids, self.table).view(-1, 1, 1)


def convert(
    module: nn.Module,
    inputs: Sequence[ct.TensorType],
    outputs: Sequence[str],
    states: Sequence[ct.StateType] = (),
) -> ct.models.MLModel:
    """Trace `module` on examples shaped like `inputs` and convert it to an fp16 ML Program for iOS 18.

    Float examples are positive, not zero: an all-zero token has no norm, and a stateful module
    traced into NaN state fails the converter's state check.
    """
    examples = tuple(
        torch.zeros(tensor.shape.shape, dtype=torch.int32)
        if tensor.dtype == types.int32
        else torch.rand(tensor.shape.shape)
        for tensor in inputs
    )
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), examples)
    return ct.convert(
        traced,
        inputs=list(inputs),
        outputs=[ct.TensorType(name=name, dtype=np.float16) for name in outputs],
        states=list(states) or None,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        # Compiling here would queue behind every other job on the shared ANE compiler service.
        skip_model_load=True,
    )


def int8_weights(model: ct.models.MLModel) -> ct.models.MLModel:
    """`model` with its weights quantized to int8, symmetric per output channel; activations stay fp16.

    The per-frame models read all their weights at every step, so halving them cuts the step time on the
    Neural Engine by about a third.
    """
    config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"
        )
    )
    return cto.linear_quantize_weights(model, config)


def fp16(name: str, shape: tuple[int, ...]) -> ct.TensorType:
    return ct.TensorType(name=name, shape=shape, dtype=np.float16)


def text_projector(model: Qwen3TTSModel) -> ct.models.MLModel:
    """The text embedding already through the text projection MLP, one row per text token."""
    talker = model.model.talker
    with torch.no_grad():
        table = torch.cat(
            [talker.text_projection(rows) for rows in talker.model.text_embedding.weight.split(8192)]
        )
    return convert(
        Lookup(table), [ct.TensorType(name="input_ids", shape=(1,), dtype=np.int32)], ["input_embeds"]
    )


def code_embedder(model: Qwen3TTSModel) -> ct.models.MLModel:
    table = model.model.talker.model.codec_embedding.weight.detach().float()
    return convert(
        Lookup(table), [ct.TensorType(name="input_ids", shape=(1,), dtype=np.int32)], ["input_embeds"]
    )


def code_decoder(model: Qwen3TTSModel, directory: Path) -> Path:
    """CodeDecoder: functions `decode` (one step, stateful cache) and `prefill` (the prompt)."""
    talker = model.model.talker
    width = len(talker.model.layers) * KV_WIDTH
    cache = [
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, width, 1, SLOTS), dtype=np.float16), name=name)
        for name in ("key_cache", "value_cache")
    ]
    decode = convert(
        TalkerStep(talker, SLOTS),
        [
            fp16("input_embeds", (1, 1024, 1, 1)),
            fp16("key_padding_mask", (1, SLOTS)),
            fp16("kv_cache_update_mask", (1, SLOTS)),
        ],
        ["logits", "hidden_states"],
        cache,
    )
    prefill = convert(
        TalkerPrefill(talker, PREFILL),
        [fp16("input_embeds", (1, 1024, 1, PREFILL)), fp16("key_padding_mask", (PREFILL, PREFILL))],
        ["logits", "hidden_states", "key_cache", "value_cache"],
    )
    # Quantized one by one: the optimizer does not take multifunction packages.
    return multifunction(
        {"decode": int8_weights(decode), "prefill": int8_weights(prefill)},
        directory / "CodeDecoder.mlpackage",
    )


def multi_code_decoder(model: Qwen3TTSModel) -> ct.models.MLModel:
    predictor = Predictor(model.model.talker, TEMPERATURE, TOP_K)
    return int8_weights(
        convert(
            predictor,
            [
                fp16("hidden_states", (1, 1024, 1, 1)),
                fp16("code_embeds", (1, 1024, 1, 1)),
                fp16("noise", (15, 2048)),
            ],
            ["codes", "codes_embeds"],
        )
    )


def speech_decoder(model: Qwen3TTSModel) -> ct.models.MLModel:
    decoder = SpeechDecoder(model.model.speech_tokenizer.model.decoder, FRAMES)
    return convert(
        decoder, [ct.TensorType(name="audio_codes", shape=(1, 16, FRAMES), dtype=np.int32)], ["audio"]
    )


def multifunction(functions: dict[str, ct.models.MLModel], path: Path) -> Path:
    """One package with a function per entry, sharing identical weights; the first is the default."""
    descriptor = ct.utils.MultiFunctionDescriptor()
    with tempfile.TemporaryDirectory() as scratch:
        for name, function in functions.items():
            single = Path(scratch) / f"{name}.mlpackage"
            function.save(str(single))
            descriptor.add_function(str(single), src_function_name="main", target_function_name=name)
        descriptor.default_function_name = next(iter(functions))
        shutil.rmtree(path, ignore_errors=True)
        ct.utils.save_multifunction(descriptor, str(path))
    return path


def write_assets(model: Qwen3TTSModel, repo: str, directory: Path) -> None:
    """Tokenizer files, the text-side control embeddings, voices and `config.json`."""
    snapshot = Path(snapshot_download(repo, allow_patterns=["vocab.json", "merges.txt"]))
    for name in ("vocab.json", "merges.txt"):
        shutil.copy(snapshot / name, directory / name)
    config = model.model.config
    talker = model.model.talker
    tokens = {
        "tts_bos": config.tts_bos_token_id,
        "tts_eos": config.tts_eos_token_id,
        "tts_pad": config.tts_pad_token_id,
    }
    with torch.no_grad():
        for name, token in tokens.items():
            embed = talker.text_projection(talker.model.text_embedding(torch.tensor([token])))[0]
            np.save(directory / f"{name}_embed.npy", embed.float().numpy())
    talker_config = config.talker_config
    voices = voice_assets(model, directory)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model": repo,
                "default_voices": {source.language: source.name for source in assets.VOICES},
                "license": "Apache-2.0",
                "sample_rate": assets.SAMPLE_RATE,
                "samples_per_frame": 1920,
                "hidden_size": 1024,
                "slots": SLOTS,
                "prefill_tokens": PREFILL,
                "speech_frames": FRAMES,
                "codebooks": 16,
                "codebook_size": 2048,
                "text_tokens": {
                    "im_start": config.im_start_token_id,
                    "im_end": config.im_end_token_id,
                    "assistant": config.assistant_token_id,
                    "newline": 198,
                    **tokens,
                },
                "codec_tokens": {
                    "pad": talker_config.codec_pad_id,
                    "bos": talker_config.codec_bos_id,
                    "eos": talker_config.codec_eos_token_id,
                    "think": talker_config.codec_think_id,
                    "nothink": talker_config.codec_nothink_id,
                    "think_bos": talker_config.codec_think_bos_id,
                    "think_eos": talker_config.codec_think_eos_id,
                },
                "languages": {
                    code: talker_config.codec_language_id[name] for code, name in assets.LANGUAGES.items()
                },
                "voices": voices,
                "sampling": {
                    "talker": {
                        "temperature": TEMPERATURE,
                        "top_k": TOP_K,
                        "repetition_penalty": REPETITION_PENALTY,
                        "suppressed_from": talker_config.vocab_size - 1024,
                        "min_frames": 2,
                    },
                    "predictor": {
                        "temperature": TEMPERATURE,
                        "top_k": TOP_K,
                        "baked_into": "MultiCodeDecoder",
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def voice_assets(model: Qwen3TTSModel, directory: Path) -> dict[str, dict[str, object]]:
    """The x-vector of each voice, written to `voices/<name>.npy` (1024 float32)."""
    (directory / "voices").mkdir(exist_ok=True)
    voices: dict[str, dict[str, object]] = {}
    for source in assets.VOICES:
        np.save(directory / "voices" / f"{source.name}.npy", assets.speaker_embedding(model, source))
        voices[source.name] = {
            "embedding": f"voices/{source.name}.npy",
            "language": source.language,
            "description": source.description,
            "source": source.url,
            "sha256": source.sha256,
            "seconds": [source.start, source.end],
            "transcript": source.transcript,
            "license": source.license,
        }
    return voices


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Convert Qwen3-TTS 0.6B to Core ML packages in {BUILD}")
    parser.add_argument("--model", choices=MODELS, action="append", help="convert only these (default: all)")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    print(f"loading {CHECKPOINT}", flush=True)
    model = assets.load(CHECKPOINT)
    print("assets", flush=True)
    write_assets(model, CHECKPOINT, BUILD)
    for name in args.model or MODELS:
        print(f"{name}: converting", flush=True)
        match name:
            case "CodeDecoder":
                path = code_decoder(model, BUILD)
            case _:
                build = {
                    "TextProjector": text_projector,
                    "CodeEmbedder": code_embedder,
                    "MultiCodeDecoder": multi_code_decoder,
                    "SpeechDecoder": speech_decoder,
                }[name]
                path = BUILD / f"{name}.mlpackage"
                shutil.rmtree(path, ignore_errors=True)
                build(model).save(str(path))
        print(f"{name}: {path}", flush=True)

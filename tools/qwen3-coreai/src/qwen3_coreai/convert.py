"""qwen3-coreai-convert: Qwen3-TTS 0.6B as Core AI models for the Neural Engine, in build/aimodel.

The models are qwen3-coreml's, rearranged so that every function runs on the Neural Engine alone:
MPSGraph puts what the Neural Engine does not take on the GPU, which iOS denies to apps in the
background.
- The norms scale by the L1 norm instead of `torch.linalg.vector_norm` (`norms`).
- The talker's decode takes its KV cache as input and returns the step's key and value
  (`talker.Decode`): the Neural Engine compiler crashes on writes into a state.
- The code predictor finds the top-k threshold by bisection, not with `torch.topk`, and keeps its
  keys and values in fixed buffers (`predictor`).
- Lookups by integer index run on the GPU, so the app does them from tables in build/assets: the
  text embedding through the text projection (TextProjector in Core ML), the codec embedding
  (CodeEmbedder) and the speech decoder's codebooks.
- The speech decoder's first upsampling convolutions are transposed, kernel 2 and stride 2, which
  `coreai-build compile` gets wrong (rel 1.06 for the first, where specializing on the device
  gives 0.0009). They run as convolutions followed by an interleave (`_StrideUpsample`).

`coreai-aot` then compiles each package for every iPhone architecture.
"""

import argparse
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from coreai_torch import TorchConverter, get_decomp_table
from qwen3_coreml import assets
from qwen3_coreml import convert as coreml
from qwen3_coreml.modules import SpeechDecoder, TalkerPrefill
from qwen_tts import Qwen3TTSModel
from torch import Tensor, nn
from torch.nn import functional as F

from qwen3_coreai import norms
from qwen3_coreai.predictor import NeuralEnginePredictor
from qwen3_coreai.quantize import int8_weights
from qwen3_coreai.talker import Decode

BUILD = Path(__file__).resolve().parents[2] / "build" / "aimodel"
ASSETS = BUILD.parent / "assets"
PACKAGES = ("CodeDecoder", "MultiCodeDecoder", "SpeechDecoder")
F16 = torch.float16


@dataclass(frozen=True)
class Function:
    module: nn.Module
    inputs: Mapping[str, Tensor]
    outputs: Sequence[str]


def save(path: Path, functions: Mapping[str, Function]) -> None:
    """Writes `functions` to `path`, an .aimodel with one entrypoint per function."""
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    for name, function in functions.items():
        with torch.inference_mode():
            exported = torch.export.export(
                function.module.eval(), args=tuple(function.inputs.values())
            ).run_decompositions(get_decomp_table())
        converter.add_exported_program(
            exported,
            input_names=list(function.inputs),
            output_names=list(function.outputs),
            entrypoint_name=name,
        )
    if path.exists():
        shutil.rmtree(path)
    converter.to_coreai().save_asset(path)


def _zeros(*shape: int) -> Tensor:
    return torch.zeros(shape, dtype=F16)


def code_decoder(model: Qwen3TTSModel) -> dict[str, Function]:
    talker = model.model.talker
    width = len(talker.model.layers) * 1024
    slots, prefill = coreml.SLOTS, coreml.PREFILL
    decode = Decode(talker, slots).half()
    int8_weights(decode)
    first = TalkerPrefill(talker, prefill).half()
    int8_weights(first)
    return {
        "prefill": Function(
            first,
            {"input_embeds": _zeros(1, 1024, 1, prefill), "key_padding_mask": _zeros(prefill, prefill)},
            ["logits", "hidden_states", "key_cache", "value_cache"],
        ),
        "decode": Function(
            decode,
            {
                "input_embeds": _zeros(1, 1024, 1, 1),
                "key_padding_mask": _zeros(1, slots),
                "position": _zeros(1, slots),
                "key_cache": _zeros(1, width, 1, slots),
                "value_cache": _zeros(1, width, 1, slots),
            },
            ["logits", "hidden_states", "key", "value"],
        ),
    }


def multi_code_decoder(model: Qwen3TTSModel) -> dict[str, Function]:
    predictor = NeuralEnginePredictor(model.model.talker, coreml.TEMPERATURE, coreml.TOP_K).half()
    int8_weights(predictor)
    inputs = {
        "hidden_states": _zeros(1, 1024, 1, 1),
        "code_embeds": _zeros(1, 1024, 1, 1),
        "noise": _zeros(15, 2048),
    }
    return {"main": Function(predictor, inputs, ["codes", "codes_embeds"])}


class _StrideUpsample(nn.Module):
    """A `ConvTranspose1d(kernel=s, stride=s)` as a 1x1 convolution to `s` times the channels
    followed by an interleave: output `t * s + r` is `W[:, :, r] x[t] + b`."""

    def __init__(self, transposed: nn.ConvTranspose1d) -> None:
        super().__init__()
        weight = transposed.weight.detach().float()  # [in, out, s]
        stride = transposed.stride[0]
        inputs, outputs, kernel = weight.shape
        assert kernel == stride
        self.weight = nn.Parameter(weight.permute(1, 2, 0).reshape(outputs * stride, inputs, 1))
        self.bias = nn.Parameter(transposed.bias.detach().float().repeat_interleave(stride))
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        batch, _, length = x.shape
        y = F.conv1d(x, self.weight, self.bias)
        return y.view(batch, -1, self.stride, length).transpose(2, 3).reshape(batch, -1, length * self.stride)


class _SpeechDecoder(nn.Module):
    """`SpeechDecoder` from the sum of the codes' embeddings, `[1, 512, frames]`, which the app
    looks up in speech_codebooks.f16."""

    def __init__(self, decoder: SpeechDecoder) -> None:
        super().__init__()
        decoder.upsample = nn.ModuleList(
            nn.ModuleList([_StrideUpsample(transposed.conv), block]) for transposed, block in decoder.upsample
        )
        self.decoder = decoder

    def forward(self, codes_sum: Tensor) -> Tensor:
        return self.decoder.decode(codes_sum)


def speech_decoder(model: Qwen3TTSModel) -> dict[str, Function]:
    decoder = SpeechDecoder(model.model.speech_tokenizer.model.decoder, coreml.FRAMES)
    module = _SpeechDecoder(decoder).half()
    return {"main": Function(module, {"codes_sum": _zeros(1, 512, coreml.FRAMES)}, ["audio"])}


def export_assets(model: Qwen3TTSModel) -> None:
    """The tables the app reads in place of lookups, Float16 and row-major:
    text_embeddings.f16 (text vocabulary, 1024), the text embedding through the text projection;
    code_embeddings.f16 (codec vocabulary, 1024); speech_codebooks.f16 (16 · 2048, 512), the
    speech decoder's codebooks, codebook-major.

    qwen3-coreml's convert writes the rest the app reads: config.json, the tokenizer, the
    control embeddings and the voices.
    """
    ASSETS.mkdir(parents=True, exist_ok=True)
    talker = model.model.talker
    with torch.inference_mode():
        text = torch.cat(
            [talker.text_projection(rows) for rows in talker.model.text_embedding.weight.split(8192)]
        )
    tables = {
        "text_embeddings": text,
        "code_embeddings": talker.model.codec_embedding.weight,
        "speech_codebooks": SpeechDecoder(model.model.speech_tokenizer.model.decoder, coreml.FRAMES).codebooks,
    }
    for name, table in tables.items():
        table.detach().float().numpy().astype("<f2").tofile(ASSETS / f"{name}.f16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", choices=(*PACKAGES, "assets"), action="append")
    args = parser.parse_args()
    norms.install()
    model = assets.load(coreml.CHECKPOINT)
    builders: Mapping[str, Callable[[Qwen3TTSModel], dict[str, Function]]] = {
        "CodeDecoder": code_decoder,
        "MultiCodeDecoder": multi_code_decoder,
        "SpeechDecoder": speech_decoder,
    }
    BUILD.mkdir(parents=True, exist_ok=True)
    selected = args.package or (*PACKAGES, "assets")
    if "assets" in selected:
        export_assets(model)
        print(f"{ASSETS}: tables", flush=True)
    for package in (p for p in PACKAGES if p in selected):
        started = time.monotonic()
        path = BUILD / f"{package}.aimodel"
        functions = builders[package](model)
        save(path, functions)
        print(f"{path}: {', '.join(functions)} in {time.monotonic() - started:.0f} s", flush=True)

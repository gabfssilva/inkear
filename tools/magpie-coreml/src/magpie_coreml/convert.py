import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil.mil import Block, Program
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from nemo.collections.tts.models import MagpieTTSModel
from torch import nn

from magpie_coreml import checkpoint, frontend, pipeline
from magpie_coreml.checkpoint import CODES, SAMPLES_PER_FRAME, VOCAB
from magpie_coreml.modules import (
    CACHE,
    CONTEXT,
    CROSS_DIM,
    HEAD_DIM,
    HEADS,
    TEXT,
    TOP_K,
    Codec,
    D,
    DecoderPrefill,
    DecoderStep,
    LocalTransformer,
    TextEncoder,
)
from magpie_coreml.stages import CODEC_FRAMES

BUILD = Path(__file__).resolve().parents[2] / "build" / "magpie"
LAYERS = 12
STAGES = ("text_encoder", "decoder_prefill", "decoder_step", "local_transformer", "nanocodec_decoder")


@dataclass(frozen=True)
class Port:
    name: str
    shape: tuple[int, ...]
    dtype: type[np.generic] = np.float32

    def example(self) -> torch.Tensor:
        """Tracing input; float ports trace in fp32 whatever their Core ML type."""
        return (
            torch.zeros(self.shape, dtype=torch.int32) if self.dtype is np.int32 else torch.ones(self.shape)
        )

    def tensor(self) -> ct.TensorType:
        return ct.TensorType(name=self.name, shape=self.shape, dtype=self.dtype)


@dataclass(frozen=True)
class Stage:
    module: nn.Module
    inputs: list[Port]
    outputs: list[Port]
    states: list[Port]


def _cross() -> list[Port]:
    return [Port(f"xa_{kv}_{i}", (1, 1, TEXT, CROSS_DIM), np.float16) for kv in "kv" for i in range(LAYERS)]


def _caches(length: int) -> list[Port]:
    return [
        Port(f"sa_{kv}_{i}", (2, HEADS, length, HEAD_DIM), np.float16) for kv in "kv" for i in range(LAYERS)
    ]


def _stage(name: str, model: MagpieTTSModel) -> Stage:
    """Tensors passed between stages are fp16, as computed; control inputs and outputs read by the
    host loop are fp32 or int32."""
    match name:
        case "text_encoder":
            return Stage(
                TextEncoder(model),
                [Port("tokens", (1, TEXT), np.int32), Port("mask", (1, TEXT))],
                [Port("encoder_output", (1, TEXT, D), np.float16)],
                [],
            )
        case "decoder_prefill":
            return Stage(
                DecoderPrefill(model),
                [
                    Port("speaker_idx", (1,), np.int32),
                    Port("encoder_output", (1, TEXT, D), np.float16),
                    Port("encoder_mask", (1, TEXT)),
                ],
                _caches(CONTEXT) + _cross(),
                [],
            )
        case "decoder_step":
            return Stage(
                DecoderStep(model, torch.float16),
                [
                    Port("audio_emb", (1, 1, D), np.float16),
                    Port("position", (1,), np.int32),
                    Port("encoder_mask", (1, TEXT)),
                    Port("attn_prior", (1, TEXT)),
                    Port("cfg_scale", (1,)),
                    *_cross(),
                ],
                [
                    Port("logits", (1, 1, CODES, VOCAB)),
                    Port("h_last", (2, 1, D), np.float16),
                    Port("alignment", (1, TEXT)),
                ],
                _caches(CACHE),
            )
        case "local_transformer":
            return Stage(
                LocalTransformer(model),
                [
                    Port("h_last", (2, 1, D), np.float16),
                    Port("gumbel", (CODES, VOCAB)),
                    Port("temperature", (1,)),
                    Port("cfg_scale", (1,)),
                    Port("forbid_eos", (1,)),
                ],
                [Port("codes", (1, CODES)), Port("audio_emb", (1, 1, D), np.float16)],
                [],
            )
        case "nanocodec_decoder":
            codec = Codec(model._codec_model.audio_decoder)
            histories = [(channels, (1, channels, length)) for channels, length in codec.histories.items()]
            return Stage(
                codec,
                [
                    Port("latents", (1, 32, CODEC_FRAMES)),
                    *(Port(f"history_{channels}", shape, np.float16) for channels, shape in histories),
                ],
                [
                    Port("audio", (1, CODEC_FRAMES * SAMPLES_PER_FRAME)),
                    *(Port(f"history_{channels}_out", shape, np.float16) for channels, shape in histories),
                ],
                [],
            )
        case _:
            raise ValueError(name)


def convert(name: str, model: MagpieTTSModel, precision: ct.precision) -> ct.models.MLModel:
    stage = _stage(name, model)
    traced = torch.jit.trace(
        stage.module.eval(), tuple(port.example() for port in stage.inputs), check_trace=False
    )
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.insert_pass(
        pipeline.passes.index("common::canonicalize_inplace_pattern"), "magpie::write_states_directly"
    )
    return ct.convert(
        traced,
        inputs=[port.tensor() for port in stage.inputs],
        outputs=[ct.TensorType(name=port.name, dtype=port.dtype) for port in stage.outputs],
        states=[
            ct.StateType(wrapped_type=ct.TensorType(shape=port.shape, dtype=port.dtype), name=port.name)
            for port in stage.states
        ],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=pipeline,
    )


@register_pass(namespace="magpie")
class write_states_directly(AbstractGraphPass):
    """Drop the whole-tensor slice_update the torch frontend puts before every state write.

    `cache[:] = value` converts to coreml_update_state(slice_update(read_state(cache), value)).
    With two or more states, a model holding that pattern fails to load (MIL execution plan
    error -14, macOS 27); writing `value` directly is equivalent and loads.
    """

    def apply(self, prog: Program) -> None:
        for function in prog.functions.values():
            self._apply(function)

    @block_context_manager
    def _apply(self, block: Block) -> None:
        for op in list(block.operations):
            update = op.value.op if op.op_type == "coreml_update_state" else None
            if (
                update is not None
                and update.op_type == "slice_update"
                and update.update.shape == update.x.shape
            ):
                block.replace_uses_of_var_after_op(
                    anchor_op=update, old_var=update.outputs[0], new_var=update.update
                )
        block.remove_ops(
            [
                op
                for op in list(block.operations)
                if op.op_type == "slice_update" and not op.outputs[0].child_ops
            ]
        )


def bos_embedding(model: MagpieTTSModel) -> np.ndarray:
    """Decoder input of the first step: mean of the AUDIO_BOS rows of the 16 code embeddings."""
    rows = torch.stack([embedding.weight[checkpoint.AUDIO_BOS] for embedding in model.audio_embeddings])
    return rows.mean(dim=0).reshape(1, 1, D).numpy().astype(np.float32)


def export_assets(model: MagpieTTSModel) -> None:
    """Front-end files, the first decoder input and the constants of the generation loop."""
    frontend.export(model, BUILD)
    bos_embedding(model).tofile(BUILD / "audio_bos_embedding.f32")
    config = {
        "model": f"{checkpoint.REPO}@{checkpoint.REVISION} (v2607)",
        "sample_rate": checkpoint.SAMPLE_RATE,
        "samples_per_frame": SAMPLES_PER_FRAME,
        "codebooks": checkpoint.CODEBOOKS,
        "frames_per_step": checkpoint.STACK,
        "codes_per_step": CODES,
        "vocab_per_codebook": VOCAB,
        "audio_bos": checkpoint.AUDIO_BOS,
        "audio_eos": checkpoint.AUDIO_EOS,
        "forbidden_codes": list(checkpoint.FORBIDDEN),
        "text_tokens": TEXT,
        "context_frames": CONTEXT,
        "cache_positions": CACHE,
        "max_steps": pipeline.MAX_STEPS,
        "min_frames": pipeline.MIN_FRAMES,
        "top_k": TOP_K,
        "attention_prior": {
            "epsilon": pipeline.PRIOR_EPSILON,
            "lookahead": pipeline.LOOKAHEAD,
            "sink_threshold": pipeline.SINK,
        },
        "quality": {
            "cfg_scale": pipeline.DEFAULT.cfg_scale,
            "temperature": pipeline.DEFAULT.temperature,
            "attention_prior": pipeline.DEFAULT.attention_prior,
        },
        "codec_frames_per_call": CODEC_FRAMES,
        "fsq_levels": [8, 7, 6, 6],
        "speakers": checkpoint.SPEAKERS,
        "languages": {"pt": "pt-BR", "en": "en-US"},
        "sources": {
            "weights": "nvidia/magpie_tts_multilingual_357m v2607 and nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps, "
            "NVIDIA Open Model License; the 5 speakers are baked into decoder_prefill.mlpackage",
            "lexicon_pt.tsv": "pt_br_prondict-v1.0.dict shipped inside the v2607 .nemo (NVIDIA Open Model License)",
            "lexicon_en.tsv": "ipa_cmudict-0.7b_nv23.01.txt shipped inside the .nemo (CMUdict, BSD-style)",
            "heteronyms_en.txt": "heteronyms-052722 shipped inside the .nemo (NeMo, Apache-2.0)",
        },
    }
    (BUILD / "magpie.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Convert Magpie TTS v2607 to Core ML packages in {BUILD}")
    parser.add_argument("--stage", choices=STAGES, action="append")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    model = checkpoint.load().model
    print("assets", flush=True)
    export_assets(model)
    stages: Sequence[str] = args.stage or STAGES
    for index, name in enumerate(stages, 1):
        print(f"[{index}/{len(stages)}] {name}", flush=True)
        convert(name, model, ct.precision.FLOAT16).save(str(BUILD / f"{name}.mlpackage"))

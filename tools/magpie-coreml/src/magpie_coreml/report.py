import argparse
import time
import wave
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import coremltools as ct
import mlx_whisper
import numpy as np
import torch
from coremltools.models.compute_plan import MLComputePlan
from nemo.collections.tts.models import MagpieTTSModel

from magpie_coreml import checkpoint
from magpie_coreml.checkpoint import SAMPLE_RATE, SPEAKERS
from magpie_coreml.convert import BUILD, STAGES, bos_embedding
from magpie_coreml.frontend import Frontend
from magpie_coreml.metrics import snr_db, word_error_rate
from magpie_coreml.modules import CONTEXT, TEXT
from magpie_coreml.pipeline import DEFAULT, Quality, generate
from magpie_coreml.stages import CoreMLStages, TorchStages

SENTENCES = {
    "pt": "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central um rapaz aqui "
    "do bairro, que eu conheço de vista e de chapéu.",
    "en": "The quick brown fox jumps over the lazy dog, and then it runs back into the forest before anyone notices.",
}
SPEAKER = SPEAKERS["Sofia"]
SAMPLES = BUILD.parent / "samples"
WHISPER = "mlx-community/whisper-large-v3-turbo"


@dataclass
class Timed:
    """CoreMLStages with the wall time of every call recorded per stage."""

    stages: CoreMLStages
    seconds: dict[str, list[float]] = field(default_factory=dict)

    def _timed[T](self, name: str, call: Callable[[], T]) -> T:
        start = time.perf_counter()
        result = call()
        self.seconds.setdefault(name, []).append(time.perf_counter() - start)
        return result

    def encode(self, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._timed("text_encoder", lambda: self.stages.encode(tokens, mask))

    def prefill(self, speaker: int, encoder_output: np.ndarray, mask: np.ndarray) -> None:
        self._timed("decoder_prefill", lambda: self.stages.prefill(speaker, encoder_output, mask))

    def step(
        self, audio_emb: np.ndarray, position: int, mask: np.ndarray, prior: np.ndarray, cfg_scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._timed(
            "decoder_step", lambda: self.stages.step(audio_emb, position, mask, prior, cfg_scale)
        )

    def sample(
        self, h_last: np.ndarray, gumbel: np.ndarray, temperature: float, cfg_scale: float, forbid_eos: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._timed(
            "local_transformer",
            lambda: self.stages.sample(h_last, gumbel, temperature, cfg_scale, forbid_eos),
        )

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self._timed("nanocodec_decoder", lambda: self.stages.decode(codes))


def placement(package: Path) -> float:
    """Share of the estimated cost MLComputePlan puts on the Neural Engine under CPU_AND_NE."""
    units = ct.ComputeUnit.CPU_AND_NE
    model = ct.models.MLModel(str(package), compute_units=units)
    plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(), compute_units=units)
    cost = Counter[str]()
    for operation in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        estimate = plan.get_estimated_cost_for_mlprogram_operation(operation)
        if usage is not None:
            cost[type(usage.preferred_compute_device).__name__] += estimate.weight if estimate else 0.0
    return cost["MLNeuralEngineComputeDevice"] / (sum(cost.values()) or 1.0)


def nemo_codes(model: MagpieTTSModel, ids: list[int], speaker: int) -> np.ndarray:
    """NeMo's own single-chunk inference (do_tts without the text normalizer) -> (frames, 8) codes."""
    model.decoder.reset_cache(use_cache=True)
    batch = {"text": torch.tensor([ids]), "text_lens": torch.tensor([len(ids)]), "speaker_indices": speaker}
    out = model.generate_speech(
        batch,
        chunk_state=model.create_chunk_state(1),
        end_of_text=[True],
        beginning_of_text=True,
        use_cfg=True,
        use_local_transformer_for_inference=True,
    )
    return out.predicted_codes[0, :, : out.predicted_codes_lens[0]].T.numpy()


def transcribe(audio: np.ndarray, path: Path, language: str) -> str:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return mlx_whisper.transcribe(str(path), path_or_hf_repo=WHISPER, language=language)["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Parity, placement, speed and samples of the packages in {BUILD}"
    )
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    ck = checkpoint.load()
    model = ck.model
    bos = bos_embedding(model)
    reference = TorchStages(ck)
    coreml = CoreMLStages(BUILD, ct.ComputeUnit.CPU_AND_NE)
    ids = {language: checkpoint.text_ids(model, text, language) for language, text in SENTENCES.items()}

    print("front end, exported files vs NeMo's tokenizer")
    for language, text in SENTENCES.items():
        print(
            f"  {language}: equal {Frontend.load(BUILD, language).encode(text) == ids[language]}, {len(ids[language])} ids"
        )

    print("parity on the pt sentence, Core ML CPU_AND_NE vs the PyTorch fp32 modules (relative max error)")
    _parity(reference, coreml, ids["pt"], bos)
    greedy = Quality(cfg_scale=DEFAULT.cfg_scale, temperature=0.0, attention_prior=True)
    model.inference_parameters.temperature = 0.0
    for language in SENTENCES:
        mine = generate(reference, ids[language], SPEAKER, greedy, bos, np.random.default_rng(0)).codes
        nemo = nemo_codes(model, ids[language], SPEAKER)
        print(
            f"  greedy {language}: PyTorch modules {len(mine)} frames, NeMo {len(nemo)}, equal {np.array_equal(mine, nemo)}"
        )
    model.inference_parameters.temperature = DEFAULT.temperature

    print("neural engine share of estimated cost (MLComputePlan, CPU_AND_NE)")
    for name in STAGES:
        print(f"  {name:18s} {placement(BUILD / f'{name}.mlpackage'):6.1%}")

    print(f"speed on the pt sentence, CPU_AND_NE, {args.runs} runs after a warm-up")
    timed = Timed(coreml)
    _synthesize(coreml, ids["pt"], bos, 0)
    rtfs = []
    for seed in range(args.runs):
        timed.seconds.clear()
        start = time.perf_counter()
        audio = _synthesize(timed, ids["pt"], bos, seed)
        rtfs.append((time.perf_counter() - start) / (len(audio) / SAMPLE_RATE))
        cells = "  ".join(
            f"{name} {np.median(values) * 1e3:.1f} ms x{len(values)}"
            for name, values in timed.seconds.items()
        )
        print(f"  {len(audio) / SAMPLE_RATE:.2f} s of audio, rtf {rtfs[-1]:.3f}  {cells}")
    print(f"  median rtf {np.median(rtfs):.3f}")

    print(f"samples in {SAMPLES}, Whisper large-v3-turbo")
    for language, text in SENTENCES.items():
        torch.manual_seed(0)
        for label, audio in (
            ("pytorch", reference.decode(nemo_codes(model, ids[language], SPEAKER))),
            ("coreml", _synthesize(coreml, ids[language], bos, 0)),
        ):
            heard = transcribe(audio, SAMPLES / f"{language}_{label}.wav", language)
            print(f"  {language} {label:8s} wer {word_error_rate(heard, text):.3f}  {heard}")
    print("speakers in pt, Core ML, 2 seeds")
    for name, speaker in SPEAKERS.items():
        errors = []
        for seed in range(2):
            audio = _synthesize(coreml, ids["pt"], bos, seed, speaker)
            heard = transcribe(audio, SAMPLES / f"pt_coreml_{name.lower()}_{seed}.wav", "pt")
            errors.append(word_error_rate(heard, SENTENCES["pt"]))
        print(f"  {name:6s} wer {np.mean(errors):.3f}")


def _synthesize(
    stages: CoreMLStages | Timed, ids: list[int], bos: np.ndarray, seed: int, speaker: int = SPEAKER
) -> np.ndarray:
    return stages.decode(generate(stages, ids, speaker, DEFAULT, bos, np.random.default_rng(seed)).codes)


def _parity(reference: TorchStages, coreml: CoreMLStages, ids: list[int], bos: np.ndarray) -> None:
    def relative(actual: np.ndarray, expected: np.ndarray) -> float:
        return float(np.abs(actual.astype(np.float32) - expected).max() / np.abs(expected).max())

    tokens = np.zeros((1, TEXT), np.int32)
    tokens[0, : len(ids)] = ids
    mask = np.zeros((1, TEXT), np.float32)
    mask[0, : len(ids)] = 1
    encoded = reference.encode(tokens, mask)
    print(
        f"  text_encoder       {relative(coreml.encode(tokens, mask)[:, : len(ids)], encoded[:, : len(ids)]):.1e}"
    )
    reference.prefill(SPEAKER, encoded, mask)
    coreml.prefill(SPEAKER, encoded, mask)
    caches = max(
        relative(
            coreml.state.read_state(f"sa_{kv}_{i}")[:, :, :CONTEXT],
            getattr(reference.stepper, f"sa_{kv}_{i}")[:, :, :CONTEXT].numpy(),
        )
        for kv in "kv"
        for i in range(12)
    )
    print(f"  decoder_prefill    {caches:.1e} (self-attention caches)")
    prior = np.ones(TEXT, np.float32)
    logits, h, _ = reference.step(bos, CONTEXT, mask, prior, DEFAULT.cfg_scale)
    logits_c, h_c, _ = coreml.step(bos, CONTEXT, mask, prior, DEFAULT.cfg_scale)
    agree = float((logits_c.argmax(-1) == logits.argmax(-1)).mean())
    print(
        f"  decoder_step       logits {relative(logits_c, logits):.1e}, h_last {relative(h_c, h):.1e}, argmax agree {agree:.0%}"
    )
    gumbel = np.random.default_rng(0).gumbel(size=(16, 2024)).astype(np.float32)
    codes, _ = reference.sample(h, gumbel, DEFAULT.temperature, DEFAULT.cfg_scale, True)
    codes_c, _ = coreml.sample(h, gumbel, DEFAULT.temperature, DEFAULT.cfg_scale, True)
    print(f"  local_transformer  same sampled codes {int((codes == codes_c).sum())}/16 (same noise)")
    random_codes = np.random.default_rng(0).integers(0, 2016, size=(96, 8)).astype(np.int32)
    print(
        f"  nanocodec_decoder  snr {snr_db(coreml.decode(random_codes), reference.decode(random_codes)):.1f} dB vs NeMo's codec"
    )

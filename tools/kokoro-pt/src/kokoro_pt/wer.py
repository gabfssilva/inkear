"""Synthesizes the test sentences with Kokoro from each phonemizer and scores Whisper's transcripts (WER)."""

import re
import sys
import unicodedata
from pathlib import Path

import jiwer
import mlx_whisper
import soundfile
import torch
from huggingface_hub import hf_hub_download
from kokoro import KModel

from kokoro_pt.score import kokoro

WHISPER = "mlx-community/whisper-large-v3-turbo"


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("–", " ").replace("—", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def main() -> None:
    sources, out = Path(sys.argv[1]), Path(sys.argv[2])
    voice = sys.argv[3] if len(sys.argv) > 3 else "pf_dora"
    model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    pack = torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{voice}.pt"), weights_only=True)
    rows = [line.split("\t") for line in sources.read_text().splitlines()]
    header, rows = rows[0], rows[1:]
    texts = [r[0] for r in rows]
    for column, name in enumerate(header[1:], 1):
        hypotheses = []
        for n, row in enumerate(rows):
            ps = kokoro(row[column])[:510]
            with torch.no_grad():
                audio = model(ps, pack[len(ps) - 1], 1.0).numpy()
            path = out / f"{name}-{n:02}.wav"
            soundfile.write(path, audio, 24000)
            hypotheses.append(
                mlx_whisper.transcribe(str(path), path_or_hf_repo=WHISPER, language="pt")["text"]
            )
            print(f"{name} {n:02} {hypotheses[-1]}", flush=True)
        wer = jiwer.wer([normalized(t) for t in texts], [normalized(h) for h in hypotheses])
        print(f"WER {name}: {wer:.4f}", flush=True)


if __name__ == "__main__":
    main()

import inspect
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

from config import CHUNK_SECONDS, HF_TOKEN, SRAVAANI_MODEL
from models.lid import VaaniLanguageIdentifier
from utils.languages import get_language


class TimestampUnavailableError(RuntimeError):
    pass


class SraVaaniTranscriber:
    """SraVaani transcription with VAD-derived, audio-grounded timestamps.

    Whisper is intentionally not used. SraVaani provides the text while
    voice-activity detection provides the start/end boundaries of each
    spoken region. These are region-level timestamps, not word timestamps.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        print(f"Loading SraVaani model on {self.device}...")
        self.model = AutoModel.from_pretrained(
            SRAVAANI_MODEL,
            **kwargs,
        ).to(self.device).eval()
        print("SraVaani model loaded successfully.")

        print("Loading Vaani language identifier...")
        self.lid = VaaniLanguageIdentifier()
        print("Vaani language identifier loaded successfully.")

    @staticmethod
    def _pick(obj, *names):
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    @staticmethod
    def _save_chunk(audio: np.ndarray, sr: int) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        sf.write(handle.name, audio, sr, subtype="PCM_16")
        return handle.name

    @staticmethod
    def _speech_regions(audio: np.ndarray, sr: int):
        """Find speech-like regions using librosa's energy-based VAD.

        The returned intervals are grounded in the actual audio. We pad each
        region slightly so the beginning/end of words are less likely to be
        clipped before SraVaani sees them.
        """
        if len(audio) == 0:
            return []

        intervals = librosa.effects.split(
            audio,
            top_db=32,
            frame_length=2048,
            hop_length=512,
        )

        regions = []
        pad = 0.12
        minimum_duration = 0.20

        for start, end in intervals:
            start_s = max(0.0, start / sr - pad)
            end_s = min(len(audio) / sr, end / sr + pad)
            if end_s - start_s >= minimum_duration:
                regions.append((start_s, end_s))

        # Merge regions separated by very short silence so subtitles don't
        # become unnecessarily fragmented.
        merged = []
        for start, end in regions:
            if merged and start - merged[-1][1] <= 0.35:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        return merged

    def _sravaani_text(self, path: str, language_label: str | None):
        signature = inspect.signature(self.model.transcribe)
        kwargs = {"return_hypotheses": True}
        if language_label and "language" in signature.parameters:
            kwargs["language"] = language_label

        try:
            hypotheses = self.model.transcribe(path, **kwargs)
        except Exception as exc:
            print(f"SraVaani inference failed: {exc}")
            return None

        print("========== SRAVAANI RAW OUTPUT ==========")
        print(f"Output type: {type(hypotheses)}")
        print(f"Raw output:\n{hypotheses}")
        print("===========================================")

        if not hypotheses:
            return None

        texts = []
        for hypothesis in hypotheses:
            text = self._pick(
                hypothesis,
                "text",
                "transcript",
                "sentence",
            )
            if text and str(text).strip():
                texts.append(str(text).strip())

        return " ".join(texts).strip() or None

    def _identify_language(self, audio_path: str, requested: str | None):
        if requested and requested != "auto":
            return get_language(requested)["label"], 1.0

        try:
            label, confidence = self.lid.identify(audio_path)
            print(f"Detected language: {label} ({confidence:.3f})")
            return label, confidence
        except Exception as exc:
            print(f"Language identification failed: {exc}")
            return None, 0.0

    def transcribe(self, audio_path: str, language: str | None = None):
        audio, sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False,
        )

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr <= 0 or len(audio) == 0:
            raise TimestampUnavailableError(
                "The normalized audio contains no samples."
            )

        total_duration = len(audio) / sr
        chunk_size = max(1, int(CHUNK_SECONDS * sr))
        total_chunks = int(np.ceil(len(audio) / chunk_size))
        segments = []

        print(
            f"Splitting audio into {total_chunks} processing chunks "
            f"of up to {CHUNK_SECONDS} seconds..."
        )

        for index, chunk_start in enumerate(
            range(0, len(audio), chunk_size),
            start=1,
        ):
            chunk_end = min(chunk_start + chunk_size, len(audio))
            chunk_audio = audio[chunk_start:chunk_end]
            chunk_offset = chunk_start / sr
            chunk_duration = len(chunk_audio) / sr

            print("\n" + "=" * 50)
            print(f"PROCESSING CHUNK {index}/{total_chunks}")
            print(f"Time offset: {chunk_offset:.2f} seconds")

            regions = self._speech_regions(chunk_audio, sr)

            if not regions:
                print("No speech-like region detected in this chunk.")
                continue

            print(f"Detected {len(regions)} speech region(s) in chunk.")

            for region_index, (region_start, region_end) in enumerate(
                regions,
                start=1,
            ):
                start_sample = max(0, int(region_start * sr))
                end_sample = min(len(chunk_audio), int(region_end * sr))
                region_audio = chunk_audio[start_sample:end_sample]

                if len(region_audio) < int(0.20 * sr):
                    continue

                absolute_start = chunk_offset + start_sample / sr
                absolute_end = min(
                    total_duration,
                    chunk_offset + end_sample / sr,
                )

                region_path = self._save_chunk(region_audio, sr)

                try:
                    print(
                        f"\nSpeech region {region_index}/{len(regions)}: "
                        f"{absolute_start:.2f}s -> {absolute_end:.2f}s"
                    )

                    language_label, confidence = self._identify_language(
                        region_path,
                        language,
                    )

                    text = self._sravaani_text(
                        region_path,
                        language_label,
                    )

                    if not text:
                        print("No speech recognized in this region.")
                        continue

                    # VAD boundaries are the timestamps. They are based on the
                    # real audio signal and therefore don't depend on a second
                    # ASR model or fabricated fixed durations.
                    segments.append(
                        {
                            "start": absolute_start,
                            "end": absolute_end,
                            "text": text,
                            "language": language_label,
                            "language_confidence": confidence,
                            "asr": "sravaani-vad",
                        }
                    )

                finally:
                    Path(region_path).unlink(missing_ok=True)

        if not segments:
            raise TimestampUnavailableError(
                "No speech segments were recognized. Check the audio, "
                "language, model access, and media quality."
            )

        segments.sort(
            key=lambda item: (item["start"], item["end"])
        )

        print("\n" + "=" * 50)
        print(f"TOTAL TIMESTAMPED SEGMENTS: {len(segments)}")
        print("Timestamp source: VAD speech boundaries + SraVaani text")
        print("=" * 50)

        return segments

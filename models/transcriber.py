import inspect
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, pipeline

from config import CHUNK_SECONDS, HF_TOKEN, SRAVAANI_MODEL, TIMESTAMP_ASR_MODEL, WHISPER_ENABLED
from models.lid import VaaniLanguageIdentifier
from utils.languages import get_language


class TimestampUnavailableError(RuntimeError):
    pass


WHISPER_LANG = {
    "English": "english", "Hindi": "hindi", "Kannada": "kannada", "Tamil": "tamil",
    "Telugu": "telugu", "Malayalam": "malayalam", "Marathi": "marathi", "Bengali": "bengali",
    "Gujarati": "gujarati", "Punjabi": "punjabi", "Odia": "odia", "Assamese": "assamese",
    "Nepali": "nepali", "Sanskrit": "sanskrit", "Urdu": "urdu",
}


class SraVaaniTranscriber:
    """SraVaani ASR with chunk-level LID and timestamp fallback."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        print(f"Loading SraVaani model on {self.device}...")
        self.model = AutoModel.from_pretrained(SRAVAANI_MODEL, **kwargs).to(self.device).eval()
        print("SraVaani model loaded successfully.")

        print("Loading Vaani language identifier...")
        self.lid = VaaniLanguageIdentifier()
        print("Vaani language identifier loaded successfully.")

        self.whisper = None
        if WHISPER_ENABLED:
            print(f"Loading timestamp ASR fallback: {TIMESTAMP_ASR_MODEL}...")
            self.whisper = pipeline(
                "automatic-speech-recognition",
                model=TIMESTAMP_ASR_MODEL,
                device=0 if torch.cuda.is_available() else -1,
                token=HF_TOKEN,
            )
            print("Timestamp ASR fallback loaded successfully.")

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

    def _sravaani_text(self, path: str, language_label: str | None):
        signature = inspect.signature(self.model.transcribe)
        kwargs = {"return_hypotheses": True}
        if language_label and "language" in signature.parameters:
            kwargs["language"] = language_label

        try:
            hypotheses = self.model.transcribe(path, **kwargs)
        except Exception as exc:
            print(f"SraVaani inference failed for chunk: {exc}")
            return None, None

        print("========== SRAVAANI RAW OUTPUT ==========")
        print(f"Output type: {type(hypotheses)}")
        print(f"Raw output:\n{hypotheses}")
        print("===========================================")

        if not hypotheses:
            return None, None

        texts = []
        timestamp = None
        for h in hypotheses:
            text = self._pick(h, "text", "transcript", "sentence")
            if text and str(text).strip():
                texts.append(str(text).strip())
            ts = self._pick(h, "timestamp", "timestamps")
            if ts is not None:
                timestamp = ts

        return (" ".join(texts).strip() or None), timestamp

    def _whisper_segments(self, path: str, offset: float, language_label: str | None):
        if self.whisper is None:
            return []

        generate_kwargs = {"task": "transcribe"}
        whisper_lang = WHISPER_LANG.get(language_label or "")
        if whisper_lang:
            generate_kwargs["language"] = whisper_lang

        try:
            result = self.whisper(path, return_timestamps="segment", generate_kwargs=generate_kwargs)
        except Exception as exc:
            print(f"Timestamp ASR fallback failed: {exc}")
            return []

        segments = []
        for item in result.get("chunks", []):
            text = str(item.get("text", "")).strip()
            ts = item.get("timestamp")
            if not text or not ts or ts[0] is None or ts[1] is None:
                continue
            segments.append({
                "start": offset + float(ts[0]),
                "end": offset + float(ts[1]),
                "text": text,
                "language": language_label,
                "asr": "whisper-fallback",
            })
        return segments

    def transcribe(self, audio_path: str, language: str | None = None):
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr <= 0 or len(audio) == 0:
            raise TimestampUnavailableError("The normalized audio contains no samples.")

        chunk_size = max(1, int(CHUNK_SECONDS * sr))
        total = len(audio)
        segments = []
        fixed_label = None
        if language and language != "auto":
            fixed_label = get_language(language)["label"]

        print(f"Splitting audio into {int(np.ceil(total / chunk_size))} chunks...")

        for index, start_sample in enumerate(range(0, total, chunk_size), start=1):
            end_sample = min(start_sample + chunk_size, total)
            offset = start_sample / sr
            chunk_path = self._save_chunk(audio[start_sample:end_sample], sr)
            try:
                print("\n" + "=" * 50)
                print(f"PROCESSING CHUNK {index}")
                print(f"Time offset: {offset:.2f} seconds")

                language_label = fixed_label
                if language == "auto" or not language:
                    try:
                        language_label, confidence = self.lid.identify(chunk_path)
                        print(f"Detected language: {language_label} ({confidence:.3f})")
                    except Exception as exc:
                        print(f"Language identification failed: {exc}")

                sravaani_text, sravaani_timestamp = self._sravaani_text(chunk_path, language_label)

                if sravaani_text and sravaani_timestamp is not None:
                    ts = sravaani_timestamp
                    if isinstance(ts, (list, tuple)) and len(ts) == 2:
                        segments.append({
                            "start": offset + float(ts[0]),
                            "end": offset + float(ts[1]),
                            "text": sravaani_text,
                            "language": language_label,
                            "asr": "sravaani",
                        })
                        continue

                fallback = self._whisper_segments(chunk_path, offset, language_label)
                if fallback:
                    segments.extend(fallback)
                elif sravaani_text:
                    raise TimestampUnavailableError(
                        "SraVaani produced text, but no usable timestamps were returned and the timestamp fallback failed."
                    )
                else:
                    print("No speech recognized in this chunk.")
            finally:
                Path(chunk_path).unlink(missing_ok=True)

        if not segments:
            raise TimestampUnavailableError(
                "No speech segments were recognized. Check the audio, language, model access, and media quality."
            )

        segments.sort(key=lambda item: (item["start"], item["end"]))
        return segments

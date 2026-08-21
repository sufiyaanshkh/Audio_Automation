import inspect
import torch
from transformers import AutoModel

from config import HF_TOKEN, SRAVAANI_MODEL


class TimestampUnavailableError(RuntimeError):
    pass


class SraVaaniTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        self.model = AutoModel.from_pretrained(SRAVAANI_MODEL, **kwargs).to(self.device)
        self.model.eval()

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

    def _normalize_hypotheses(self, hypotheses):
        segments = []
        for hypothesis in hypotheses:
            text = self._pick(hypothesis, "text", "transcript", "sentence")
            start = self._pick(hypothesis, "start", "start_time", "start_sec", "begin")
            end = self._pick(hypothesis, "end", "end_time", "end_sec", "stop")
            if text is None:
                continue
            if start is None or end is None:
                raise TimestampUnavailableError(
                    "SraVaani returned text but no usable start/end timestamps. "
                    "This model/backend configuration cannot safely generate an SRT file yet."
                )
            segments.append({"start": float(start), "end": float(end), "text": str(text).strip()})
        if not segments:
            raise TimestampUnavailableError("No timestamped transcription segments were returned.")
        return segments

    def transcribe(self, audio_path: str, language: str | None = None):
        signature = inspect.signature(self.model.transcribe)
        kwargs = {"return_hypotheses": True}
        if language and "language" in signature.parameters:
            kwargs["language"] = language
        hypotheses = self.model.transcribe(audio_path, **kwargs)
        return self._normalize_hypotheses(hypotheses)

import torch
from transformers import pipeline

from config import HF_TOKEN, LID_MIN_CONFIDENCE, LID_MODEL


class VaaniLanguageIdentifier:
    """Identify the dominant spoken language in an audio chunk."""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        self.pipe = pipeline(
            "audio-classification",
            model=LID_MODEL,
            device=self.device,
            **kwargs,
        )

    def identify(self, audio_path: str):
        results = self.pipe(audio_path, top_k=5)
        if not results:
            return None, 0.0
        best = results[0]
        label = str(best["label"])
        score = float(best["score"])
        if score < LID_MIN_CONFIDENCE:
            return None, score
        return label, score

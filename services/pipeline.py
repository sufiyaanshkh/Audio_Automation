from pathlib import Path

from models.transcriber import SraVaaniTranscriber
from models.translator import IndicTranslator
from services.media import normalize_to_wav
from utils.languages import get_language, language_from_label


class SubtitlePipeline:
    def __init__(self):
        self.transcriber = SraVaaniTranscriber()
        self.translator = IndicTranslator()

    def process(
        self,
        input_path: str | Path,
        work_wav: str | Path,
        task: str,
        source_name: str,
        target_name: str | None = None,
    ):
        normalized = normalize_to_wav(input_path, work_wav)
        segments = self.transcriber.transcribe(str(normalized), language=source_name)

        if task == "transcribe":
            return segments
        if task != "translate":
            raise ValueError("Task must be 'transcribe' or 'translate'.")
        if not target_name:
            raise ValueError("A target language is required for translation.")

        target = get_language(target_name)
        translated = []

        for segment in segments:
            detected = language_from_label(segment.get("language"))
            if detected is None:
                detected = get_language(source_name) if source_name != "auto" else None
            if detected is None:
                # Keep an unknown-language segment instead of inventing a route.
                translated.append({**segment, "translation_error": "Language could not be identified."})
                continue

            text = self.translator.translate_batch(
                [segment["text"]],
                detected,
                target,
            )[0]
            translated.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
                "language": detected["label"],
                "asr": segment.get("asr"),
            })

        return translated

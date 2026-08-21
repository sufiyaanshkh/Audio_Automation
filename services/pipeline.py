from pathlib import Path

from models.transcriber import SraVaaniTranscriber
from models.translator import IndicTranslator
from services.media import normalize_to_wav
from utils.languages import get_language


class SubtitlePipeline:
    def __init__(self):
        self.transcriber = SraVaaniTranscriber()
        self.translator = IndicTranslator()

    def process(self, input_path: str | Path, work_wav: str | Path, task: str, source_name: str, target_name: str | None = None):
        source = get_language(source_name)
        normalized = normalize_to_wav(input_path, work_wav)
        segments = self.transcriber.transcribe(str(normalized), language=source_name)

        if task == "transcribe":
            return segments
        if task != "translate":
            raise ValueError("Task must be 'transcribe' or 'translate'.")
        if not target_name:
            raise ValueError("A target language is required for translation.")

        target = get_language(target_name)
        translated = self.translator.translate_batch([segment["text"] for segment in segments], source, target)
        return [
            {"start": segment["start"], "end": segment["end"], "text": text}
            for segment, text in zip(segments, translated)
        ]

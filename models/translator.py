import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import HF_TOKEN, INDIC_EN_MODEL, EN_INDIC_MODEL, INDIC_INDIC_MODEL
from utils.languages import translation_direction


class IndicTranslator:
    """IndicTrans2 translation router without IndicTransToolkit.

    The tokenizer/model are loaded lazily. Language direction is selected by
    utils.languages.translation_direction().
    """

    MODELS = {
        "indic_en": INDIC_EN_MODEL,
        "en_indic": EN_INDIC_MODEL,
        "indic_indic": INDIC_INDIC_MODEL,
    }

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache = {}

    def _load(self, direction: str):
        if direction in self.cache:
            return self.cache[direction]

        model_name = self.MODELS[direction]
        kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **kwargs
        ).to(self.device)
        model.eval()

        self.cache[direction] = (tokenizer, model)
        return self.cache[direction]

    @staticmethod
    def _prepare(texts: list[str], source_code: str, target_code: str) -> list[str]:
        """Add explicit language routing tags expected by multilingual models.

        This keeps the application free from the IndicTransToolkit dependency.
        """
        return [f"{source_code} {target_code} {text.strip()}" for text in texts]

    @staticmethod
    def _clean(decoded: list[str]) -> list[str]:
        return [text.strip() for text in decoded]

    def translate_batch(
        self, texts: list[str], source: dict, target: dict
    ) -> list[str]:
        direction = translation_direction(source, target)

        if direction == "identity":
            return texts

        if not texts:
            return []

        tokenizer, model = self._load(direction)

        prepared = self._prepare(
            texts,
            source["code"],
            target["code"],
        )

        inputs = tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                num_beams=5,
                max_length=512,
                use_cache=True,
            )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        return self._clean(decoded)

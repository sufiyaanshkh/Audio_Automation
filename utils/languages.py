LANGUAGES = {
    "english": {"label": "English", "code": "eng_Latn", "group": "english"},
    "assamese": {"label": "Assamese", "code": "asm_Beng", "group": "indic"},
    "bengali": {"label": "Bengali", "code": "ben_Beng", "group": "indic"},
    "bodo": {"label": "Bodo", "code": "brx_Deva", "group": "indic"},
    "dogri": {"label": "Dogri", "code": "doi_Deva", "group": "indic"},
    "gujarati": {"label": "Gujarati", "code": "guj_Gujr", "group": "indic"},
    "hindi": {"label": "Hindi", "code": "hin_Deva", "group": "indic"},
    "kannada": {"label": "Kannada", "code": "kan_Knda", "group": "indic"},
    "kashmiri_arabic": {"label": "Kashmiri (Arabic)", "code": "kas_Arab", "group": "indic"},
    "kashmiri_devanagari": {"label": "Kashmiri (Devanagari)", "code": "kas_Deva", "group": "indic"},
    "konkani": {"label": "Konkani", "code": "gom_Deva", "group": "indic"},
    "maithili": {"label": "Maithili", "code": "mai_Deva", "group": "indic"},
    "malayalam": {"label": "Malayalam", "code": "mal_Mlym", "group": "indic"},
    "manipuri_bengali": {"label": "Manipuri (Bengali)", "code": "mni_Beng", "group": "indic"},
    "manipuri_meitei": {"label": "Manipuri (Meitei)", "code": "mni_Mtei", "group": "indic"},
    "marathi": {"label": "Marathi", "code": "mar_Deva", "group": "indic"},
    "nepali": {"label": "Nepali", "code": "npi_Deva", "group": "indic"},
    "odia": {"label": "Odia", "code": "ory_Orya", "group": "indic"},
    "punjabi": {"label": "Punjabi", "code": "pan_Guru", "group": "indic"},
    "sanskrit": {"label": "Sanskrit", "code": "san_Deva", "group": "indic"},
    "santali": {"label": "Santali", "code": "sat_Olck", "group": "indic"},
    "sindhi_arabic": {"label": "Sindhi (Arabic)", "code": "snd_Arab", "group": "indic"},
    "sindhi_devanagari": {"label": "Sindhi (Devanagari)", "code": "snd_Deva", "group": "indic"},
    "tamil": {"label": "Tamil", "code": "tam_Taml", "group": "indic"},
    "telugu": {"label": "Telugu", "code": "tel_Telu", "group": "indic"},
    "urdu": {"label": "Urdu", "code": "urd_Arab", "group": "indic"},
}


def get_language(name: str):
    key = (name or "").strip().lower()
    if key not in LANGUAGES:
        raise ValueError(f"Unsupported language: {name}")
    return LANGUAGES[key]


def translation_direction(source: dict, target: dict) -> str:
    if source["group"] == "indic" and target["group"] == "english":
        return "indic_en"
    if source["group"] == "english" and target["group"] == "indic":
        return "en_indic"
    if source["group"] == "indic" and target["group"] == "indic":
        return "indic_indic"
    if source["group"] == target["group"] == "english":
        return "identity"
    raise ValueError("This translation direction is not supported by the configured IndicTrans2 models.")

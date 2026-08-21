# Audio Automation

Universal audio/video subtitle generation pipeline.

## Goal

Upload an audio/video file and generate an `.srt` subtitle file with timestamps. The first implementation supports:

- Speech transcription with SraVaani 1.0
- Optional translation with AI4Bharat IndicTrans2
- Automatic translation-direction routing for Indic/English combinations
- SRT generation
- Flask web interface
- FFmpeg-based audio extraction for common media formats

## Current translation coverage

The application routes between:

- Indic → English: `ai4bharat/indictrans2-indic-en-1B`
- English → Indic: `ai4bharat/indictrans2-en-indic-1B`
- Indic → Indic: `ai4bharat/indictrans2-indic-indic-1B`

The ASR layer is deliberately isolated so a timestamp-capable ASR backend can be swapped in without changing the UI or translation layer.

## Prerequisites

- Python 3.10+
- FFmpeg installed and available on PATH
- Hugging Face account/token with access to the required gated model repositories
- GPU recommended for practical inference; CPU is supported but slower

## Setup

```powershell
git clone https://github.com/sufiyaanshkh/Audio_Automation.git
cd Audio_Automation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `HF_TOKEN`.

Then run:

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

## Important timestamp behavior

SRT generation requires reliable segment start/end times. The service does **not fabricate timestamps**. If the configured transcription backend does not expose timestamps, the API returns a clear error explaining that a timestamp-capable backend/alignment step is required.

This is intentional: inaccurate subtitle timings are worse than refusing to produce an apparently valid SRT file.

## Supported input

- MP3
- WAV
- M4A
- MP4
- MOV
- WebM
- OGG
- FLAC

Video/audio inputs are converted to a mono 16 kHz WAV working file using FFmpeg before ASR.

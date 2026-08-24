# Audio Automation

Universal AI audio/video subtitle generator for multilingual and mixed-language recordings.

## What it does

1. Upload audio or video.
2. Normalize the media to mono 16 kHz WAV with FFmpeg.
3. Detect the dominant spoken language per 15-second chunk with `ARTPARK-IISc/Vaani-LID_v0` when Source is `Auto / Mixed`.
4. Use `ARTPARK-IISc/SraVaani-1.0` as the primary speech recognizer.
5. If SraVaani returns text without usable segment timestamps, use `openai/whisper-small` as a timestamp-capable fallback for that chunk.
6. Preserve absolute timestamps across chunks.
7. For translation, route each detected-language segment through the appropriate IndicTrans2 direction.
8. Generate a downloadable `.srt` file.

The application never invents timestamps.

## Models

### Speech recognition

- SraVaani 1.0: multilingual Indic ASR covering 65 Indian languages/dialects. The model is gated on Hugging Face, so the account used by `HF_TOKEN` must have accepted the model access conditions. See the model card: https://huggingface.co/ARTPARK-IISc/SraVaani-1.0

### Language identification

- Vaani-LID_v0: open MIT-licensed spoken-language classifier covering 42 Indian languages plus English. It is used for chunk-level routing in Auto / Mixed mode. See: https://huggingface.co/ARTPARK-IISc/Vaani-LID_v0

### Timestamp fallback

- `openai/whisper-small`: used only when SraVaani does not expose usable timestamps for a chunk. This is an implementation fallback for subtitle timing, not a replacement for the SraVaani integration.

### Translation

- Indic → English: `ai4bharat/indictrans2-indic-en-1B`
- English → Indic: `ai4bharat/indictrans2-en-indic-1B`
- Indic → Indic: `ai4bharat/indictrans2-indic-indic-1B`

The translation implementation uses the IndicTrans2 language-tag input format directly and does **not** require `IndicTransToolkit`, avoiding the Windows C++ compilation problem encountered previously.

## Important: Hugging Face access

SraVaani is a gated model. You must accept its access conditions on Hugging Face and create a read token. Put that token in `.env` as `HF_TOKEN`.

The Vaani-LID model is not gated.

## Prerequisites

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended
- FFmpeg installed and available on PATH
- Hugging Face account and read token with SraVaani access
- GPU recommended; CPU works but will be significantly slower

## Clean Windows setup

```powershell
git clone https://github.com/sufiyaanshkh/Audio_Automation.git
cd Audio_Automation

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

copy .env.example .env
```

Edit `.env`:

```env
HF_TOKEN=hf_your_real_token
```

Verify FFmpeg:

```powershell
ffmpeg -version
```

Run:

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

## Recommended first test

Use a short 20–60 second recording first. Select:

- Task: `Translate and generate SRT`
- Source: `Auto / Mixed languages`
- Target: `English`

Once that works, test the full recording.

## Output

Example:

```srt
1
00:00:00,000 --> 00:00:03,420
Hello everyone.

2
00:00:03,420 --> 00:00:07,810
Today we are going to discuss the project.
```

## Architecture

```text
Audio / Video
      |
      v
FFmpeg normalization
      |
      v
15-second chunks
      |
      +--> Vaani-LID_v0 --> detected language
      |
      v
SraVaani ASR
      |
      +--> usable timestamps --> keep SraVaani result
      |
      +--> no timestamps --> Whisper timestamp fallback
      |
      v
Absolute timestamped segments
      |
      +--> Transcribe --> SRT
      |
      +--> Translate
                |
                +--> Indic -> English
                +--> English -> Indic
                +--> Indic -> Indic
                |
                v
               SRT
```

## Notes

- Auto / Mixed language detection is chunk-level. A 15-second chunk containing several languages is assigned its dominant detected language for routing. It is not claimed to provide word-level language identification.
- SraVaani's current public Transformers example documents `model.transcribe(..., return_hypotheses=True)`, but its basic example does not promise a stable segment timestamp schema. The fallback exists specifically because subtitle generation requires actual timings.
- Do not commit `.env`, model caches, uploads, work files, or generated outputs.

import subprocess
from pathlib import Path


def normalize_to_wav(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert audio/video to mono 16 kHz PCM WAV for ASR."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg and add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg could not process this media file: {exc.stderr[-1000:]}") from exc
    return output_path

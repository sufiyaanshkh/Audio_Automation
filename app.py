import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, DEBUG, HOST, MAX_CONTENT_LENGTH, OUTPUT_DIR, PORT, UPLOAD_DIR, WORK_DIR
from models.transcriber import TimestampUnavailableError
from services.pipeline import SubtitlePipeline
from services.srt import write_srt
from utils.languages import LANGUAGES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
pipeline = None


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = SubtitlePipeline()
    return pipeline


@app.get("/")
def index():
    return render_template("index.html", languages=LANGUAGES)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/process")
def process():
    media = request.files.get("media")
    task = request.form.get("task", "transcribe").strip().lower()
    source = request.form.get("source_language", "").strip().lower()
    target = request.form.get("target_language", "").strip().lower() or None

    if media is None or not media.filename:
        return jsonify({"error": "Please upload an audio or video file."}), 400
    if not allowed(media.filename):
        return jsonify({"error": "Unsupported media format."}), 400
    if source not in LANGUAGES:
        return jsonify({"error": "Please select a supported source language."}), 400

    job_id = uuid.uuid4().hex
    original_name = secure_filename(media.filename)
    upload_path = UPLOAD_DIR / f"{job_id}_{original_name}"
    wav_path = WORK_DIR / f"{job_id}.wav"
    srt_name = f"{job_id}.srt"
    srt_path = OUTPUT_DIR / srt_name

    try:
        media.save(upload_path)
        segments = get_pipeline().process(upload_path, wav_path, task, source, target)
        write_srt(segments, srt_path)
        return jsonify({
            "success": True,
            "segments": segments,
            "download_url": f"/download/{srt_name}",
            "filename": srt_name,
        })
    except TimestampUnavailableError as exc:
        return jsonify({"error": str(exc), "code": "TIMESTAMPS_UNAVAILABLE"}), 422
    except Exception as exc:
        app.logger.exception("Processing failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)
        if wav_path.exists():
            wav_path.unlink(missing_ok=True)


@app.get("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)

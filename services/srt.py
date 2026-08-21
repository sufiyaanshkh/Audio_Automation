from pathlib import Path


def _format_timestamp(value: float) -> str:
    total_ms = max(0, int(round(float(value) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def validate_segments(segments: list[dict]) -> None:
    if not segments:
        raise ValueError("No subtitle segments were produced.")
    for index, segment in enumerate(segments, start=1):
        if "start" not in segment or "end" not in segment:
            raise ValueError(f"Segment {index} has no timestamp data.")
        if float(segment["end"]) <= float(segment["start"]):
            raise ValueError(f"Segment {index} has an invalid time range.")
        if not str(segment.get("text", "")).strip():
            raise ValueError(f"Segment {index} has no text.")


def write_srt(segments: list[dict], output_path: str | Path) -> Path:
    validate_segments(segments)
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for number, segment in enumerate(segments, start=1):
            handle.write(f"{number}\n")
            handle.write(f"{_format_timestamp(segment['start'])} --> {_format_timestamp(segment['end'])}\n")
            handle.write(f"{segment['text'].strip()}\n\n")
    return output_path

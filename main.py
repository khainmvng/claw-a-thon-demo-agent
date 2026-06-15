import asyncio
import base64
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue

from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from greennode_agentbase import (
    GreenNodeAgentBaseApp,
    RequestContext,
    PingStatus,
)

load_dotenv()

app = GreenNodeAgentBaseApp()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

# Models are downloaded once then cached. Force offline so pyannote never makes
# a blocking network call (which can hang the pipeline load). Set
# HF_FORCE_ONLINE=1 in .env if you ever need to re-download.
if not os.environ.get("HF_FORCE_ONLINE"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

_whisper_model = None
_diarization_pipeline = None
_executor = ThreadPoolExecutor(max_workers=1)


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None and HF_TOKEN:
        from pyannote.audio import Pipeline
        import torch
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN,
        )
        _diarization_pipeline.to(torch.device("cpu"))
    return _diarization_pipeline


# Languages written in Latin script. When Whisper detects one of these, any
# Hangul / CJK / Kana / Cyrillic / Arabic character in the output is a
# cross-language hallucination and is safe to drop.
_LATIN_LANGS = {
    "vi", "en", "fr", "es", "de", "it", "pt", "nl", "id", "ms",
    "tr", "pl", "sv", "da", "no", "fi", "cs", "ro", "hu", "ca",
}

# Unicode ranges of "foreign" scripts to strip for Latin-script languages.
_FOREIGN_RANGES = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0x0400, 0x04FF),  # Cyrillic
    (0x0500, 0x052F),  # Cyrillic Supplement
    (0x0600, 0x06FF),  # Arabic
)


def _is_foreign(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _FOREIGN_RANGES)


def clean_text(text: str, lang: str) -> str:
    """Remove garbage characters produced by Whisper.

    1. U+FFFD replacement chars come from incomplete multi-byte tokens — always
       garbage, dropped for every language.
    2. For Latin-script languages, stray foreign-script characters (e.g. Korean
       러, Cyrillic і/й) are hallucinations and are removed too.
    """
    text = text.replace("�", "")

    if lang in _LATIN_LANGS:
        text = "".join(ch for ch in text if not _is_foreign(ch))

    # Collapse spaces left behind by removed characters.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


def get_speaker_for_segment(annotation, start: float, end: float) -> str:
    best_overlap = 0.0
    best_speaker = "SPEAKER_00"
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def speaker_label(raw: str) -> str:
    num = raw.replace("SPEAKER_", "")
    try:
        return f"Speaker {int(num) + 1}"
    except ValueError:
        return raw


def process_audio(audio_path: str, progress: Queue) -> tuple[str, float, int]:
    def emit(pct: int, msg: str):
        progress.put({"pct": pct, "msg": msg})

    try:
        emit(5, "Đang tải Whisper model...")
        model = get_whisper_model()

        emit(15, "Đang nhận dạng giọng nói (Whisper)...")
        result = model.transcribe(audio_path, task="transcribe")
        segments = result["segments"]
        detected_lang = result.get("language", "unknown")
        duration = segments[-1]["end"] if segments else 0.0

        emit(50, f"Whisper xong — {len(segments)} đoạn, ngôn ngữ: {detected_lang}")

        pipeline = get_diarization_pipeline()

        lines: list[str] = [
            "# Speech-to-Text Transcript",
            f"# Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Whisper     : {WHISPER_MODEL_SIZE}",
            f"# Language    : {detected_lang}",
            f"# Diarization : {'enabled' if pipeline else 'disabled (set HUGGINGFACE_TOKEN to enable)'}",
            "",
        ]

        if pipeline:
            emit(55, "Đang phân tách người nói (diarization)...")

            # Map pyannote's internal sub-steps into the 55–83% band so the bar
            # keeps moving during the slow CPU inference instead of freezing.
            step_labels = {
                "segmentation": "Đang phân đoạn âm thanh...",
                "speaker_counting": "Đang đếm số người nói...",
                "embeddings": "Đang trích đặc trưng giọng nói...",
                "discrete_diarization": "Đang gán nhãn người nói...",
            }

            def diar_hook(step_name, step_artifact, file=None, total=None, completed=None):
                label = step_labels.get(step_name, f"Đang xử lý: {step_name}")
                if total and completed is not None and total > 0:
                    frac = min(completed / total, 1.0)
                    pct = 55 + int(frac * 28)  # 55 → 83
                    emit(pct, f"{label} ({completed}/{total})")
                else:
                    emit(56, label)

            output = pipeline(audio_path, hook=diar_hook)
            annotation = output.speaker_diarization

            emit(85, "Đang ghép speaker với transcript...")
            current_speaker = None
            for seg in segments:
                start, end = seg["start"], seg["end"]
                text = clean_text(seg["text"], detected_lang)
                if not text:
                    continue
                spk = speaker_label(get_speaker_for_segment(annotation, start, end))
                if spk != current_speaker:
                    if current_speaker is not None:
                        lines.append("")
                    lines.append(f"[{spk}]  {format_ts(start)}")
                    current_speaker = spk
                lines.append(text)
        else:
            for seg in segments:
                text = clean_text(seg["text"], detected_lang)
                if text:
                    lines.append(f"[{format_ts(seg['start'])} → {format_ts(seg['end'])}]  {text}")

        emit(95, "Đang hoàn thiện transcript...")
        transcript = "\n".join(lines)
        emit(100, "Hoàn thành!")
        return transcript, duration, len(segments)

    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stdout)
        raise


async def handle_transcribe_stream(request: Request) -> Response:
    tmp_path = None

    try:
        form = await request.form()
        audio_file = form.get("audio")
        if not audio_file:
            return JSONResponse({"error": "No audio file provided"}, status_code=400)

        content = await audio_file.read()
        if not content:
            return JSONResponse({"error": "Empty file"}, status_code=400)

        suffix = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
        stem = Path(audio_file.filename).stem if audio_file.filename else "transcript"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    progress: Queue = Queue()
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(_executor, process_audio, tmp_path, progress)

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        try:
            yield sse({"type": "progress", "pct": 0, "msg": "Đang upload file..."})

            while True:
                # Drain all queued progress messages
                while not progress.empty():
                    item = progress.get_nowait()
                    yield sse({"type": "progress", "pct": item["pct"], "msg": item["msg"]})

                if future.done():
                    # Drain any remaining messages
                    while not progress.empty():
                        item = progress.get_nowait()
                        yield sse({"type": "progress", "pct": item["pct"], "msg": item["msg"]})
                    break

                await asyncio.sleep(0.3)

            transcript, duration, seg_count = future.result()

            yield sse({
                "type": "done",
                "filename": f"{stem}.txt",
                "transcript": transcript,
                "duration_seconds": round(duration, 2),
                "segments": seg_count,
                "model": WHISPER_MODEL_SIZE,
                "diarization": bool(HF_TOKEN),
            })

        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stdout)
            yield sse({"type": "error", "message": str(exc)})

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.entrypoint
def handler(payload: dict, context: RequestContext) -> dict:
    audio_b64 = payload.get("audio_b64", "")
    filename = payload.get("filename", "audio.wav")

    if not audio_b64:
        return {"status": "error", "message": "Field 'audio_b64' (base64-encoded audio) is required."}

    try:
        content = base64.b64decode(audio_b64)
    except Exception:
        return {"status": "error", "message": "Invalid base64 data."}

    suffix = Path(filename).suffix or ".wav"
    stem = Path(filename).stem or "transcript"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    progress: Queue = Queue()
    try:
        transcript, duration, seg_count = process_audio(tmp_path, progress)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "status": "success",
        "filename": f"{stem}.txt",
        "transcript": transcript,
        "duration_seconds": round(duration, 2),
        "segments": seg_count,
        "model": WHISPER_MODEL_SIZE,
        "diarization": bool(HF_TOKEN),
    }


async def serve_ui(request: Request) -> HTMLResponse:
    with open(Path(__file__).parent / "index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


app.add_route("/", serve_ui, methods=["GET"])
app.add_route("/transcribe", handle_transcribe_stream, methods=["POST"])


@app.ping
def health_check() -> PingStatus:
    return PingStatus.HEALTHY


def preload_models():
    """Eagerly load Whisper + pyannote on the main thread BEFORE serving.

    Loading pyannote inside a worker thread on first request can hang
    indefinitely (torch/network init off the main thread). Pre-loading here
    guarantees the first request is fast and never stalls at diarization.
    """
    print("[startup] Preloading Whisper model...", flush=True)
    get_whisper_model()
    print("[startup] Whisper ready.", flush=True)

    if HF_TOKEN:
        print("[startup] Preloading pyannote diarization pipeline...", flush=True)
        get_diarization_pipeline()
        print("[startup] Diarization ready.", flush=True)
    else:
        print("[startup] No HUGGINGFACE_TOKEN — diarization disabled.", flush=True)

    print("[startup] All models loaded. Server is ready.", flush=True)


if __name__ == "__main__":
    preload_models()
    app.run(port=8080, host="0.0.0.0")

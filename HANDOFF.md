# HANDOFF — Speech-to-Text Agent (Whisper + Speaker Diarization)

> Cập nhật: 2026-06-15. Tài liệu này bàn giao cho developer tiếp nhận dự án.

## 1. Agent này làm gì

AI agent **Speech-to-Text** chạy trên GreenNode AgentBase:

- Upload file audio (MP3/WAV/M4A/FLAC/OGG/MP4/WEBM) qua web UI hoặc JSON API
- Dùng **Whisper** (openai-whisper) để transcribe
- Dùng **pyannote.audio** để **speaker diarization** (phân tách & gán nhãn người nói)
- Trả về transcript dạng `.txt`, có timestamp + label `[Speaker N]`
- Stream tiến trình realtime về UI qua **Server-Sent Events (SSE)**

## 2. Cấu trúc & các file chính

| File | Vai trò |
|---|---|
| `main.py` | Toàn bộ backend: load model, transcribe, diarize, SSE, routes |
| `index.html` | Web UI (drag-drop upload, progress bar, preview, download .txt) |
| `requirements.txt` | `openai-whisper`, `pyannote.audio>=3.1.0`, `python-multipart`, `greennode-agentbase`, `python-dotenv` |
| `Dockerfile` | python:3.12-slim + `ffmpeg` + torch CPU-only |
| `.env` | `WHISPER_MODEL`, `HUGGINGFACE_TOKEN` (KHÔNG commit) |
| `.env.example` | Template biến môi trường |
| `0614_1.txt` | File transcript mẫu (đã được fix lỗi ký tự — xem mục 6) |

## 3. Kiến trúc code (`main.py`)

```
preload_models()          # chạy ở __main__ TRƯỚC khi serve — load Whisper + pyannote ở main thread
  ├─ get_whisper_model()       # lazy singleton, whisper.load_model(WHISPER_MODEL_SIZE)
  └─ get_diarization_pipeline()# lazy singleton, pyannote speaker-diarization-3.1

Routes:
  GET  /            → serve_ui()              # trả index.html
  POST /transcribe → handle_transcribe_stream # multipart upload, trả SSE stream
  POST /invocations→ handler() (@app.entrypoint) # JSON API, base64 audio (đồng bộ, không stream)

process_audio(audio_path, progress: Queue)   # CHẠY TRONG THREAD (ThreadPoolExecutor, max_workers=1)
  ├─ Whisper transcribe → segments
  ├─ pyannote pipeline(audio, hook=diar_hook) → annotation.speaker_diarization
  ├─ clean_text() từng segment  # lọc ký tự rác
  └─ ghép [Speaker N] + timestamp → transcript string
```

### Các quyết định kỹ thuật quan trọng (đừng vô tình revert)

1. **`run_in_executor` + `ThreadPoolExecutor(max_workers=1)`** — Whisper/pyannote là blocking CPU work. PHẢI chạy ngoài event loop, nếu không sẽ block toàn bộ server và connection bị drop âm thầm (không báo lỗi).

2. **`preload_models()` ở main thread trước `app.run()`** — pyannote nếu load lần đầu trong worker thread sẽ **treo vô hạn**. Preload ở startup giải quyết triệt để (xem mục 6, lỗi "stuck 55%").

3. **`HF_HUB_OFFLINE=1`** (set ở `main.py:40-41`) — models đã cache → tránh pyannote gọi network bị treo. Set `HF_FORCE_ONLINE=1` trong `.env` nếu cần tải lại model.

4. **pyannote 4.x API** — KHÁC pyannote 3.x:
   - `Pipeline.from_pretrained(..., token=HF_TOKEN)` (KHÔNG phải `use_auth_token`)
   - `output = pipeline(audio)` rồi `output.speaker_diarization.itertracks(...)` (output KHÔNG có `.itertracks` trực tiếp)
   - Hỗ trợ `hook=` callback để báo progress

5. **SSE + `ensure_ascii=False`** (`main.py:251`) — để tiếng Việt không bị escape; response UTF-8.

6. **`clean_text(text, lang)`** — lọc ký tự rác do Whisper hallucinate (xem mục 6).

## 4. Setup & chạy local

```bash
cd /Users/lap15077/Documents/GitHub/claw-a-thon-demo-agent

# Dependencies (đã cài sẵn trong venv/)
source venv/bin/activate
pip install -r requirements.txt

# ffmpeg (BẮT BUỘC — Whisper cần để decode audio)
brew install ffmpeg     # macOS

# .env
WHISPER_MODEL=medium                          # tiny|base|small|medium|large
HUGGINGFACE_TOKEN=hf_xxx                       # bắt buộc cho diarization

# Chạy (preload model ~vài chục giây với medium)
python main.py
# → http://localhost:8080
```

### HuggingFace token — phải accept license 3 model (gated)

Token cần được cấp quyền truy cập **cả 3** repo (vào link, bấm "Agree and access repository"):
1. `huggingface.co/pyannote/segmentation-3.0`
2. `huggingface.co/pyannote/speaker-diarization-3.1`
3. `huggingface.co/pyannote/speaker-diarization-community-1`

Nếu thiếu → lỗi `403 Cannot access gated repo`. Nếu không set token → agent vẫn transcribe (có timestamp) nhưng KHÔNG có speaker label.

## 5. Output format

```
# Speech-to-Text Transcript
# Generated   : 2026-06-15 18:04:33
# Whisper     : medium
# Language    : vi
# Diarization : enabled

[Speaker 1]  00:01:00.00
nội dung người nói 1...

[Speaker 2]  00:01:23.00
nội dung người nói 2...
```

(Không diarization thì mỗi dòng có dạng `[hh:mm:ss.cc → hh:mm:ss.cc]  text`)

## 6. Lịch sử bug đã fix (đừng lặp lại)

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `No such file or directory: 'ffmpeg'` | ffmpeg chưa cài | `brew install ffmpeg` + đã thêm vào Dockerfile |
| `use_auth_token unexpected keyword` | pyannote 4.x đổi API | dùng `token=` |
| `DiarizeOutput has no attribute itertracks` | pyannote 4.x đổi return type | dùng `output.speaker_diarization.itertracks()` |
| `403 gated repo` | chưa accept license | accept 3 model trên HF (mục 4) |
| Treo ở 55%, không báo lỗi | (a) blocking event loop (b) pyannote load trong worker thread bị treo | (a) `run_in_executor` (b) `preload_models()` ở main thread + `HF_HUB_OFFLINE=1` |
| "Stuck 55%" nhưng thực ra đang chạy | diarization trên CPU rất chậm, progress bar đứng yên | thêm `diar_hook` map sub-step vào 55–83% |
| Ký tự lỗi `��`, `러`, `і`, `й` trong transcript | Whisper `base` hallucinate token đa ngôn ngữ + emit UTF-8 dở dang | `clean_text()`: xóa U+FFFD + strip ký tự lạc script khi lang là Latin |
| Tiếng Việt bị escape trong JSON | thiếu charset | `ensure_ascii=False` + `charset=utf-8` |

## 7. Việc đang dang dở / TODO

### 🔴 Đang làm: TĂNG TỐC TRANSCRIBE (chưa hoàn thành)

User yêu cầu tăng tốc. Đã phân tích nhưng **CHƯA implement**. Kết luận nghiên cứu:

- **GreenNode KHÔNG có GPU** — chỉ 2 flavor CPU: `runtime-s2-general-2x4` (2vCPU/4GB) và `runtime-s2-general-4x8` (4vCPU/8GB). Deploy lên GreenNode sẽ **chậm hơn** local (máy dev là M-series 8-core/16GB).
- **Giải pháp đề xuất: chuyển sang `faster-whisper` (CTranslate2)**
  - Nhanh ~4x trên CPU, RAM giảm từ ~3.7GB → ~1.5GB (lúc đó mới vừa flavor GreenNode nhỏ)
  - Độ chính xác `medium` giữ nguyên
  - Dùng `compute_type="int8"` + `vad_filter=True` (skip khoảng lặng)
  - **Cần sửa**: `get_whisper_model()` → `WhisperModel(size, device="cpu", compute_type="int8")`; `process_audio()` → segments là generator với attr `.start/.end/.text`, `info.language` (KHÁC dict hiện tại); thêm `faster-whisper` vào `requirements.txt`
  - Bonus: faster-whisper trả generator → có thể emit progress realtime cho bước transcribe (hiện 15→50% đang là khối kín)

> User chưa chốt hướng đi (faster-whisper vs chỉ tinh chỉnh tham số). Cần xác nhận trước khi code.

### 🟡 TODO khác

- [ ] Model `large` chưa test trên 16GB RAM (có thể thiếu RAM)
- [ ] Endpoint `/invocations` (JSON API) chạy đồng bộ, không stream progress — OK cho API call nhưng audio dài có thể timeout
- [ ] Chưa có giới hạn kích thước file upload
- [ ] `get_speaker_for_segment()` dùng O(n²) overlap matching — chậm với audio rất dài, có thể tối ưu

## 8. Deploy lên GreenNode (khi cần chia sẻ/chạy 24-7)

Dùng skill `/agentbase-deploy`. Lưu ý:
- Image phải build có `ffmpeg` (Dockerfile đã có)
- Model phải có trong image hoặc tải lúc startup — cân nhắc bake model vào image để tránh tải lại
- RAM: với `openai-whisper medium` cần flavor 4x8; nếu chuyển faster-whisper int8 thì 2x4 có thể đủ
- Set `HUGGINGFACE_TOKEN` qua secret/env của runtime

## 9. Git state

- Branch: `claude/wizardly-bardeen-yp95g8`
- Các file đã sửa (chưa commit): `main.py`, `index.html`, `requirements.txt`, `Dockerfile`, `.env.example`, `.dockerignore`
- File mới: `0614_1.txt` (transcript mẫu), `HANDOFF.md`
- **Các thay đổi Speech-to-Text chưa được commit** (commit gần nhất trong lịch sử git vẫn là bản khởi tạo trước đó)

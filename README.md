# speech-to-text-agent

A GreenNode AgentBase agent that transcribes audio to text with **speaker diarization** using **Whisper** + **pyannote.audio**.

Upload an audio file → get a `.txt` transcript with timestamps and `[Speaker N]` labels.

## Features

- Speech-to-text with [openai-whisper](https://github.com/openai/whisper)
- Speaker diarization (who spoke when) with [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- Web UI: drag-and-drop upload, realtime progress (SSE), transcript preview, download `.txt`
- JSON API endpoint for programmatic use
- Automatic cleanup of cross-language hallucination characters

## Prerequisites

- Python 3.10+
- `ffmpeg` (required by Whisper to decode audio): `brew install ffmpeg` (macOS) / `apt-get install ffmpeg` (Linux)
- A HuggingFace token with access to the pyannote gated models (see below) — required for speaker diarization
- A GreenNode IAM Service Account ([create one here](https://iam.console.vngcloud.vn/service-accounts)) — only for deployment

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env:
   #   WHISPER_MODEL=medium          # tiny | base | small | medium | large
   #   HUGGINGFACE_TOKEN=hf_xxx      # required for speaker diarization
   ```

### HuggingFace token (for speaker diarization)

The pyannote pipeline depends on 3 **gated** models. Log in to HuggingFace with the account that owns your token and accept the license on each (click "Agree and access repository"):

1. https://huggingface.co/pyannote/segmentation-3.0
2. https://huggingface.co/pyannote/speaker-diarization-3.1
3. https://huggingface.co/pyannote/speaker-diarization-community-1

Without a token the agent still transcribes (with timestamps) but won't label speakers.

## Run Locally

```bash
python3 main.py
```

The agent preloads the models, then starts on `http://0.0.0.0:8080`. Open it in a browser and upload an audio file.

### JSON API

```bash
curl -X POST http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"audio_b64": "<base64-encoded-audio>", "filename": "audio.mp3"}'
```

## Output format

```
# Speech-to-Text Transcript
# Generated   : 2026-06-15 18:04:33
# Whisper     : medium
# Language    : vi
# Diarization : enabled

[Speaker 1]  00:01:00.00
...

[Speaker 2]  00:01:23.00
...
```

## Deploy to AgentBase Runtime

1. Build and push the Docker image (the `Dockerfile` already installs `ffmpeg`) — or use the `/agentbase-deploy` skill
2. Create a Runtime and Endpoint at https://aiplatform.console.vngcloud.vn/agent-runtime
3. Set `WHISPER_MODEL` and `HUGGINGFACE_TOKEN` as runtime env vars

> Note: GreenNode runtimes are CPU-only. Whisper on CPU is slow; consider switching to `faster-whisper` (see `HANDOFF.md`) for ~4x speedup and lower memory.

## Project Structure

- `main.py` — backend: model loading, transcription, diarization, SSE streaming, routes
- `index.html` — web UI
- `Dockerfile` — container image (python:3.12-slim + ffmpeg + torch CPU)
- `requirements.txt` — Python dependencies
- `.env.example` — environment variable template
- `HANDOFF.md` — developer handoff notes (architecture, gotchas, TODO)

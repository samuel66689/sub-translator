import io
import re
import json
import asyncio
import requests
import edge_tts
from pydub import AudioSegment
from flask import Flask, request, jsonify, Response, send_file

app = Flask(__name__)
# 100MB Upload limit for video/audio
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
DEFAULT_MODEL = 'gemini-3.5-flash-lite'

# Reuse connections (TCP/TLS keep-alive) across requests
http = requests.Session()

_FENCE_START = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_END = re.compile(r"\s*```$")
_MODEL_CLEAN = re.compile(r'[^a-zA-Z0-9\-\.]')


def strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE_START.sub("", t)
        t = _FENCE_END.sub("", t)
    return t.strip()


def fmt_srt_time(sec: float) -> str:
    total_ms = max(0, int(round(sec * 1000)))
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def parse_srt_time_to_ms(time_str: str) -> int:
    try:
        hms, ms = time_str.replace('.', ',').split(',')
        h, m, s = map(int, hms.split(':'))
        return (h * 3600 + m * 60 + s) * 1000 + int(ms)
    except Exception:
        return 0


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "ဖိုင်အရွယ်အစား 100MB ထက် မကျော်ရပါ"}), 413


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    file = request.files.get('file')
    api_key = request.form.get('apiKey', '').strip()
    if not file or not api_key:
        return jsonify({"error": "Video/Audio transcribe လုပ်ရန် Groq API Key လိုအပ်ပါသည်"}), 400

    try:
        files = {'file': (file.filename, file.read(), file.content_type or 'application/octet-stream')}
        data = {'model': 'whisper-large-v3-turbo', 'response_format': 'verbose_json'}
        headers = {'Authorization': f'Bearer {api_key}'}

        res = http.post(GROQ_URL, headers=headers, files=files, data=data, timeout=180)

        if res.status_code != 200:
            try:
                err_msg = res.json().get('error', {}).get('message') or f'Groq Error ({res.status_code})'
            except Exception:
                err_msg = f'Groq Error ({res.status_code})'
            # Forward the real status (429/5xx/...) instead of always 400
            return jsonify({"error": err_msg}), res.status_code

        segments = res.json().get('segments') or []
        items = [
            {
                "id": idx,
                "startTime": fmt_srt_time(float(seg.get('start') or 0)),
                "endTime": fmt_srt_time(float(seg.get('end') or 0)),
                "originalText": (seg.get('text') or '').strip(),
                "translatedText": ""
            }
            for idx, seg in enumerate(segments, 1)
            if isinstance(seg, dict)
        ]
        return jsonify({"subtitles": items})
    except requests.exceptions.Timeout:
        return jsonify({"error": "Audio transcription timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

async def generate_edge_tts_audio(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data


# Only voices offered in the UI may be used (prevents abuse / weird failures)
_ALLOWED_VOICES = frozenset({'my-MM-ThihaNeural', 'my-MM-NilarNeural'})
_TTS_CONCURRENCY = 6      # parallel edge-tts requests
_TTS_TIMEOUT = 120        # seconds per TTS request
_TTS_CHUNK_LIMIT = 800    # chars per TTS request; long lines are split
_MAX_SPEEDUP = 1.35       # max TTS speed-up to fit a subtitle time slot


def chunk_text_for_tts(text: str, limit: int = _TTS_CHUNK_LIMIT):
    """Split long subtitle text into sentence-boundary chunks for TTS."""
    parts = [p for p in re.split(r'(?<=[။.!?!\n])\s+', text.strip()) if p]
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= limit:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            while len(p) > limit:  # hard-split a single overlong sentence
                chunks.append(p[:limit])
                p = p[limit:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks or [text]


@app.route('/api/health')
def health():
    return jsonify({"ok": True})


@app.route('/api/dubbing', methods=['POST'])
async def dubbing():
    req = request.get_json(silent=True) or {}
    subtitles = req.get('subtitles', [])
    voice = req.get('voice', 'my-MM-ThihaNeural')
    if voice not in _ALLOWED_VOICES:
        voice = 'my-MM-ThihaNeural'

    if not subtitles:
        return jsonify({"error": "Dubbing ပြုလုပ်ရန် စာတန်းထိုး မရှိပါ"}), 400

    try:
        # Sort subtitles by startTime to ensure chronological order just in case
        sorted_subs = sorted(subtitles, key=lambda x: parse_srt_time_to_ms(x.get('startTime') or '00:00:00,000'))

        # Base track must cover the latest end time (not just the last-by-start cue)
        total_duration_ms = max(
            parse_srt_time_to_ms(s.get('endTime') or '00:00:00,000') for s in sorted_subs
        )
        # 24kHz matches edge-tts output so pydub doesn't resample everything
        combined_audio = AudioSegment.silent(duration=total_duration_ms + 500, frame_rate=24000)

        # Flatten into (sub_index, chunk_index, text) jobs, then synthesize in parallel
        jobs = []
        for i, sub in enumerate(sorted_subs):
            text = (sub.get("translatedText") or sub.get("originalText") or "").strip()
            if not text:
                continue
            for j, piece in enumerate(chunk_text_for_tts(text)):
                jobs.append((i, j, piece))

        sem = asyncio.Semaphore(_TTS_CONCURRENCY)

        async def synth_job(text):
            async with sem:
                try:
                    return await asyncio.wait_for(
                        generate_edge_tts_audio(text, voice), timeout=_TTS_TIMEOUT
                    )
                except Exception:
                    return b""  # one failed chunk must not kill the whole job

        audio_blobs = await asyncio.gather(*(synth_job(t) for _, _, t in jobs))

        failed_chunks = sum(1 for b in audio_blobs if not b)
        if jobs and failed_chunks == len(jobs):
            # TTS itself is unreachable (e.g. host blocks Microsoft TTS) —
            # returning a silent MP3 would be worse than a clear error.
            return jsonify({"error": "Dubbing error: TTS server ကို ဆက်သွယ်မရပါ။ Server ကနေ Microsoft TTS ကို block ထားနိုင်ပါတယ် — နောက်မှ ပြန်စမ်းကြည့်ပါ။"}), 500

        # Reassemble chunks per subtitle, in order
        per_sub = {}
        for (i, j, _), blob in zip(jobs, audio_blobs):
            per_sub.setdefault(i, []).append((j, blob))

        failed_segments = 0
        for i in sorted(per_sub):
            sub = sorted_subs[i]
            try:
                segment = AudioSegment.silent(duration=0, frame_rate=24000)
                for _, blob in sorted(per_sub[i]):
                    if not blob:
                        continue
                    segment += AudioSegment.from_file(io.BytesIO(blob), format="mp3")
                if len(segment) == 0:
                    continue

                start_ms = parse_srt_time_to_ms(sub.get('startTime') or '00:00:00,000')
                end_ms = parse_srt_time_to_ms(sub.get('endTime') or '00:00:00,000')
                slot_ms = max(0, end_ms - start_ms - 120)  # 120ms breathing room
                if slot_ms > 0 and len(segment) > slot_ms:
                    speed = len(segment) / slot_ms
                    if speed <= _MAX_SPEEDUP:
                        segment = segment.speedup(playback_speed=speed)
                    else:
                        segment = segment.speedup(playback_speed=_MAX_SPEEDUP)[:slot_ms]

                # Overlay onto the main track
                combined_audio = combined_audio.overlay(segment, position=start_ms)
            except Exception:
                # One bad segment (corrupt audio, speedup failure, ...) must not
                # kill the whole dubbing job — skip it and keep going.
                failed_segments += 1
                continue

        out_fp = io.BytesIO()
        combined_audio.export(out_fp, format="mp3", bitrate="128k")
        out_fp.seek(0)

        resp = send_file(
            out_fp,
            mimetype="audio/mpeg",
            as_attachment=True,
            download_name="dubbing.mp3"
        )
        if failed_chunks or failed_segments:
            resp.headers["X-Dubbing-Failed-Chunks"] = str(failed_chunks + failed_segments)
        return resp
    except Exception as e:
        return jsonify({"error": f"Dubbing error: {str(e)}"}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    req = request.get_json(silent=True) or {}
    subtitles = req.get('subtitles', [])
    target_lang = req.get('targetLang', 'Burmese')
    tone_style = req.get('toneStyle', 'natural')
    api_key = (req.get('apiKey') or '').strip()
    raw_model = (req.get('modelName') or DEFAULT_MODEL).strip()

    if not api_key:
        return jsonify({"error": "Gemini API Key လိုအပ်ပါသည်"}), 400

    if not subtitles:
        return jsonify({"error": "ဘာသာပြန်ရန် စာတန်းထိုး မရှိပါ"}), 400

    model_id = _MODEL_CLEAN.sub('', raw_model) or DEFAULT_MODEL

    tone_descriptions = {
        'natural': 'natural spoken conversational style suitable for movie subtitles (သဘာဝကျကျ စကားပြောဟန်)',
        'formal': 'polite, elegant literary style for documentaries (ယဉ်ကျေးသပ်ရပ်သော စာဟန်ပေဟန်)',
        'explaining': 'clear, educational style (နားလည်လွယ်အောင် ရှင်းပြဟန်)',
        'casual': 'relaxed youthful casual style with modern slangs (ပေါ့ပေါ့ပါးပါး လူငယ်သုံး)'
    }
    chosen_tone = tone_descriptions.get(tone_style, tone_descriptions['natural'])

    system_prompt = (
        f"You are a professional audiovisual subtitle translator. "
        f"Translate the following subtitles into {target_lang}. "
        f"Style: {chosen_tone}. Output STRICTLY a valid JSON array of objects with keys 'id' (number) and 'translatedText' (string). "
        f"Do NOT alter or omit any IDs. Example: [{{\"id\": 1, \"translatedText\": \"မင်္ဂလာပါ\"}}]"
    )

    glossary = "\n".join(
        ln.strip() for ln in str(req.get('glossary') or '').splitlines() if ln.strip()
    )[:4000]
    if glossary:
        system_prompt += (
            " Glossary (source term = required translation): whenever a source term below appears, "
            "translate it exactly as given and keep it consistent across all subtitles.\n" + glossary
        )

    payload_data = [{"id": s["id"], "text": s["originalText"]} for s in subtitles]

    try:
        endpoint = f"{GEMINI_BASE_URL}{model_id}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": system_prompt},
                        {"text": json.dumps(payload_data, ensure_ascii=False)}
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.25
            }
        }

        res = http.post(endpoint, headers=headers, json=body, timeout=60)

        if res.status_code != 200:
            try:
                err_data = res.json().get('error', {})
                err_msg = err_data.get('message', f'Gemini Error ({res.status_code})')
            except Exception:
                err_msg = f'Gemini Error ({res.status_code})'
            return jsonify({"error": err_msg}), res.status_code

        res_json = res.json()
        candidates = res_json.get('candidates') or []
        if not candidates or 'content' not in candidates[0]:
            return jsonify({"error": "Gemini မှ စာပြန်မထုတ်ပေးနိုင်ပါ"}), 400

        parts = candidates[0]['content'].get('parts') or []
        raw_text = ''.join(p.get('text', '') for p in parts)
        cleaned_json = strip_markdown_fences(raw_text)
        translations = json.loads(cleaned_json)

        if isinstance(translations, dict):
            translations = translations.get('translations', translations.get('subtitles', []))

        return jsonify({"translations": translations if isinstance(translations, list) else []})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Server Timeout ဖြစ်သွားပါသည် (Block Size ကို လျှော့ပေးပါ)"}), 504
    except json.JSONDecodeError:
        return jsonify({"error": "AI ပြန်ပို့သော JSON format မမှန်ပါ"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

HTML_PAGE = """<!DOCTYPE html>
<html lang="my">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Thiri's Koko — AI Subtitle Studio Pro</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #150610;
      background-image: radial-gradient(circle at 50% 0%, #2f0e24 0%, #12040d 100%);
      color: #ffe4e6;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      min-height: 100vh;
      padding-bottom: 105px;
      overflow-x: hidden;
    }
    header {
      position: sticky; top: 0; z-index: 40;
      background: rgba(36, 12, 26, 0.96);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(244, 114, 182, 0.2);
      padding: 12px 16px;
      display: flex; align-items: center; justify-content: space-between;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-icon {
      width: 32px; height: 32px; border-radius: 10px;
      background: linear-gradient(135deg, #e11d48, #ec4899);
      display: flex; align-items: center; justify-content: center;
      color: white; font-weight: bold; font-size: 16px;
    }
    .brand-title { font-size: 15px; font-weight: 700; color: #fff; line-height: 1.2; }
    .brand-subtitle { font-size: 10px; color: #f472b6; }
    
    .btn-icon {
      background: rgba(225, 29, 72, 0.15);
      border: 1px solid rgba(244, 114, 182, 0.3);
      color: #fbcfe8; width: 36px; height: 36px; border-radius: 10px;
      font-size: 18px; cursor: pointer; display: flex; align-items: center; justify-content: center;
    }

    .container { max-width: 680px; margin: 0 auto; padding: 12px 16px; display: flex; flex-direction: column; gap: 12px; }
    
    .card {
      background: rgba(43, 15, 31, 0.85);
      border: 1px solid rgba(244, 114, 182, 0.2);
      border-radius: 16px; padding: 14px;
    }

    .upload-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .upload-info { display: flex; align-items: center; gap: 12px; }
    .upload-icon {
      width: 42px; height: 42px; min-width: 42px; max-width: 42px;
      border-radius: 12px; background: rgba(225, 29, 72, 0.15);
      color: #f472b6; display: flex; align-items: center; justify-content: center;
    }
    .upload-icon svg { width: 22px; height: 22px; }
    .upload-btn {
      background: linear-gradient(135deg, #e11d48, #ec4899);
      color: white; font-size: 12px; font-weight: 600;
      padding: 9px 16px; border-radius: 12px; border: none; cursor: pointer;
    }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(244, 114, 182, 0.15); }
    @media(max-width: 480px) { .grid-2 { grid-template-columns: 1fr; } }
    
    label { display: block; font-size: 11px; font-weight: 500; color: #f472b6; margin-bottom: 5px; }
    select, input[type="text"], input[type="password"] {
      width: 100%; background: #1f0716; border: 1px solid rgba(244, 114, 182, 0.25);
      color: #ffe4e6; padding: 9px 12px; border-radius: 10px; font-size: 12px; outline: none;
    }

    /* Video Player Box */
    #videoContainer {
      display: none; position: sticky; top: var(--header-h, 61px); z-index: 35;
      box-shadow: 0 8px 16px rgba(0, 0, 0, 0.45); border-radius: 16px; overflow: hidden;
      background: #000; border: 1px solid rgba(244, 114, 182, 0.3);
    }
    video { width: 100%; max-height: 240px; display: block; outline: none; }
    #videoSubOverlay {
      position: absolute; bottom: 35px; left: 10px; right: 10px; text-align: center;
      pointer-events: none;
    }
    #videoSubOverlay span {
      background: rgba(0, 0, 0, 0.82); color: #fef08a; padding: 4px 10px;
      border-radius: 6px; font-size: 13px; font-weight: 600; text-shadow: 0 1px 2px #000;
    }

    /* Timing Sync Bar */
    .sync-bar {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 12px; background: rgba(30, 8, 22, 0.7);
      border-radius: 12px; border: 1px solid rgba(244, 114, 182, 0.15);
      font-size: 11px; color: #f472b6;
    }
    .sync-btns { display: flex; gap: 6px; }
    .btn-sync {
      background: rgba(225, 29, 72, 0.2); border: 1px solid rgba(244, 114, 182, 0.3);
      color: #ffe4e6; padding: 4px 9px; border-radius: 8px; font-size: 11px; font-weight: bold; cursor: pointer;
    }

    #progressContainer { display: none; margin-top: 5px; }
    .progress-bar-bg { width: 100%; background: #26081c; border: 1px solid rgba(244, 114, 182, 0.2); border-radius: 20px; height: 8px; overflow: hidden; }
    .progress-bar-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #e11d48, #ec4899); transition: width 0.3s; }
    .progress-text { display: flex; justify-content: space-between; font-size: 10px; color: #f472b6; margin-top: 4px; }
    
    #statusPill {
      display: none; background: rgba(54, 14, 38, 0.95); border: 1px solid rgba(244, 114, 182, 0.35);
      border-radius: 14px; padding: 10px 14px; font-size: 11px; color: #fbcfe8;
      align-items: center; justify-content: space-between; gap: 10px; width: 100%;
    }
    #statusText { flex: 1; min-width: 0; word-break: break-word; overflow-wrap: anywhere; line-height: 1.4; }
    .btn-stop {
      background: #be123c; color: white; border: none; padding: 6px 12px;
      border-radius: 10px; font-size: 11px; font-weight: bold; cursor: pointer; flex-shrink: 0;
    }

    /* Subtitle Item Cards */
    .sub-item {
      background: rgba(38, 12, 27, 0.85); border: 1px solid rgba(244, 114, 182, 0.15);
      border-radius: 14px; padding: 12px; margin-bottom: 8px;
    }
    .sub-header {
      display: flex; justify-content: space-between; align-items: center;
      font-size: 10px; font-family: monospace; color: #f472b6;
      border-bottom: 1px solid rgba(244, 114, 182, 0.1); padding-bottom: 6px; margin-bottom: 8px;
    }
    .sub-id {
      background: rgba(225, 29, 72, 0.15); padding: 2px 6px; border-radius: 6px;
      font-weight: bold; cursor: pointer;
    }
    .btn-retrans {
      background: rgba(244, 114, 182, 0.15); border: 1px solid rgba(244, 114, 182, 0.3);
      color: #fbcfe8; font-size: 10px; font-family: sans-serif; padding: 2px 8px; border-radius: 6px; cursor: pointer;
    }
    .sub-orig { font-size: 12px; color: #fecdd3; margin-bottom: 8px; line-height: 1.4; }
    textarea {
      width: 100%; background: #170410; border: 1px solid rgba(244, 114, 182, 0.2);
      border-radius: 10px; color: #fff; padding: 8px; font-size: 12px; resize: none; outline: none;
    }

    footer {
      position: fixed; bottom: 0; left: 0; right: 0; z-index: 30;
      background: rgba(30, 8, 22, 0.98); backdrop-filter: blur(12px);
      border-top: 1px solid rgba(244, 114, 182, 0.2);
      padding: 10px 16px; max-width: 680px; margin: 0 auto;
    }
    .footer-info { display: flex; justify-content: space-between; font-size: 11px; color: #f472b6; margin-bottom: 6px; }
    .footer-btns { display: flex; gap: 8px; }
    .btn-translate {
      flex: 1; background: linear-gradient(135deg, #e11d48, #ec4899);
      color: white; border: none; font-size: 12px; font-weight: 700;
      padding: 11px 12px; border-radius: 12px; cursor: pointer;
    }
    .btn-export {
      background: #2b0b1f; border: 1px solid rgba(244, 114, 182, 0.3);
      color: #ffe4e6; font-size: 11px; font-weight: 600; padding: 11px 12px; border-radius: 12px; cursor: pointer;
    }

    /* Modals & Drawer */
    .modal-overlay {
      position: fixed; inset: 0; z-index: 50; background: rgba(0,0,0,0.75);
      backdrop-filter: blur(6px); display: none; align-items: center; justify-content: center; padding: 16px;
    }
    .modal-box {
      background: #240b1b; border: 1px solid rgba(244, 114, 182, 0.3);
      border-radius: 20px; width: 100%; max-width: 440px; padding: 18px; max-height: 90vh; overflow-y: auto;
    }
    .modal-head { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(244, 114, 182, 0.15); padding-bottom: 10px; margin-bottom: 12px; }
    .close-btn { background: none; border: none; color: #f472b6; font-size: 18px; cursor: pointer; }

    /* Hamburger Drawer */
    #drawer {
      position: fixed; top: 0; right: 0; bottom: 0; width: 290px;
      background: #230919; border-left: 1px solid rgba(244, 114, 182, 0.3);
      z-index: 60; transform: translateX(100%); transition: transform 0.25s ease-in-out;
      padding: 20px; display: flex; flex-direction: column; gap: 12px;
    }
    #drawer.open { transform: translateX(0); }
    .drawer-item {
      display: flex; align-items: center; gap: 10px; padding: 12px;
      background: rgba(225, 29, 72, 0.1); border: 1px solid rgba(244, 114, 182, 0.2);
      border-radius: 12px; color: #fff; text-decoration: none; font-size: 13px; font-weight: 600; cursor: pointer;
    }

    /* Follow-while-watching (Video နှင့်အတူ စာတန်းလိုက်ပြ / လိုက်ပြင်) */
    #followBar { display: none; flex-wrap: wrap; justify-content: flex-start; gap: 8px 16px; }
    .follow-opt { display: flex; align-items: center; gap: 6px; margin-bottom: 0; cursor: pointer; }
    .follow-opt input { accent-color: #ec4899; width: 15px; height: 15px; }
    .sub-item.active { border-color: #f472b6; background: rgba(78, 20, 54, 0.95); box-shadow: 0 0 0 1px #f472b6; }
    .sub-orig[contenteditable] { white-space: pre-wrap; min-height: 1.4em; outline: none; border-radius: 6px; padding: 2px 4px; margin-left: -4px; margin-right: -4px; }
    .sub-orig[contenteditable]:focus { background: rgba(244, 114, 182, 0.1); }
  </style>
</head>
<body>

  <!-- Header with Hamburger -->
  <header>
    <div class="brand">
      <div class="brand-icon">❤</div>
      <div>
        <div class="brand-title">Thiri's Koko</div>
        <div class="brand-subtitle">AI Subtitle Studio Pro</div>
      </div>
    </div>
    <button class="btn-icon" onclick="toggleDrawer(true)">☰</button>
  </header>

  <!-- Hamburger Drawer -->
  <div id="drawerOverlay" class="modal-overlay" onclick="toggleDrawer(false)"></div>
  <div id="drawer">
    <div class="modal-head">
      <div style="font-size: 15px; font-weight: bold; color: #fff;">Menu</div>
      <button class="close-btn" onclick="toggleDrawer(false)">✕</button>
    </div>
    
    <div class="drawer-item" onclick="toggleDrawer(false); openSettingsModal();">
      <span>⚙</span> Settings & Tuning
    </div>

    <div class="drawer-item" onclick="toggleDrawer(false); openGuideModal();">
      <span>🔑</span> API Key ယူနည်း အသေးစိတ် Guide
    </div>


  </div>

  <div class="container">
    <!-- Synced Video Player Box -->
    <div id="videoContainer">
      <video id="mainVideo" controls></video>
      <div id="videoSubOverlay"><span id="liveSubText"></span></div>
    </div>

    <!-- Progress Bar -->
    <div id="progressContainer">
      <div class="progress-bar-bg">
        <div id="progressBar" class="progress-bar-fill"></div>
      </div>
      <div class="progress-text">
        <span id="progressText">0%</span>
        <span id="progressCount">0 / 0</span>
      </div>
    </div>

    <!-- Status Banner -->
    <div id="statusPill">
      <span id="statusText">Ready</span>
      <button id="btnStop" class="btn-stop" onclick="stopTranslation()" style="display:none;">Stop ⏹</button>
    </div>

    <!-- Upload Card -->
    <div class="card">
      <div class="upload-row">
        <div class="upload-info">
          <div class="upload-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>
          <div>
            <div style="font-size: 13px; font-weight: 700; color: #fff;">Upload SRT / Audio / Video</div>
            <div style="font-size: 11px; color: #f472b6;">Total: <b id="subCount" style="color: #fff;">0</b> items</div>
          </div>
        </div>
        <input type="file" id="fileInput" accept=".srt,video/*,audio/*" style="display:none;"/>
        <button class="upload-btn" onclick="document.getElementById('fileInput').click()">Choose File</button>
      </div>

      <div class="grid-2">
        <div>
          <label>Target Language</label>
          <select id="targetLang">
            <option value="Burmese">မြန်မာစာ (Burmese)</option>
            <option value="English">English</option>
            <option value="Thai">ภาษาไทย (Thai)</option>
            <option value="Japanese">日本語 (Japanese)</option>
          </select>
        </div>
        <div>
          <label>စကားပြောပုံစံ / အသုံးအနှုန်းဟန်</label>
          <select id="toneStyle">
            <option value="natural">🗣️ သဘာဝကျ စကားပြောဟန် (ရုပ်ရှင်)</option>
            <option value="formal">📖 စာဟန်ပေဟန် (ယဉ်ကျေး/သပ်ရပ်)</option>
            <option value="explaining">🎓 ရှင်းပြသလိုဟန် (နားလည်လွယ်)</option>
            <option value="casual">🎭 ပေါ့ပေါ့ပါးပါး လူငယ်သုံး</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Timing Adjustment Controls (±0.5s) -->
    <div class="sync-bar">
      <span>⏱ Subtitle Sync (အသံနှင့်စာ ချိန်ညှိရန်):</span>
      <div class="sync-btns">
        <button class="btn-sync" onclick="adjustTiming(-0.5)">-0.5s ⏪</button>
        <button class="btn-sync" onclick="adjustTiming(0.5)">+0.5s ⏩</button>
      </div>
    </div>

    <!-- Video Follow / Edit Options (Video တင်ထားမှသာ ပေါ်မည်) -->
    <div class="sync-bar" id="followBar">
      <label class="follow-opt"><input type="checkbox" id="optFollow" checked> 🎯 Video နှင့်အတူ စာတန်း လိုက်ပြမည်</label>
      <label class="follow-opt"><input type="checkbox" id="optAutoPause" checked> ⏸ စာရေးလျှင် Video ရပ်မည်</label>
    </div>

    <!-- Subtitle Cards List -->
    <div id="subList">
      <div class="card" style="text-align: center; padding: 40px 10px; color: #f472b6;">
        <div style="font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 4px;">No Subtitles Loaded</div>
        <div style="font-size: 11px; opacity: 0.8;">Choose File နှိပ်၍ .srt ဖိုင် သို့မဟုတ် ဗီဒီယို တင်ပေးပါ</div>
      </div>
    </div>
  </div>

  <!-- Sticky Footer -->
  <footer>
    <div class="footer-info">
      <span>Model: <b id="footerModelName" style="color:#fff;">gemini-3.5-flash-lite</b></span>
      <span>Block: <b id="footerBlockSize" style="color:#fff;">25</b> | Delay: <b id="footerDelaySec" style="color:#fff;">15s</b></span>
    </div>
    <div class="footer-btns">
      <button class="btn-translate" onclick="startTranslation()" id="btnTranslate">Translate All ⚡</button>
      <button class="btn-export" onclick="downloadOriginalSRT()">Orig .SRT</button>
      <button class="btn-export" onclick="downloadTranslatedSRT()">Trans .SRT</button>
      <button class="btn-export" onclick="autoSplitLongLines()" title="ရှည်လွန်းသောစာကြောင်းများ အလိုအလျောက်ခွဲမယ်">✂️ Auto-split</button>
      <button class="btn-export" onclick="downloadDubbing()" style="background: #e11d48; font-weight: bold;">🎙️ Dub</button>
    </div>
  </footer>

  <!-- Settings Modal -->
  <div id="settingsModal" class="modal-overlay">
    <div class="modal-box">
      <div class="modal-head">
        <div style="font-size: 14px; font-weight: bold; color: #fff;">⚙ Settings & Tuning</div>
        <button class="close-btn" onclick="closeSettingsModal()">✕</button>
      </div>

      <div style="display: flex; flex-direction: column; gap: 12px;">
        <div>
          <label>Official Gemini Model</label>
          <select id="modalGeminiSelect">
            <option value="gemini-3.5-flash-lite">Gemini 3.5 Flash Lite (အကြံပြုချက်: RPM 20)</option>
            <option value="gemini-3.1-flash-lite">Gemini 3.1 Flash Lite</option>
            <option value="gemini-3.5-flash">Gemini 3.5 Flash</option>
            <option value="gemini-3-flash">Gemini 3 Flash</option>
            <option value="gemini-3.6-flash">Gemini 3.6 Flash</option>
            <option value="gemini-3.8-flash">Gemini 3.8 Flash</option>
          </select>
        </div>

        <div class="grid-2" style="margin-top:0; padding-top:0; border:none;">
          <div>
            <label>Block Size (စာကြောင်းရေ)</label>
            <select id="modalBlockSize">
              <option value="15">15 (အလွန်ငြိမ်)</option>
              <option value="25" selected>25 (အသင့်တော်ဆုံး)</option>
              <option value="30">30 (ပုံမှန်)</option>
              <option value="50">50 (အများဆုံး)</option>
            </select>
          </div>
          <div>
            <label>Delay (စက္ကန့်)</label>
            <select id="modalDelaySec">
              <option value="5">5s (အမြန်)</option>
              <option value="15" selected>15s (Safe)</option>
              <option value="30">30s (RPM 2 နှုန်း)</option>
            </select>
          </div>
        </div>

        <div>
          <label>Dubbing အသံရွေးရန် (TTS Voice)</label>
          <select id="modalDubbingVoice">
            <option value="my-MM-ThihaNeural">Thiha (သီဟ - ယောကျ်ားလေးအသံ)</option>
            <option value="my-MM-NilarNeural">Nilar (နီလာ - မိန်းကလေးအသံ)</option>
          </select>
        </div>

        <div>
          <label>Gemini API Key</label>
          <input type="password" id="modalGeminiKey" placeholder="AIzaSy...">
        </div>

        <div>
          <label>Groq API Key (Audio Transcribe အတွက်)</label>
          <input type="password" id="modalGroqKey" placeholder="gsk_...">
        </div>

        <div>
          <label>Glossary (နာမည် / စကားလုံး ပုံသေဘာသာပြန်ချက်)</label>
          <textarea id="modalGlossary" rows="4" placeholder="John = ဂျွန်&#10;Hogwarts = ဟော့ဂွတ်"></textarea>
          <div style="font-size: 10px; color: #f472b6; opacity: 0.8; margin-top: 4px;">တစ်ကြောင်းကို တစ်ခုစီ — မူရင်း = ဘာသာပြန်</div>
        </div>

        <button class="upload-btn" onclick="saveSettings()" style="width: 100%; margin-top: 5px; padding: 11px;">
          Save Settings & Close
        </button>
      </div>
    </div>
  </div>

  <!-- Detailed Guide Modal -->
  <div id="guideModal" class="modal-overlay">
    <div class="modal-box">
      <div class="modal-head">
        <div style="font-size: 14px; font-weight: bold; color: #fff;">🔑 API Key ရယူနည်း Guide</div>
        <button class="close-btn" onclick="closeGuideModal()">✕</button>
      </div>

      <div style="display: flex; flex-direction: column; gap: 12px; font-size: 12px; line-height: 1.5; color: #fbcfe8;">
        <!-- Gemini Guide -->
        <div class="card" style="background: rgba(225,29,72,0.12); padding: 12px;">
          <div style="font-weight: bold; color: #fff; margin-bottom: 6px;">၁။ Google Gemini API Key (အခမဲ့)</div>
          <p style="margin-bottom: 6px;">• အောက်ပါခလုတ်ကို နှိပ်ပြီး Google AI Studio သို့ Gmail ဖြင့် Sign in ဝင်ပါ</p>
          <p style="margin-bottom: 6px;">• ပေါ်လာသော စာမျက်နှာတွင် <b>"Create API key"</b> ခလုတ်ကို နှိပ်ပါ</p>
          <p style="margin-bottom: 8px;">• ရလာသော <b>AIzaSy...</b> စာကြောင်းကို Copy ယူပြီး Settings တွင် Paste ချပါ</p>
          <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" 
             style="display: inline-block; background: #4f46e5; color: #fff; padding: 6px 12px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 11px;">
            Google AI Studio သို့ သွားရန် ➔
          </a>
        </div>

        <!-- Groq Guide -->
        <div class="card" style="background: rgba(225,29,72,0.12); padding: 12px;">
          <div style="font-weight: bold; color: #fff; margin-bottom: 6px;">၂။ Groq API Key (ဗီဒီယို/အသံဖိုင်အတွက်)</div>
          <p style="margin-bottom: 6px;">• အောက်ပါခလုတ်ကို နှိပ်ပြီး Groq Console တွင် Free Account ဖွင့်ပါ</p>
          <p style="margin-bottom: 6px;">• <b>"Create API Key"</b> ခလုတ်ကို နှိပ်ပြီး နာမည်တစ်ခုခု ပေးပါ</p>
          <p style="margin-bottom: 8px;">• ရလာသော <b>gsk_...</b> စာကြောင်းကို Copy ယူပြီး Settings တွင် ထည့်ပါ</p>
          <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer" 
             style="display: inline-block; background: #e11d48; color: #fff; padding: 6px 12px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 11px;">
            Groq Console သို့ သွားရန် ➔
          </a>
        </div>

        <button class="btn-export" onclick="closeGuideModal()" style="width: 100%; padding: 10px;">ပိတ်မည်</button>
      </div>
    </div>
  </div>

  <!-- Download File Name Modal -->
  <div id="downloadModal" class="modal-overlay">
    <div class="modal-box">
      <div class="modal-head">
        <div style="font-size: 14px; font-weight: bold; color: #fff;">💾 Download ဖိုင်နာမည်</div>
        <button class="close-btn" onclick="closeDownloadModal()">✕</button>
      </div>
      <div style="display: flex; flex-direction: column; gap: 12px;">
        <div>
          <label>ဖိုင်နာမည် (.srt ကို အလိုအလျောက် ထည့်ပေးမည်)</label>
          <input type="text" id="downloadName" autocomplete="off" onkeydown="if (event.key === 'Enter') confirmDownload()">
        </div>
        <button class="upload-btn" onclick="confirmDownload()" style="width: 100%; padding: 11px;">Download ⬇</button>
      </div>
    </div>
  </div>

  <script>
    let subtitles = [];
    let isTranslating = false;
    let uploadedFileName = "subtitles";
    let currentVideoUrl = null;
    let timeIndex = [];
    let lastOverlayText = null;
    let statusTimer = null;
    let progressTimer = null;
    let activeIdx = -1;
    let lastUserScroll = 0;
    let pendingDownload = null;

    const KEY_GROQ = 'thiri_koko_groq_key';
    const KEY_GEMINI = 'thiri_koko_gemini_key';
    const KEY_GEMINI_MODEL = 'thiri_koko_gemini_model';
    const KEY_BLOCK_SIZE = 'thiri_koko_block_size';
    const KEY_DELAY_SEC = 'thiri_koko_delay_sec';
    const KEY_GLOSSARY = 'thiri_koko_glossary';
    const KEY_FOLLOW = 'thiri_koko_follow';
    const KEY_AUTOPAUSE = 'thiri_koko_autopause';
    const KEY_DUB_VOICE = 'thiri_koko_dub_voice';

    const sleep = (ms) => new Promise(res => setTimeout(res, ms));

    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    function timeToSec(t) {
      const [hms, ms] = t.split(/[,.]/);
      const [h, m, s] = hms.split(':').map(Number);
      return h * 3600 + m * 60 + s + (Number(ms || 0) / 1000);
    }

    function secToTime(sec) {
      sec = Math.max(0, sec);
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = Math.floor(sec % 60);
      const ms = Math.floor((sec % 1) * 1000);
      const pad = (n, z=2) => String(n).padStart(z, '0');
      return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
    }

    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('modalGroqKey').value = localStorage.getItem(KEY_GROQ) || '';
      document.getElementById('modalGeminiKey').value = localStorage.getItem(KEY_GEMINI) || '';
      
      const savedModel = localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.5-flash-lite';
      const savedBlock = localStorage.getItem(KEY_BLOCK_SIZE) || '25';
      const savedDelay = localStorage.getItem(KEY_DELAY_SEC) || '15';
      const savedDubVoice = localStorage.getItem(KEY_DUB_VOICE) || 'my-MM-ThihaNeural';

      document.getElementById('modalGeminiSelect').value = savedModel;
      document.getElementById('modalBlockSize').value = savedBlock;
      document.getElementById('modalDelaySec').value = savedDelay;
      document.getElementById('modalDubbingVoice').value = savedDubVoice;
      document.getElementById('modalGlossary').value = localStorage.getItem(KEY_GLOSSARY) || '';

      document.getElementById('footerModelName').innerText = savedModel;
      document.getElementById('footerBlockSize').innerText = savedBlock;
      document.getElementById('footerDelaySec').innerText = `${savedDelay}s`;

      document.getElementById('fileInput').addEventListener('change', handleFileSelected);

      const video = document.getElementById('mainVideo');
      const overlay = document.getElementById('liveSubText');
      const optFollow = document.getElementById('optFollow');
      const optAutoPause = document.getElementById('optAutoPause');
      const header = document.querySelector('header');

      const setHeaderH = () => document.documentElement.style.setProperty('--header-h', `${header.offsetHeight}px`);
      setHeaderH();
      window.addEventListener('resize', setHeaderH);

      optFollow.checked = localStorage.getItem(KEY_FOLLOW) !== '0';
      optAutoPause.checked = localStorage.getItem(KEY_AUTOPAUSE) !== '0';
      optFollow.addEventListener('change', () => localStorage.setItem(KEY_FOLLOW, optFollow.checked ? '1' : '0'));
      optAutoPause.addEventListener('change', () => localStorage.setItem(KEY_AUTOPAUSE, optAutoPause.checked ? '1' : '0'));

      // ကိုယ်တိုင် scroll လုပ်နေချိန် ၃ စက္ကန့် auto-follow မလုပ်ရန်
      const markUserScroll = () => { lastUserScroll = Date.now(); };
      window.addEventListener('wheel', markUserScroll, { passive: true });
      window.addEventListener('touchmove', markUserScroll, { passive: true });

      video.addEventListener('timeupdate', () => {
        const cur = video.currentTime;
        const hit = timeIndex.find(x => cur >= x.st && cur <= x.et);
        const text = hit ? (hit.s.translatedText || hit.s.originalText) : '';
        if (text !== lastOverlayText) {
          lastOverlayText = text;
          overlay.innerText = text;
          overlay.style.display = text ? 'inline-block' : 'none';
        }
        setActiveCue(calcActive(cur));
      });
      video.addEventListener('seeked', () => { lastUserScroll = 0; });

      // စာရေးဖို့ နှိပ်လျှင် Video ရပ်ပြီး ထို card ကို Video အောက်တွင် ပြမည်
      document.getElementById('subList').addEventListener('focusin', (e) => {
        if (!currentVideoUrl || !e.target.matches('textarea, [contenteditable]')) return;
        if (optAutoPause.checked && !video.paused) video.pause();
        const card = e.target.closest('.sub-item');
        if (card) setTimeout(() => scrollToCue(card, true), 250);
      });

      // ESC နှိပ်ရင် modal/drawer တွေ ပိတ်ရန်
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          closeSettingsModal(); closeGuideModal(); closeDownloadModal(); toggleDrawer(false);
        }
      });
    });

    function toggleDrawer(open) {
      document.getElementById('drawer').classList.toggle('open', open);
      document.getElementById('drawerOverlay').style.display = open ? 'flex' : 'none';
    }

    function openSettingsModal() { document.getElementById('settingsModal').style.display = 'flex'; }
    function closeSettingsModal() { document.getElementById('settingsModal').style.display = 'none'; }
    function openGuideModal() { document.getElementById('guideModal').style.display = 'flex'; }
    function closeGuideModal() { document.getElementById('guideModal').style.display = 'none'; }

    function saveSettings() {
      const gKey = document.getElementById('modalGroqKey').value.trim();
      const gmKey = document.getElementById('modalGeminiKey').value.trim();
      const selectedModel = document.getElementById('modalGeminiSelect').value;
      const blockSize = document.getElementById('modalBlockSize').value;
      const delaySec = document.getElementById('modalDelaySec').value;
      const dubVoice = document.getElementById('modalDubbingVoice').value;

      localStorage.setItem(KEY_GROQ, gKey);
      localStorage.setItem(KEY_GEMINI, gmKey);
      localStorage.setItem(KEY_GEMINI_MODEL, selectedModel);
      localStorage.setItem(KEY_BLOCK_SIZE, blockSize);
      localStorage.setItem(KEY_DELAY_SEC, delaySec);
      localStorage.setItem(KEY_DUB_VOICE, dubVoice);
      localStorage.setItem(KEY_GLOSSARY, document.getElementById('modalGlossary').value.trim());

      document.getElementById('footerModelName').innerText = selectedModel;
      document.getElementById('footerBlockSize').innerText = blockSize;
      document.getElementById('footerDelaySec').innerText = `${delaySec}s`;

      closeSettingsModal();
      setStatus("Settings မှတ်သားပြီးပါပြီ!", 3000);
    }

    function setStatus(text, duration = 0) {
      const pill = document.getElementById('statusPill');
      document.getElementById('statusText').innerText = text;
      pill.style.display = 'flex';
      clearTimeout(statusTimer);
      if (duration > 0) statusTimer = setTimeout(() => { pill.style.display = 'none'; }, duration);
    }

    function updateProgress(done, total) {
      const pContainer = document.getElementById('progressContainer');
      pContainer.style.display = 'block';
      const pct = total ? Math.round((done / total) * 100) : 0;
      document.getElementById('progressBar').style.width = `${pct}%`;
      document.getElementById('progressText').innerText = `${pct}%`;
      document.getElementById('progressCount').innerText = `${done} / ${total}`;
      clearTimeout(progressTimer);
      if (done >= total) progressTimer = setTimeout(() => { pContainer.style.display = 'none'; }, 3000);
    }

    async function handleFileSelected(e) {
      const file = e.target.files[0];
      e.target.value = '';
      if (!file) return;

      const MAX_WHISPER_SIZE = 25 * 1024 * 1024; // 25 MB

      uploadedFileName = file.name.substring(0, file.name.lastIndexOf('.')) || "subtitles";
      const ext = file.name.split('.').pop().toLowerCase();

      let attachOnly = false;
      if (['mp4', 'webm', 'mov'].includes(ext)) {
        const video = document.getElementById('mainVideo');
        if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
        currentVideoUrl = URL.createObjectURL(file);
        video.src = currentVideoUrl;
        document.getElementById('videoContainer').style.display = 'block';
        document.getElementById('followBar').style.display = 'flex';
        attachOnly = subtitles.length > 0 && confirm("Subtitle ရှိပြီးသား ဖြစ်ပါသည်။\\n\\nOK = Video ကို SRT နှင့် တိုက်စစ်ရန်သာ ထည့်မည် (Transcribe မလုပ်ပါ)\\nCancel = Whisper AI ဖြင့် Subtitle အသစ်ထုတ်မည် (ရှိပြီးသားကို အစားထိုးမည်)");
      }

      if (ext === 'srt') {
        const txt = await file.text();
        parseSRT(txt);
        setStatus("SRT file loaded!", 3000);
      } else if (attachOnly) {
        setStatus("Video ကို SRT နှင့် တွဲထည့်ပြီးပါပြီ", 3000);
      } else {
        if (file.size > MAX_WHISPER_SIZE) {
          alert("Groq Whisper API သည် 25MB ထက်ကြီးသော ဖိုင်များကို လက်မခံပါ။ ဖိုင်ဆိုဒ်သေးအောင် လုပ်ပြီးမှ ပြန်တင်ပေးပါ။");
          return;
        }

        const groqKey = (localStorage.getItem(KEY_GROQ) || '').trim();
        if (!groqKey) {
          alert("Audio/Video transcribe လုပ်ရန် Groq API Key လိုအပ်ပါသည်။ Menu ထဲက Guide ကို ဖတ်၍ အခမဲ့ ယူနိုင်ပါသည်ခင်ဗျာ။");
          openSettingsModal();
          return;
        }

        setStatus("Whisper AI ဖြင့် Subtitle ထုတ်ယူနေပါသည်...");
        const fd = new FormData();
        fd.append('file', file);
        fd.append('apiKey', groqKey);

        try {
          const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
          const rawText = await res.text();
          let data;
          try { data = JSON.parse(rawText); } catch(e) { throw new Error("Server Timeout ဖြစ်သွားပါသည်"); }

          if (!res.ok || data.error) throw new Error(data.error || "Transcription Failed");
          subtitles = data.subtitles;
          renderList();
          setStatus("Transcription အောင်မြင်ပါသည်!", 3000);
        } catch(err) {
          alert("Error: " + err.message);
          setStatus("Transcription မအောင်မြင်ပါ", 3000);
        }
      }
    }

    function parseSRT(txt) {
      const blocks = txt.trim().replace(/\\r\\n/g, '\\n').split(/\\n\\s*\\n/);
      subtitles = [];
      blocks.forEach((b, i) => {
        const l = b.trim().split('\\n');
        if (l.length >= 2) {
          const tIdx = /^\\d+$/.test(l[0].trim()) ? 1 : 0;
          if (l[tIdx] && l[tIdx].includes('-->')) {
            const [s, e] = l[tIdx].split('-->').map(x => x.trim());
            subtitles.push({
              id: i + 1,
              startTime: s,
              endTime: e,
              originalText: l.slice(tIdx + 1).join('\\n'),
              translatedText: ''
            });
          }
        }
      });
      renderList();
    }

    function adjustTiming(delta) {
      if (!subtitles.length) return;
      subtitles.forEach(s => {
        const st = Math.max(0, timeToSec(s.startTime) + delta);
        const et = Math.max(0, timeToSec(s.endTime) + delta);
        s.startTime = secToTime(st);
        s.endTime = secToTime(et);
      });
      renderList();
      setStatus(`စာတန်းထိုး အချိန် ${delta > 0 ? '+' : ''}${delta}s ချိန်ညှိပြီးပါပြီ!`, 2500);
    }

    function seekVideoTo(timeStr) {
      const video = document.getElementById('mainVideo');
      if (video && video.src) {
        video.currentTime = timeToSec(timeStr);
        video.play();
      }
    }

    function calcActive(cur) {
      let best = -1, bestSt = -1;
      for (let i = 0; i < timeIndex.length; i++) {
        const st = timeIndex[i].st;
        if (st <= cur && st >= bestSt) { best = i; bestSt = st; }
      }
      return best;
    }

    function scrollToCue(el, force = false) {
      if (!force) {
        const ae = document.activeElement;
        if (ae && (ae.tagName === 'TEXTAREA' || ae.isContentEditable)) return;
      }
      const vc = document.getElementById('videoContainer');
      const offset = document.querySelector('header').offsetHeight + (vc.offsetHeight || 0) + 8;
      const top = el.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top: Math.max(0, top), behavior: force ? 'auto' : 'smooth' });
    }

    function setActiveCue(idx) {
      if (idx === activeIdx) return;
      const box = document.getElementById('subList');
      const prev = activeIdx >= 0 ? box.children[activeIdx] : null;
      if (prev) prev.classList.remove('active');
      activeIdx = idx;
      const el = idx >= 0 ? box.children[idx] : null;
      if (!el) return;
      el.classList.add('active');
      if (document.getElementById('optFollow').checked && Date.now() - lastUserScroll > 3000) scrollToCue(el);
    }

    // Chunk တစ်ခုပြီးတိုင်း renderList() အပြည့်ပြန်ခေါ်ရင် user ရိုက်နေတဲ့စာ ပျက်သွားမယ်။
    // ဒါကြောင့် ဘာသာပြန်ပြီးတဲ့ textarea တွေကိုပဲ in-place update လုပ်တယ်။
    function refreshTranslatedTexts() {
      const box = document.getElementById('subList');
      subtitles.forEach((s, idx) => {
        const card = box.children[idx];
        if (!card || !card.classList.contains('sub-item')) return;
        const ta = card.querySelector('textarea');
        if (ta && document.activeElement !== ta && ta.value !== s.translatedText) {
          ta.value = s.translatedText;
        }
      });
      document.getElementById('subCount').innerText = subtitles.length;
    }

    // ---------- Merge / Split subtitles ----------
    const SPLIT_LIMIT = 110;   // auto-split threshold (chars)
    const SPLIT_MIN_MS = 400;  // each split part keeps at least this much time

    function renumberSubtitles() {
      subtitles.forEach((s, i) => { s.id = i + 1; });
    }

    function parseSrtToMs(t) {
      try { return Math.max(0, Math.round(timeToSec(t) * 1000)); }
      catch (e) { return 0; }
    }
    function fmtSrt(ms) { return secToTime(ms / 1000); }

    // textarea / contenteditable ထဲက လက်ရှိစာကို model ထဲ ပြန်သိမ်း
    function syncCardToModel(idx) {
      const card = document.getElementById('subList').children[idx];
      if (!card || !card.classList.contains('sub-item') || !subtitles[idx]) return;
      const ta = card.querySelector('textarea');
      const orig = card.querySelector('.sub-orig');
      if (ta) subtitles[idx].translatedText = ta.value;
      if (orig) subtitles[idx].originalText = orig.innerText.trim();
    }
    function syncAllCards() {
      subtitles.forEach((_, idx) => syncCardToModel(idx));
    }

    function joinText(x, y) {
      x = (x || '').trim(); y = (y || '').trim();
      return (x && y) ? x + ' ' + y : (x || y);
    }

    // pos အနီးဆုံး စာကြောင်း/စကားလုံး အဆုံးသတ်နေရာကို ရှာ
    function snapCut(text, pos) {
      pos = Math.max(1, Math.min(text.length - 1, Math.round(pos)));
      const sent = /[။.!?!\n]/g;
      let m, best = -1;
      while ((m = sent.exec(text))) {
        const p = m.index + 1;
        if (Math.abs(p - pos) < Math.abs(best - pos)) best = p;
      }
      if (best > 0 && best < text.length && Math.abs(best - pos) < 40) return best;
      const fwd = text.indexOf(' ', pos), bwd = text.lastIndexOf(' ', pos);
      if (fwd === -1) return bwd > 0 ? bwd : pos;
      if (bwd === -1) return fwd;
      return (pos - bwd <= fwd - pos) ? bwd : fwd;
    }

    function splitTimeProportionally(startMs, endMs, len1, len2) {
      const dur = Math.max(1, endMs - startMs);
      const ratio = len1 / Math.max(1, len1 + len2);
      let b = startMs + Math.round(dur * ratio);
      if (b - startMs < SPLIT_MIN_MS) b = startMs + SPLIT_MIN_MS;
      if (endMs - b < SPLIT_MIN_MS) b = endMs - SPLIT_MIN_MS;
      if (b <= startMs || b >= endMs) b = startMs + Math.floor(dur / 2);
      return Math.max(startMs + 1, Math.min(endMs - 1, b));
    }

    // Model-only split (DOM မထိ). cutPos > 0 ဆို cursor နေရာ အတိအကျ၊ မဟုတ်ရင် auto.
    function splitModelAt(idx, cutPos) {
      const s = subtitles[idx];
      if (!s) return false;
      const primary = (s.translatedText || '').trim() ? 'translatedText' : 'originalText';
      const other = primary === 'translatedText' ? 'originalText' : 'translatedText';
      const text = (s[primary] || '').trim();
      if (text.length < 2) return false;

      const cut = (cutPos > 0 && cutPos < text.length)
        ? Math.floor(cutPos)
        : snapCut(text, text.length / 2);
      const t1 = text.slice(0, cut).trim(), t2 = text.slice(cut).trim();
      if (!t1 || !t2) return false;

      const startMs = parseSrtToMs(s.startTime), endMs = parseSrtToMs(s.endTime);
      const boundary = splitTimeProportionally(startMs, endMs, t1.length, t2.length);

      // ကျန်တစ်ဖက်စာသားကိုလည်း အချိုးကျ ခွဲ (မခွဲနိုင်ရင် အတိုင်းထား)
      const otext = (s[other] || '').trim();
      let o1 = '', o2 = '';
      if (otext) {
        const oc = snapCut(otext, otext.length * (cut / text.length));
        o1 = otext.slice(0, oc).trim(); o2 = otext.slice(oc).trim();
        if (!o1 || !o2) { o1 = otext; o2 = ''; }
      }

      const oldEnd = s.endTime;
      const first = Object.assign({}, s, { endTime: fmtSrt(boundary) });
      first[primary] = t1; first[other] = o1;
      const second = { id: 0, startTime: fmtSrt(boundary), endTime: oldEnd, originalText: '', translatedText: '' };
      second[primary] = t2; second[other] = o2;

      subtitles[idx] = first;
      subtitles.splice(idx + 1, 0, second);
      return true;
    }

    // Card ပေါ်က ✂️ ခလုတ် — cursor နေရာမှာ ခွဲ၊ cursor မရှိရင် auto
    function splitSubtitle(idx) {
      if (isTranslating) return setStatus('ဘာသာပြန်နေတုန်း ပြင်ဆင်လို့မရပါ — ပြီးအောင်စောင့်ပါ', 3000);
      syncCardToModel(idx);
      const card = document.getElementById('subList').children[idx];
      const ta = card ? card.querySelector('textarea') : null;
      let cut = -1;
      if (ta && document.activeElement === ta && ta.selectionStart > 0 && ta.selectionStart < ta.value.length) {
        cut = ta.selectionStart;
      }
      if (splitModelAt(idx, cut)) {
        renumberSubtitles();
        renderList();
        setStatus(`#${idx + 1} ကို ၂ ပိုင်းခွဲပြီးပါပြီ ✂️`, 2500);
      } else {
        setStatus('ခွဲ၍မရပါ — စာအရမ်းတိုနေနိုင်ပါတယ်', 3000);
      }
    }

    // Card ပေါ်က 🔗 ခလုတ် — နောက်စာကြောင်းနဲ့ ပေါင်း
    function mergeWithNext(idx) {
      if (isTranslating) return setStatus('ဘာသာပြန်နေတုန်း ပြင်ဆင်လို့မရပါ — ပြီးအောင်စောင့်ပါ', 3000);
      const a = subtitles[idx], b = subtitles[idx + 1];
      if (!a || !b) return;
      syncCardToModel(idx); syncCardToModel(idx + 1);
      a.originalText = joinText(a.originalText, b.originalText);
      a.translatedText = joinText(a.translatedText, b.translatedText);
      a.endTime = b.endTime;
      subtitles.splice(idx + 1, 1);
      renumberSubtitles();
      renderList();
      setStatus(`#${a.id} ပေါင်းစပ်ပြီးပါပြီ 🔗`, 2500);
    }

    // Footer ခလုတ် — စာလုံး 110 ထက်ရှည်တဲ့ စာကြောင်းအားလုံး အလိုအလျောက်ခွဲ
    function autoSplitLongLines() {
      if (isTranslating) return setStatus('ဘာသာပြန်နေတုန်း ပြင်ဆင်လို့မရပါ — ပြီးအောင်စောင့်ပါ', 3000);
      if (!subtitles.length) return;
      syncAllCards();
      let count = 0;
      subtitles.forEach(s => { if ((s.translatedText || '').trim().length > SPLIT_LIMIT) count++; });
      if (!count) return setStatus(`စာလုံး ${SPLIT_LIMIT} ထက်ရှည်သော စာကြောင်းမရှိပါ ✓`, 3000);
      if (!confirm(`ရှည်လွန်းသော စာကြောင်း ${count} ကြောင်း တွေ့ပါတယ်။ အလိုအလျောက် ခွဲပေးမလား?`)) return;
      let splits = 0, i = 0, guard = 0;
      while (i < subtitles.length && guard++ < 5000) {
        const t = (subtitles[i].translatedText || '').trim();
        if (t.length > SPLIT_LIMIT && splitModelAt(i, -1)) { splits++; continue; }
        i++;
      }
      renumberSubtitles();
      renderList();
      setStatus(`စာကြောင်း ${splits} ကြောင်း ခွဲပြီးပါပြီ ✂️`, 3500);
    }

    function renderList() {
      timeIndex = subtitles.map(s => ({ st: timeToSec(s.startTime), et: timeToSec(s.endTime), s }));
      activeIdx = currentVideoUrl ? calcActive(document.getElementById('mainVideo').currentTime) : -1;
      const box = document.getElementById('subList');
      document.getElementById('subCount').innerText = subtitles.length;

      if (!subtitles.length) {
        box.innerHTML = '<div class="card" style="text-align: center; padding: 40px 10px; color: #f472b6;"><div style="font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 4px;">No Subtitles Loaded</div><div style="font-size: 11px; opacity: 0.8;">Choose File နှိပ်၍ .srt ဖိုင် တင်ပေးပါ</div></div>';
        return;
      }

      box.innerHTML = subtitles.map((s, idx) => `
        <div class="sub-item${idx === activeIdx ? ' active' : ''}">
          <div class="sub-header">
            <span class="sub-id" data-start="${escapeHtml(s.startTime)}" onclick="seekVideoTo(this.dataset.start)">#${s.id} (${escapeHtml(s.startTime.split(',')[0])}) ▶</span>
            <button class="btn-retrans" onclick="retranslateSingle(${s.id})">🔄 Re-translate</button>
            <button class="btn-retrans" onclick="splitSubtitle(${idx})" title="ဒီစာကြောင်းကို ၂ ပိုင်းခွဲမယ်">✂️</button>
            ${idx < subtitles.length - 1 ? `<button class="btn-retrans" onclick="mergeWithNext(${idx})" title="နောက်စာကြောင်းနဲ့ ပေါင်းမယ်">🔗</button>` : ''}
          </div>
          <div class="sub-orig" contenteditable="plaintext-only" onblur="subtitles[${idx}].originalText = this.innerText.trim()">${escapeHtml(s.originalText)}</div>
          <textarea rows="2" onchange="subtitles[${idx}].translatedText = this.value">${escapeHtml(s.translatedText)}</textarea>
        </div>
      `).join('');
    }

    async function requestTranslation(items, modelName, apiKey) {
      const res = await fetch('/api/translate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          subtitles: items.map(s => ({ id: s.id, originalText: s.originalText })),
          targetLang: document.getElementById('targetLang').value,
          toneStyle: document.getElementById('toneStyle').value,
          glossary: localStorage.getItem(KEY_GLOSSARY) || '',
          modelName: modelName,
          apiKey: apiKey
        })
      });

      const responseText = await res.text();
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (jsonErr) {
        if (responseText.includes('<html') || responseText.includes('504') || responseText.includes('502')) {
          throw new Error("Server Timeout ဖြစ်သွားပါသည် (Block Size လျှော့ပါ)");
        }
        throw new Error("AI output decode error");
      }

      if (!res.ok || data.error) throw new Error(data.error || "Request failed");
      return Array.isArray(data.translations) ? data.translations : [];
    }

    async function retranslateSingle(subId) {
      const item = subtitles.find(s => s.id === subId);
      if (!item) return;

      const geminiKey = (localStorage.getItem(KEY_GEMINI) || '').trim();
      const modelName = (localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.5-flash-lite').trim();
      if (!geminiKey) return alert("Gemini API Key လိုအပ်ပါသည်");

      setStatus(`#${subId} ကို အသစ်ပြန်ဆိုနေပါသည်...`);
      try {
        const translations = await requestTranslation([item], modelName, geminiKey);

        const match = translations.find(t => Number(t.id) === subId) || translations[0];
        if (match) {
          item.translatedText = match.translatedText;
          renderList();
          setStatus(`#${subId} အသစ်ပြန်ဆိုပြီးပါပြီ!`, 3000);
        }
      } catch(err) {
        alert("Re-translate Error: " + err.message);
      }
    }

    function stopTranslation() {
      isTranslating = false;
      document.getElementById('btnStop').style.display = 'none';
      setStatus("ရပ်တန့်လိုက်ပါပြီ", 3000);
    }

    async function startTranslation() {
      if (isTranslating) return;
      const geminiKey = (localStorage.getItem(KEY_GEMINI) || '').trim();
      const modelName = (localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.5-flash-lite').trim();
      const chunkSize = parseInt(localStorage.getItem(KEY_BLOCK_SIZE) || '25', 10);
      const cooldownSec = parseInt(localStorage.getItem(KEY_DELAY_SEC) || '15', 10);

      if (!geminiKey) {
        alert("Gemini API Key ထည့်သွင်းပေးပါ (Settings တွင် ထည့်နိုင်ပါသည်)");
        openSettingsModal();
        return;
      }
      if (!subtitles.length) return alert("Subtitle မရှိသေးပါ");

      const pending = subtitles.filter(s => !s.translatedText);
      if (!pending.length) return alert("စာကြောင်းအားလုံး ဘာသာပြန်ပြီးပါပြီ");

      isTranslating = true;
      document.getElementById('btnStop').style.display = 'inline-block';

      const byId = new Map(subtitles.map(s => [s.id, s]));
      let completedCount = subtitles.length - pending.length;
      const failedIds = [];
      updateProgress(completedCount, subtitles.length);

      for (let i = 0; i < pending.length; i += chunkSize) {
        if (!isTranslating) break;
        const chunk = pending.slice(i, i + chunkSize);

        setStatus(`[${modelName}] ဘာသာပြန်နေပါသည်: #${chunk[0].id} မှ #${chunk[chunk.length - 1].id}...`);

        let success = false;
        let retries = 0;

        while (!success && retries < 3 && isTranslating) {
          try {
            const translations = await requestTranslation(chunk, modelName, geminiKey);

            if (translations.length > 0) {
              let applied = 0;
              translations.forEach(t => {
                const item = byId.get(Number(t.id));
                if (item) { item.translatedText = t.translatedText; applied++; }
              });
              completedCount += applied;
              updateProgress(completedCount, subtitles.length);
              refreshTranslatedTexts();  // ရိုက်နေတဲ့စာ မပျက်အောင် in-place update
            }
            success = true;

            const isLastChunk = (i + chunkSize) >= pending.length;
            if (!isLastChunk && isTranslating) {
              for (let sec = cooldownSec; sec > 0; sec--) {
                if (!isTranslating) break;
                setStatus(`Cooldown (Delay ${cooldownSec}s): ${sec} စက္ကန့်...`);
                await sleep(1000);
              }
            }

          } catch(e) {
            retries++;
            const waitSec = retries * 8;
            const shortErr = e.message.length > 50 ? e.message.substring(0, 50) + "..." : e.message;
            setStatus(`Warning: ${shortErr} — ${waitSec}s အကြာတွင် ပြန်လည်ကြိုးစားပါမည် (${retries}/3)...`);
            await sleep(waitSec * 1000);
          }
        }

        // ၃ ကြိမ်လုံး fail ရင် တိတ်တိတ်လေး မကျော်ဘဲ user ကို အသိပေး + ဆက်မယ်/ရပ်မယ် မေး
        if (!success && isTranslating) {
          failedIds.push(...chunk.map(s => s.id));
          const goOn = confirm(
            `#${chunk[0].id} မှ #${chunk[chunk.length - 1].id} ကို ၃ ကြိမ်လုံး ဘာသာပြန်မရပါ။\n\n` +
            `OK = ကျန်တာတွေ ဆက်ဘာသာပြန်မယ်\nCancel = ဒီမှာတင် ရပ်မယ်`
          );
          if (!goOn) { isTranslating = false; break; }
        }
      }

      isTranslating = false;
      document.getElementById('btnStop').style.display = 'none';
      if (failedIds.length) {
        const shown = failedIds.slice(0, 10).join(', ') + (failedIds.length > 10 ? '…' : '');
        setStatus(`ပြီးစီးပါပြီ — ${failedIds.length} ကြောင်း မအောင်မြင်ပါ (#${shown}). Re-translate နဲ့ တစ်ကြောင်းချင်း ပြန်လုပ်နိုင်ပါတယ်။`, 8000);
      } else {
        setStatus("ဘာသာပြန်ဆိုခြင်း ပြီးစီးပါပြီ!", 4000);
      }
    }

    function downloadOriginalSRT() {
      if (!subtitles.length) return;
      let out = subtitles.map((s, i) => `${i + 1}\\n${s.startTime} --> ${s.endTime}\\n${s.originalText}\\n`).join('\\n');
      askDownloadName(out, `${uploadedFileName}_original`);
    }

    function downloadTranslatedSRT() {
      if (!subtitles.length) return;
      let out = subtitles.map((s, i) => `${i + 1}\\n${s.startTime} --> ${s.endTime}\\n${s.translatedText || s.originalText}\\n`).join('\\n');
      askDownloadName(out, `${uploadedFileName}_translated`);
    }

    async function downloadDubbing() {
      if (!subtitles.length) return alert("စာတန်းထိုး မရှိသေးပါ");

      const btn = document.querySelector('button[onclick="downloadDubbing()"]');
      const origText = btn.innerHTML;
      btn.innerHTML = "⏳ Wait...";
      btn.disabled = true;
      setStatus("Dubbing အသံဖိုင် ဖန်တီးနေပါသည်... စောင့်ပေးပါ...");

      try {
        const voice = localStorage.getItem(KEY_DUB_VOICE) || 'my-MM-ThihaNeural';
        const res = await fetch('/api/dubbing', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ subtitles: subtitles, voice: voice })
        });

        if (!res.ok) {
          let errData;
          try { errData = await res.json(); } catch(e) {}
          throw new Error(errData?.error || "Dubbing failed");
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${uploadedFileName}_dubbing.mp3`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        setStatus("Dubbing အသံဖိုင် ရရှိပါပြီ!", 4000);
      } catch (err) {
        alert("Error: " + err.message);
        setStatus("Dubbing ဖန်တီးခြင်း မအောင်မြင်ပါ", 4000);
      } finally {
        btn.innerHTML = origText;
        btn.disabled = false;
      }
    }

    function askDownloadName(content, defaultName) {
      pendingDownload = { content, defaultName };
      const input = document.getElementById('downloadName');
      input.value = defaultName;
      document.getElementById('downloadModal').style.display = 'flex';
      input.focus();
      input.select();
    }

    function closeDownloadModal() {
      document.getElementById('downloadModal').style.display = 'none';
      pendingDownload = null;
    }

    function confirmDownload() {
      if (!pendingDownload) return;
      const name = document.getElementById('downloadName').value
        .replace(/[\\\\/:*?"<>|]+/g, '')
        .replace(/\\.srt$/i, '')
        .trim() || pendingDownload.defaultName;
      triggerDownload(pendingDownload.content, `${name}.srt`);
      closeDownloadModal();
    }

    function triggerDownload(content, filename) {
      // BOM helps some Windows video players detect UTF-8 (Myanmar text)
      const blob = new Blob(["\ufeff", content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return Response(HTML_PAGE, mimetype='text/html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

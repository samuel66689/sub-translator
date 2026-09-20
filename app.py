import os
import re
import json
import requests
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
# 100MB File upload limit
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  

def strip_markdown_fences(text: str) -> str:
    """LLM ပြန်ပို့သော JSON မှ ```json code fences များကို သန့်စင်ပေးသည့် helper"""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()

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
        
        # Pure Groq Whisper Endpoint
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        res = requests.post(url, headers=headers, files=files, data=data, timeout=180)
        
        if res.status_code != 200:
            err_msg = res.json().get('error', {}).get('message', f'Groq Error ({res.status_code})')
            return jsonify({"error": err_msg}), 400

        result = res.json()
        items = []
        for idx, seg in enumerate(result.get('segments', []), 1):
            s, e = seg['start'], seg['end']
            def fmt(sec):
                h, m, sc = int(sec // 3600), int((sec % 3600) // 60), sec % 60
                return f"{h:02}:{m:02}:{int(sc):02},{int((sc % 1) * 1000):03}"
            
            items.append({
                "id": idx,
                "startTime": fmt(s),
                "endTime": fmt(e),
                "originalText": seg['text'].strip(),
                "translatedText": ""
            })
        return jsonify({"subtitles": items})
    except requests.exceptions.Timeout:
        return jsonify({"error": "Audio transcription timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    req = request.json or {}
    subtitles = req.get('subtitles', [])
    target_lang = req.get('targetLang', 'Burmese')
    tone_style = req.get('toneStyle', 'natural')
    api_key = req.get('apiKey', '').strip()
    raw_model = req.get('modelName', 'gemini-3.5-flash-lite').strip()

    if not api_key:
        return jsonify({"error": "Gemini API Key လိုအပ်ပါသည်"}), 400

    if not subtitles:
        return jsonify({"error": "ဘာသာပြန်ရန် စာတန်းထိုး မရှိပါ"}), 400

    # Model ID စစ်ဆေးသန့်စင်ခြင်း
    model_id = re.sub(r'[^a-zA-Z0-9\-\.]', '', raw_model)
    if not model_id:
        model_id = 'gemini-3.5-flash-lite'

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

    payload_data = [{"id": s["id"], "text": s["originalText"]} for s in subtitles]

    try:
        # Standard Clean Gemini Endpoint
        endpoint = "https://generativelanguage.googleapis.com/v1beta/models/" + model_id + ":generateContent"
        
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

        res = requests.post(endpoint, headers=headers, json=body, timeout=60)

        if res.status_code != 200:
            try:
                err_data = res.json().get('error', {})
                err_msg = err_data.get('message', f'Gemini Error ({res.status_code})')
            except Exception:
                err_msg = f'Gemini Error ({res.status_code})'
            return jsonify({"error": err_msg}), res.status_code

        res_json = res.json()
        candidates = res_json.get('candidates', [])
        if not candidates or 'content' not in candidates[0]:
            return jsonify({"error": "Gemini မှ စာပြန်မထုတ်ပေးနိုင်ပါ (Filter သို့မဟုတ် Quota Limit ကြောင့်ဖြစ်နိုင်သည်)"}), 400

        raw_text = candidates[0]['content']['parts'][0]['text']
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
      padding-bottom: 100px;
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
    .header-btns { display: flex; align-items: center; gap: 8px; }
    .btn-header {
      background: rgba(225, 29, 72, 0.15);
      border: 1px solid rgba(244, 114, 182, 0.3);
      color: #fbcfe8; padding: 6px 12px; border-radius: 10px;
      font-size: 12px; font-weight: 600; cursor: pointer;
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

    #progressContainer { display: none; margin-top: 5px; }
    .progress-bar-bg { width: 100%; background: #26081c; border: 1px solid rgba(244, 114, 182, 0.2); border-radius: 20px; height: 8px; overflow: hidden; }
    .progress-bar-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #e11d48, #ec4899); transition: width 0.3s; }
    .progress-text { display: flex; justify-content: space-between; font-size: 10px; color: #f472b6; margin-top: 4px; }
    
    #statusPill {
      display: none; background: rgba(54, 14, 38, 0.95); border: 1px solid rgba(244, 114, 182, 0.35);
      border-radius: 14px; padding: 10px 14px; font-size: 11px; color: #fbcfe8;
      align-items: center; justify-content: space-between; gap: 10px; width: 100%;
    }
    #statusText {
      flex: 1; min-width: 0; word-break: break-word; overflow-wrap: anywhere; line-height: 1.4;
    }
    .btn-stop {
      background: #be123c; color: white; border: none; padding: 6px 12px;
      border-radius: 10px; font-size: 11px; font-weight: bold; cursor: pointer;
      flex-shrink: 0; white-space: nowrap;
    }

    .sub-item {
      background: rgba(38, 12, 27, 0.85); border: 1px solid rgba(244, 114, 182, 0.15);
      border-radius: 14px; padding: 12px; margin-bottom: 8px;
    }
    .sub-header { display: flex; justify-content: space-between; font-size: 10px; font-family: monospace; color: #f472b6; border-bottom: 1px solid rgba(244, 114, 182, 0.1); padding-bottom: 6px; margin-bottom: 8px; }
    .sub-id { background: rgba(225, 29, 72, 0.15); padding: 2px 6px; border-radius: 6px; font-weight: bold; }
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

    .modal-overlay {
      position: fixed; inset: 0; z-index: 50; background: rgba(0,0,0,0.75);
      backdrop-filter: blur(6px); display: none; align-items: center; justify-content: center; padding: 16px;
    }
    .modal-box {
      background: #240b1b; border: 1px solid rgba(244, 114, 182, 0.3);
      border-radius: 20px; width: 100%; max-width: 400px; padding: 18px; max-height: 90vh; overflow-y: auto;
    }
    .modal-head { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(244, 114, 182, 0.15); padding-bottom: 10px; margin-bottom: 12px; }
    .close-btn { background: none; border: none; color: #f472b6; font-size: 18px; cursor: pointer; }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">❤</div>
      <div>
        <div class="brand-title">Thiri's Koko</div>
        <div class="brand-subtitle">AI Subtitle Studio Pro</div>
      </div>
    </div>
    <div class="header-btns">
      <button class="btn-header" onclick="openSettingsModal()">⚙ Settings</button>
      <button class="btn-header" onclick="openGuideModal()">🔑 Guide</button>
    </div>
  </header>

  <div class="container">
    <div id="progressContainer">
      <div class="progress-bar-bg">
        <div id="progressBar" class="progress-bar-fill"></div>
      </div>
      <div class="progress-text">
        <span id="progressText">0%</span>
        <span id="progressCount">0 / 0</span>
      </div>
    </div>

    <div id="statusPill">
      <span id="statusText">Ready</span>
      <button id="btnStop" class="btn-stop" onclick="stopTranslation()" style="display:none;">Stop ⏹</button>
    </div>

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

    <div id="subList">
      <div class="card" style="text-align: center; padding: 40px 10px; color: #f472b6;">
        <div style="font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 4px;">No Subtitles Loaded</div>
        <div style="font-size: 11px; opacity: 0.8;">Choose File နှိပ်၍ .srt ဖိုင် တင်ပေးပါ</div>
      </div>
    </div>
  </div>

  <footer>
    <div class="footer-info">
      <span>Model: <b id="footerModelName" style="color:#fff;">gemini-3.5-flash-lite</b></span>
      <span>Block: <b id="footerBlockSize" style="color:#fff;">25</b> | Delay: <b id="footerDelaySec" style="color:#fff;">15s</b></span>
    </div>
    <div class="footer-btns">
      <button class="btn-translate" onclick="startTranslation()" id="btnTranslate">Translate All ⚡</button>
      <button class="btn-export" onclick="downloadOriginalSRT()">Orig .SRT</button>
      <button class="btn-export" onclick="downloadTranslatedSRT()">Trans .SRT</button>
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
            <option value="gemini-3.8-flash">Gemini 3.8 Flash (RPM 5 သာရှိသဖြင့် Delay တိုးပါ)</option>
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
          <label>Gemini API Key</label>
          <input type="password" id="modalGeminiKey" placeholder="AIzaSy...">
        </div>

        <div>
          <label>Groq API Key (Audio Transcribe အတွက်)</label>
          <input type="password" id="modalGroqKey" placeholder="gsk_...">
        </div>

        <button class="upload-btn" onclick="saveSettings()" style="width: 100%; margin-top: 5px; padding: 11px;">
          Save Settings & Close
        </button>
      </div>
    </div>
  </div>

  <!-- Guide Modal -->
  <div id="guideModal" class="modal-overlay">
    <div class="modal-box">
      <div class="modal-head">
        <div style="font-size: 14px; font-weight: bold; color: #fff;">🔑 API Key ရယူနည်း Guide</div>
        <button class="close-btn" onclick="closeGuideModal()">✕</button>
      </div>

      <div style="display: flex; flex-direction: column; gap: 10px; font-size: 12px; line-height: 1.5; color: #fbcfe8;">
        <div class="card" style="background: rgba(225,29,72,0.1);">
          <div style="font-weight: bold; color: #fff; margin-bottom: 4px;">၁။ Gemini API Key (အခမဲ့)</div>
          <p>• aistudio.google.com သို့ Gmail ဖြင့် Sign in ဝင်ပါ<br>
             • Create API key ကို နှိပ်ပြီး ရလာသော Key ကို ထည့်ပါ</p>
        </div>

        <div class="card" style="background: rgba(225,29,72,0.1);">
          <div style="font-weight: bold; color: #fff; margin-bottom: 4px;">၂။ Groq API Key (အသံဖိုင်အတွက်)</div>
          <p>• console.groq.com တွင် အကောင့်ဖွင့်ပါ<br>
             • API Keys ထဲမှ အခမဲ့ ရယူနိုင်ပါသည်</p>
        </div>

        <button class="btn-export" onclick="closeGuideModal()" style="width: 100%;">နားလည်ပါပြီ</button>
      </div>
    </div>
  </div>

  <script>
    let subtitles = [];
    let isTranslating = false;
    let uploadedFileName = "subtitles";

    const KEY_GROQ = 'thiri_koko_groq_key';
    const KEY_GEMINI = 'thiri_koko_gemini_key';
    const KEY_GEMINI_MODEL = 'thiri_koko_gemini_model';
    const KEY_BLOCK_SIZE = 'thiri_koko_block_size';
    const KEY_DELAY_SEC = 'thiri_koko_delay_sec';

    const sleep = (ms) => new Promise(res => setTimeout(res, ms));

    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('modalGroqKey').value = localStorage.getItem(KEY_GROQ) || '';
      document.getElementById('modalGeminiKey').value = localStorage.getItem(KEY_GEMINI) || '';
      
      const savedModel = localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.5-flash-lite';
      const savedBlock = localStorage.getItem(KEY_BLOCK_SIZE) || '25';
      const savedDelay = localStorage.getItem(KEY_DELAY_SEC) || '15';

      document.getElementById('modalGeminiSelect').value = savedModel;
      document.getElementById('modalBlockSize').value = savedBlock;
      document.getElementById('modalDelaySec').value = savedDelay;

      document.getElementById('footerModelName').innerText = savedModel;
      document.getElementById('footerBlockSize').innerText = savedBlock;
      document.getElementById('footerDelaySec').innerText = `${savedDelay}s`;

      document.getElementById('fileInput').addEventListener('change', handleFileSelected);
    });

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

      localStorage.setItem(KEY_GROQ, gKey);
      localStorage.setItem(KEY_GEMINI, gmKey);
      localStorage.setItem(KEY_GEMINI_MODEL, selectedModel);
      localStorage.setItem(KEY_BLOCK_SIZE, blockSize);
      localStorage.setItem(KEY_DELAY_SEC, delaySec);

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
      if (duration > 0) setTimeout(() => pill.style.display = 'none', duration);
    }

    function updateProgress(done, total) {
      const pContainer = document.getElementById('progressContainer');
      pContainer.style.display = 'block';
      const pct = Math.round((done / total) * 100);
      document.getElementById('progressBar').style.width = `${pct}%`;
      document.getElementById('progressText').innerText = `${pct}%`;
      document.getElementById('progressCount').innerText = `${done} / ${total}`;
      if (done >= total) setTimeout(() => pContainer.style.display = 'none', 3000);
    }

    async function handleFileSelected(e) {
      const file = e.target.files[0];
      if (!file) return;

      uploadedFileName = file.name.substring(0, file.name.lastIndexOf('.')) || "subtitles";
      const ext = file.name.split('.').pop().toLowerCase();

      if (ext === 'srt') {
        const txt = await file.text();
        parseSRT(txt);
        setStatus("SRT file loaded!", 3000);
      } else {
        const groqKey = (localStorage.getItem(KEY_GROQ) || '').trim();
        if (!groqKey) {
          alert("Audio/Video transcribe လုပ်ရန် Settings တွင် Groq API Key ထည့်ပေးပါခင်ဗျာ။");
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

    function renderList() {
      const box = document.getElementById('subList');
      document.getElementById('subCount').innerText = subtitles.length;

      if (!subtitles.length) {
        box.innerHTML = '<div class="card" style="text-align: center; padding: 40px 10px; color: #f472b6;"><div style="font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 4px;">No Subtitles Loaded</div><div style="font-size: 11px; opacity: 0.8;">Choose File နှိပ်၍ .srt ဖိုင် တင်ပေးပါ</div></div>';
        return;
      }

      box.innerHTML = subtitles.map((s, idx) => `
        <div class="sub-item">
          <div class="sub-header">
            <span class="sub-id">#${s.id}</span>
            <span>${s.startTime.split(',')[0]} ➔ ${s.endTime.split(',')[0]}</span>
          </div>
          <div class="sub-orig">${escapeHtml(s.originalText)}</div>
          <textarea rows="2" onchange="subtitles[${idx}].translatedText = this.value">${escapeHtml(s.translatedText)}</textarea>
        </div>
      `).join('');
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

      let completedCount = subtitles.length - pending.length;
      updateProgress(completedCount, subtitles.length);

      for (let i = 0; i < pending.length; i += chunkSize) {
        if (!isTranslating) break;
        const chunk = pending.slice(i, i + chunkSize);

        setStatus(`[${modelName}] ဘာသာပြန်နေပါသည်: #${chunk[0].id} မှ #${chunk[chunk.length - 1].id}...`);

        let success = false;
        let retries = 0;

        while (!success && retries < 3 && isTranslating) {
          try {
            const res = await fetch('/api/translate', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                subtitles: chunk,
                targetLang: document.getElementById('targetLang').value,
                toneStyle: document.getElementById('toneStyle').value,
                modelName: modelName,
                apiKey: geminiKey
              })
            });

            const responseText = await res.text();
            let data;

            try {
              data = JSON.parse(responseText);
            } catch(jsonErr) {
              if (responseText.includes('<html') || responseText.includes('504') || responseText.includes('502')) {
                throw new Error("Server Timeout ဖြစ်သွားပါသည် (Block Size လျှော့ပါ)");
              }
              throw new Error("AI output decode error");
            }

            if (!res.ok || data.error) throw new Error(data.error || "Request failed");

            if (Array.isArray(data.translations) && data.translations.length > 0) {
              data.translations.forEach(t => {
                const item = subtitles.find(x => x.id === t.id);
                if (item) item.translatedText = t.translatedText;
              });
              completedCount += chunk.length;
              updateProgress(completedCount, subtitles.length);
              renderList();
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
      }

      isTranslating = false;
      document.getElementById('btnStop').style.display = 'none';
      setStatus("ဘာသာပြန်ဆိုခြင်း ပြီးစီးပါပြီ!", 4000);
    }

    function downloadOriginalSRT() {
      if (!subtitles.length) return;
      let out = subtitles.map((s, i) => `${i + 1}\\n${s.startTime} --> ${s.endTime}\\n${s.originalText}\\n`).join('\\n');
      triggerDownload(out, `${uploadedFileName}_original.srt`);
    }

    function downloadTranslatedSRT() {
      if (!subtitles.length) return;
      let out = subtitles.map((s, i) => `${i + 1}\\n${s.startTime} --> ${s.endTime}\\n${s.translatedText || s.originalText}\\n`).join('\\n');
      triggerDownload(out, `${uploadedFileName}_translated.srt`);
    }

    function triggerDownload(content, filename) {
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
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

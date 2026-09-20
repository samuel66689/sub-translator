import os
import re
import json
import requests
import time
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

def parse_srt(srt_text):
    blocks = re.split(r'\n\s*\n', srt_text.strip().replace('\r\n', '\n'))
    items = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2:
            time_idx = 1 if lines[0].strip().isdigit() else 0
            if time_idx < len(lines) and '-->' in lines[time_idx]:
                time_line = lines[time_idx]
                text = "\n".join(lines[time_idx + 1:]).strip()
                start, end = [t.strip() for t in time_line.split('-->')]
                items.append({
                    "id": len(items) + 1,
                    "startTime": start,
                    "endTime": end,
                    "originalText": text,
                    "translatedText": ""
                })
    return items

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    file = request.files.get('file')
    api_key = request.form.get('apiKey')
    if not file or not api_key:
        return jsonify({"error": "Video/Audio transcribe လုပ်ရန် Groq API Key လိုအပ်ပါသည်"}), 400

    try:
        files = {'file': (file.filename, file.read(), file.content_type or 'application/octet-stream')}
        data = {'model': 'whisper-large-v3', 'response_format': 'verbose_json'}
        headers = {'Authorization': f'Bearer {api_key}'}
        
        res = requests.post(
            'https://api.groq.com/openai/v1/audio/transcriptions',
            headers=headers,
            files=files,
            data=data,
            timeout=300
        )
        
        if res.status_code != 200:
            err_msg = res.json().get('error', {}).get('message', 'Audio Transcription Failed')
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    req = request.json or {}
    subtitles = req.get('subtitles', [])
    target_lang = req.get('targetLang', 'Burmese')
    tone_style = req.get('toneStyle', 'natural')
    provider = req.get('provider', 'gemini')
    api_key = req.get('apiKey', '')
    model_name = req.get('modelName', 'gemini-3.5-flash').strip()

    if not api_key:
        return jsonify({"error": f"{provider.upper()} API Key လိုအပ်ပါသည်"}), 400

    tone_descriptions = {
        'natural': 'natural spoken conversational style suitable for movie/drama subtitles (သဘာဝကျကျ စကားပြောဟန်)',
        'formal': 'polite, elegant literary style for documentaries or official media (ယဉ်ကျေးသပ်ရပ်သော စာဟန်ပေဟန်)',
        'explaining': 'clear, easy-to-understand educational/explaining style (နားလည်လွယ်အောင် ရှင်းပြဟန်)',
        'casual': 'relaxed, youthful casual style with modern slangs (ပေါ့ပေါ့ပါးပါး လူငယ်သုံး)'
    }
    chosen_tone = tone_descriptions.get(tone_style, tone_descriptions['natural'])

    system_prompt = (
        f"You are a professional audiovisual subtitle translator. "
        f"Translate the following subtitles into {target_lang}. "
        f"Translation Style Instruction: {chosen_tone}. "
        f"Maintain natural flow, concise phrasing, and subtitle reading speed. "
        f"Output STRICTLY a valid JSON array of objects with keys 'id' (number) and 'translatedText' (string). "
        f"Do NOT omit or merge any IDs. "
        f"Example: [{{\"id\": 1, \"translatedText\": \"မင်္ဂလာပါ\"}}]"
    )

    payload_data = [{"id": s["id"], "text": s["originalText"]} for s in subtitles]

    try:
        if provider == 'gemini':
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            body = {
                "contents": [{"parts": [{"text": system_prompt}, {"text": json.dumps(payload_data)}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.25}
            }
            res = requests.post(url, headers=headers, json=body, timeout=90)
            if res.status_code != 200:
                err_data = res.json().get('error', {})
                return jsonify({"error": err_data.get('message', 'Gemini API Error')}), 400
            
            res_json = res.json()
            raw_text = res_json['candidates'][0]['content']['parts'][0]['text']
            translations = json.loads(raw_text)

        elif provider == 'groq':
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            body = {
                "model": model_name or "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload_data)}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.25
            }
            res = requests.post(url, headers=headers, json=body, timeout=90)
            if res.status_code != 200:
                return jsonify({"error": res.json().get('error', {}).get('message', 'Groq API Error')}), 400
            
            parsed = json.loads(res.json()['choices'][0]['message']['content'])
            translations = parsed if isinstance(parsed, list) else parsed.get('translations', parsed.get('subtitles', []))

        return jsonify({"translations": translations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

HTML_PAGE = """<!DOCTYPE html>
<html lang="my" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Thiri's Koko — AI Subtitle Studio Pro</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Padauk:wght@400;700&display=swap" rel="stylesheet">
  <style>
    body {
      background: radial-gradient(circle at 50% 0%, #290d1f 0%, #11050c 100%);
      font-family: 'Plus Jakarta Sans', 'Padauk', sans-serif;
      -webkit-tap-highlight-color: transparent;
    }
    .romantic-card {
      background: rgba(36, 14, 26, 0.82);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid rgba(243, 146, 189, 0.18);
    }
    .romantic-glow {
      box-shadow: 0 4px 20px -2px rgba(219, 39, 119, 0.35);
    }
    .drawer-transition {
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
  </style>
</head>
<body class="text-rose-100 min-h-full flex flex-col selection:bg-rose-500 selection:text-white pb-36">

  <!-- Header -->
  <header class="sticky top-0 z-40 romantic-card border-b border-rose-900/40 px-4 py-3 flex items-center justify-between">
    <div class="flex items-center gap-2.5">
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-rose-600 to-pink-500 flex items-center justify-center text-white font-bold shadow-md shadow-rose-900/40">
        ❤
      </div>
      <div>
        <h1 class="text-sm font-bold text-white tracking-tight">Thiri's Koko</h1>
        <p class="text-[10px] text-rose-300/80">AI Subtitle Studio Pro</p>
      </div>
    </div>

    <div class="flex items-center gap-2">
      <button onclick="toggleFindReplace()" class="p-2 rounded-xl bg-rose-900/40 border border-rose-800/50 text-rose-300 hover:text-white text-xs flex items-center gap-1 transition" title="Find & Replace">
        <span>🔍</span>
      </button>

      <button onclick="toggleMenu(true)" class="p-2 rounded-xl bg-rose-900/40 border border-rose-800/50 text-rose-200 hover:text-white transition active:scale-95" aria-label="Menu">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>
    </div>
  </header>

  <!-- Find & Replace -->
  <div id="findReplaceBar" class="hidden px-4 pt-2.5 max-w-3xl mx-auto w-full">
    <div class="romantic-card rounded-2xl p-3 border border-rose-700/50 flex flex-col gap-2">
      <div class="flex justify-between items-center text-xs text-rose-300 font-semibold border-b border-rose-900/60 pb-1.5">
        <span>🔍 Find & Replace</span>
        <button onclick="toggleFindReplace()" class="text-rose-400 hover:text-white">✕</button>
      </div>
      <div class="grid grid-cols-2 gap-2 text-xs">
        <input id="findInput" type="text" placeholder="Find..." class="bg-rose-950/90 border border-rose-800 rounded-xl p-2 text-rose-100 focus:outline-none">
        <input id="replaceInput" type="text" placeholder="Replace with..." class="bg-rose-950/90 border border-rose-800 rounded-xl p-2 text-rose-100 focus:outline-none">
      </div>
      <button onclick="applyFindReplace()" class="bg-rose-800/80 hover:bg-rose-700 text-white font-medium py-1.5 rounded-xl text-xs transition">
        Replace All
      </button>
    </div>
  </div>

  <!-- Status Pill -->
  <div id="statusPill" class="hidden mx-4 mt-2.5 p-2.5 rounded-xl text-xs romantic-card border border-rose-500/40 text-rose-200 flex items-center justify-center gap-2 shadow-sm">
    <div class="w-2 h-2 rounded-full bg-rose-400 animate-ping"></div>
    <span id="statusText">Processing...</span>
  </div>

  <!-- Workspace -->
  <main class="flex-1 px-4 py-3 max-w-3xl mx-auto w-full flex flex-col gap-3">
    
    <!-- Video Player Preview -->
    <div id="videoContainer" class="hidden romantic-card rounded-2xl p-3 border border-rose-800/40 flex flex-col gap-2">
      <div class="relative w-full aspect-video bg-black rounded-xl overflow-hidden flex items-center justify-center shadow-lg">
        <video id="videoElement" controls class="w-full h-full object-contain"></video>
        <div id="liveSubtitleBox" class="absolute bottom-6 left-3 right-3 text-center pointer-events-none hidden">
          <span id="liveSubtitleText" class="bg-black/85 text-pink-200 text-xs md:text-sm px-3.5 py-1.5 rounded-xl border border-pink-500/40 shadow-lg"></span>
        </div>
      </div>
    </div>

    <!-- Upload & Language Preferences -->
    <div class="romantic-card rounded-2xl p-4 border border-rose-800/40 flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <div class="w-9 h-9 rounded-xl bg-rose-500/10 text-rose-300 flex items-center justify-center">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
          </div>
          <div>
            <h2 class="text-xs font-semibold text-rose-100">Upload SRT / Video / Audio</h2>
            <p class="text-[10px] text-rose-400">Total Subtitles: <span id="subCount" class="font-bold text-pink-300">0</span> items</p>
          </div>
        </div>
        
        <input type="file" id="fileInput" accept=".srt,video/*,audio/*" class="hidden"/>
        <label for="fileInput" class="cursor-pointer bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white text-xs font-semibold py-2 px-3.5 rounded-xl shadow transition active:scale-95">
          Choose File
        </label>
      </div>

      <div class="pt-3 border-t border-rose-900/60 grid grid-cols-1 md:grid-cols-2 gap-2.5">
        <div>
          <label class="block text-[11px] font-medium text-rose-300 mb-1">Target Language</label>
          <select id="targetLang" onchange="handleLangChange()" class="w-full bg-rose-950/90 border border-rose-800/60 text-xs text-rose-100 rounded-xl p-2 focus:outline-none">
            <option value="Burmese">မြန်မာစာ (Burmese)</option>
            <option value="English">English</option>
            <option value="Thai">ภาษาไทย (Thai)</option>
            <option value="Japanese">日本語 (Japanese)</option>
          </select>
        </div>

        <div id="toneContainer">
          <label class="block text-[11px] font-medium text-rose-300 mb-1">စကားပြောပုံစံ / အသုံးအနှုန်းဟန်</label>
          <select id="toneStyle" class="w-full bg-rose-950/90 border border-rose-800/60 text-xs text-rose-100 rounded-xl p-2 focus:outline-none">
            <option value="natural">🗣️ သဘာဝကျ စကားပြောဟန်</option>
            <option value="formal">📖 စာဟန်ပေဟန် (ယဉ်ကျေး/တရားဝင်)</option>
            <option value="explaining">🎓 ရှင်းပြသလိုဟန် (နားလည်လွယ်)</option>
            <option value="casual">🎭 ပေါ့ပေါ့ပါးပါး လူငယ်သုံး</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Subtitle Cards List -->
    <div id="subList" class="space-y-2.5">
      <div class="romantic-card rounded-2xl p-10 text-center border border-rose-900/40 text-rose-300/80 my-2">
        <h3 class="text-sm font-semibold text-rose-100 mb-1">No Subtitles Loaded</h3>
        <p class="text-xs text-rose-400/80 max-w-xs mx-auto mb-3">
          ဗီဒီယို၊ အသံဖိုင် သို့မဟုတ် .srt ဖိုင်ကို Choose File နှိပ်၍ တင်ပေးပါ
        </p>
      </div>
    </div>
  </main>

  <!-- Sticky Bottom Controls -->
  <footer class="fixed bottom-0 left-0 right-0 z-30 romantic-card border-t border-rose-900/40 p-3 flex flex-col gap-2 max-w-lg mx-auto md:max-w-xl">
    <div class="flex items-center justify-between text-xs px-1">
      <span class="text-[11px] text-rose-300 font-medium">Translate Engine:</span>
      <div class="flex items-center gap-2">
        <label class="flex items-center gap-1.5 cursor-pointer">
          <input type="radio" name="transEngine" value="gemini" checked onchange="saveActiveEngine('gemini')" class="accent-rose-500">
          <span id="radioGeminiLabel" class="text-xs text-indigo-300 font-semibold">Gemini 3.5 Flash</span>
        </label>
        <span class="text-rose-700">|</span>
        <label class="flex items-center gap-1.5 cursor-pointer">
          <input type="radio" name="transEngine" value="groq" onchange="saveActiveEngine('groq')" class="accent-rose-500">
          <span class="text-xs text-pink-300 font-semibold">Groq (Llama 3.3)</span>
        </label>
      </div>
    </div>

    <div class="flex items-center gap-2">
      <button onclick="translateAllSafe()" id="btnTranslate" class="flex-1 bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white text-xs font-semibold py-3 px-3 rounded-xl flex items-center justify-center gap-1.5 transition romantic-glow active:scale-[0.98]">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
        </svg>
        <span>Translate All</span>
      </button>

      <button onclick="downloadOriginalSRT()" class="bg-rose-950/90 hover:bg-rose-900 text-rose-200 border border-rose-800/60 text-xs font-medium py-3 px-3 rounded-xl flex items-center gap-1 transition active:scale-[0.98]">
        <span>Original .SRT</span>
      </button>

      <button onclick="downloadTranslatedSRT()" class="bg-rose-900/80 hover:bg-rose-800 text-white border border-rose-700/60 text-xs font-medium py-3 px-3 rounded-xl flex items-center gap-1 transition active:scale-[0.98]">
        <span>Translated .SRT</span>
      </button>
    </div>
  </footer>

  <!-- Alert Modal -->
  <div id="customAlertModal" class="fixed inset-0 bg-black/70 backdrop-blur-md z-50 hidden flex items-center justify-center p-4">
    <div class="bg-rose-950 border border-rose-700/70 rounded-2xl w-full max-w-sm p-5 shadow-2xl space-y-4 romantic-card text-center">
      <div class="w-12 h-12 rounded-full bg-rose-500/20 text-rose-400 mx-auto flex items-center justify-center text-xl shadow">✨</div>
      <div>
        <h3 id="customAlertTitle" class="text-sm font-bold text-white mb-1.5">သတိပေးချက်</h3>
        <p id="customAlertMessage" class="text-xs text-rose-200/90 leading-relaxed"></p>
      </div>
      <button onclick="closeCustomAlert()" class="w-full bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 text-white font-semibold py-2.5 rounded-xl text-xs shadow transition active:scale-95">
        နားလည်ပါပြီ
      </button>
    </div>
  </div>

  <!-- Right Drawer Menu -->
  <div id="menuOverlay" onclick="toggleMenu(false)" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 hidden transition-opacity"></div>
  <aside id="menuDrawer" class="fixed top-0 right-0 bottom-0 w-80 max-w-[85vw] bg-rose-950 border-l border-rose-800/60 z-50 transform translate-x-full drawer-transition flex flex-col p-5 shadow-2xl">
    <div class="flex items-center justify-between border-b border-rose-900/60 pb-4 mb-4">
      <div class="flex items-center gap-2">
        <span class="text-rose-400 font-bold text-lg">☰</span>
        <h2 class="text-sm font-bold text-white tracking-wide">Menu</h2>
      </div>
      <button onclick="toggleMenu(false)" class="p-1 rounded-lg text-rose-400 hover:text-white">✕</button>
    </div>

    <div class="flex-1 overflow-y-auto space-y-4 text-xs">
      <button onclick="toggleMenu(false); openSettingsModal();" class="w-full bg-rose-900/50 hover:bg-rose-900/80 border border-rose-800/60 rounded-xl p-3 text-left flex items-center justify-between text-rose-100 transition">
        <div class="flex items-center gap-2.5">
          <span class="text-base">⚙</span>
          <div>
            <div class="font-semibold text-white">Settings</div>
            <div class="text-[10px] text-rose-300/70">API Keys & Model စီမံရန်</div>
          </div>
        </div>
        <span class="text-rose-400 font-bold">➔</span>
      </button>

      <div class="romantic-card rounded-2xl p-4 border border-rose-900/60 space-y-3">
        <div class="flex items-center gap-2 text-rose-300 font-semibold border-b border-rose-900/60 pb-2">
          <span>🔑</span>
          <span>API Key ယူနည်း</span>
        </div>

        <div class="space-y-1.5">
          <a href="https://console.groq.com/keys" target="_blank" class="block font-bold text-pink-300 hover:underline flex items-center justify-between bg-rose-900/30 p-2 rounded-lg border border-rose-800/40">
            <span>👉 Groq API Key ယူရန်နှိပ်ပါ</span>
            <span class="text-[10px] bg-pink-500/20 text-pink-200 px-1.5 py-0.5 rounded">Console</span>
          </a>
        </div>

        <div class="border-t border-rose-900/40 pt-2 space-y-1.5">
          <a href="https://aistudio.google.com/app/apikey" target="_blank" class="block font-bold text-indigo-300 hover:underline flex items-center justify-between bg-indigo-950/40 p-2 rounded-lg border border-indigo-800/40">
            <span>👉 Gemini API Key ယူရန်နှိပ်ပါ</span>
            <span class="text-[10px] bg-indigo-500/20 text-indigo-200 px-1.5 py-0.5 rounded">AI Studio</span>
          </a>
        </div>
      </div>
    </div>
  </aside>

  <!-- Settings Modal -->
  <div id="settingsModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
    <div class="bg-rose-950 border border-rose-800/80 rounded-2xl w-full max-w-sm p-5 shadow-2xl space-y-4">
      <div class="flex items-center justify-between border-b border-rose-900/80 pb-2.5">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <span>⚙</span> API & Model Settings
        </h3>
        <button onclick="closeSettingsModal()" class="text-rose-400 hover:text-white text-base">✕</button>
      </div>

      <div class="space-y-3 text-xs">
        <div>
          <label class="block text-[11px] font-medium text-pink-300 mb-1">Groq API Key (Whisper STT)</label>
          <input id="modalGroqKey" type="password" placeholder="gsk_..." class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>

        <div>
          <label class="block text-[11px] font-medium text-indigo-300 mb-1">Gemini API Key (Subtitle Translate)</label>
          <input id="modalGeminiKey" type="password" placeholder="AIzaSy..." class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>

        <div>
          <label class="block text-[11px] font-medium text-rose-300 mb-1">Gemini Model Name</label>
          <input id="modalGeminiModel" type="text" value="gemini-3.5-flash" class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none" placeholder="e.g. gemini-3.5-flash">
        </div>
      </div>

      <div class="pt-2">
        <button onclick="saveSettings()" class="w-full bg-gradient-to-r from-rose-600 to-pink-500 text-white font-semibold py-2.5 rounded-xl text-xs shadow transition active:scale-95">
          Save Keys & Close
        </button>
      </div>
    </div>
  </div>

  <script>
    let subtitles = [];

    const KEY_GROQ = 'thiri_koko_groq_key';
    const KEY_GEMINI = 'thiri_koko_gemini_key';
    const KEY_TRANS_ENGINE = 'thiri_koko_trans_engine';
    const KEY_GEMINI_MODEL = 'thiri_koko_gemini_model';

    const sleep = (ms) => new Promise(res => setTimeout(res, ms));

    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('modalGroqKey').value = localStorage.getItem(KEY_GROQ) || '';
      document.getElementById('modalGeminiKey').value = localStorage.getItem(KEY_GEMINI) || '';
      const savedModel = localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.5-flash';
      document.getElementById('modalGeminiModel').value = savedModel;
      document.getElementById('radioGeminiLabel').innerText = savedModel;

      const engine = localStorage.getItem(KEY_TRANS_ENGINE) || 'gemini';
      const radio = document.querySelector(`input[name="transEngine"][value="${engine}"]`);
      if (radio) radio.checked = true;

      const vid = document.getElementById('videoElement');
      vid.addEventListener('timeupdate', () => {
        const t = vid.currentTime;
        const active = subtitles.find(s => {
          const start = toSeconds(s.startTime);
          const end = toSeconds(s.endTime);
          return t >= start && t <= end;
        });

        const box = document.getElementById('liveSubtitleBox');
        const txt = document.getElementById('liveSubtitleText');
        if (active) {
          txt.innerText = active.translatedText || active.originalText;
          box.classList.remove('hidden');
        } else {
          box.classList.add('hidden');
        }
      });
    });

    function toSeconds(tStr) {
      const [hms, ms] = tStr.trim().split(/[,.]/);
      const [h, m, s] = hms.split(':').map(Number);
      return h * 3600 + m * 60 + s + (Number(ms || 0) / 1000);
    }

    function fromSeconds(sec) {
      const sClamped = Math.max(0, sec);
      const h = Math.floor(sClamped / 3600);
      const m = Math.floor((sClamped % 3600) / 60);
      const s = Math.floor(sClamped % 60);
      const ms = Math.floor((sClamped % 1) * 1000);
      const pad = (n, z = 2) => String(n).padStart(z, '0');
      return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
    }

    function adjustTimestamp(idx, deltaSec) {
      const s = subtitles[idx];
      const newStart = Math.max(0, toSeconds(s.startTime) + deltaSec);
      const newEnd = Math.max(0, toSeconds(s.endTime) + deltaSec);
      s.startTime = fromSeconds(newStart);
      s.endTime = fromSeconds(newEnd);
      renderList();
    }

    function toggleFindReplace() {
      document.getElementById('findReplaceBar').classList.toggle('hidden');
    }

    function applyFindReplace() {
      const findVal = document.getElementById('findInput').value;
      const replaceVal = document.getElementById('replaceInput').value;
      if (!findVal) return;

      let count = 0;
      subtitles.forEach(s => {
        if (s.translatedText && s.translatedText.includes(findVal)) {
          s.translatedText = s.translatedText.replaceAll(findVal, replaceVal);
          count++;
        }
      });
      renderList();
      showCustomAlert(`စာလုံးပေါင်း (${count}) နေရာကို အောင်မြင်စွာ အစားထိုးပြီးပါပြီ!`, "Find & Replace Complete");
    }

    async function retranslateSingle(idx) {
      const selectedEngine = document.querySelector('input[name="transEngine"]:checked')?.value || 'gemini';
      const groqKey = localStorage.getItem(KEY_GROQ) || '';
      const geminiKey = localStorage.getItem(KEY_GEMINI) || '';
      const geminiModel = localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.5-flash';
      const apiKey = (selectedEngine === 'gemini') ? geminiKey : groqKey;
      const modelName = (selectedEngine === 'gemini') ? geminiModel : 'llama-3.3-70b-versatile';

      if (!apiKey) {
        showCustomAlert("API Key အရင်ထည့်ပေးပါ");
        return;
      }

      setStatus(`စာကြောင်း #${subtitles[idx].id} ကို AI ပြန်ဆိုနေပါသည်...`);
      try {
        const res = await fetch('/api/translate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            subtitles: [{ id: subtitles[idx].id, originalText: subtitles[idx].originalText }],
            targetLang: document.getElementById('targetLang').value,
            toneStyle: document.getElementById('toneStyle').value,
            provider: selectedEngine,
            modelName: modelName,
            apiKey: apiKey
          })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        if (data.translations && data.translations.length > 0) {
          subtitles[idx].translatedText = data.translations[0].translatedText;
          renderList();
          setStatus("Re-translation ပြီးပါပြီ!", 2500);
        }
      } catch(err) {
        showCustomAlert("Error: " + err.message);
        clearStatus();
      }
    }

    function showCustomAlert(msg, title = "သတိပေးချက်") {
      document.getElementById('customAlertTitle').innerText = title;
      document.getElementById('customAlertMessage').innerText = msg;
      document.getElementById('customAlertModal').classList.remove('hidden');
    }

    function closeCustomAlert() {
      document.getElementById('customAlertModal').classList.add('hidden');
    }

    function handleLangChange() {
      const lang = document.getElementById('targetLang').value;
      document.getElementById('toneContainer').style.display = (lang === 'Burmese') ? 'block' : 'none';
    }

    function saveActiveEngine(engine) {
      localStorage.setItem(KEY_TRANS_ENGINE, engine);
    }

    function toggleMenu(show) {
      const drawer = document.getElementById('menuDrawer');
      const overlay = document.getElementById('menuOverlay');
      if (show) {
        overlay.classList.remove('hidden');
        setTimeout(() => drawer.classList.remove('translate-x-full'), 10);
      } else {
        drawer.classList.add('translate-x-full');
        setTimeout(() => overlay.classList.add('hidden'), 300);
      }
    }

    function openSettingsModal() {
      document.getElementById('settingsModal').classList.remove('hidden');
    }

    function closeSettingsModal() {
      document.getElementById('settingsModal').classList.add('hidden');
    }

    function saveSettings() {
      const gKey = document.getElementById('modalGroqKey').value.trim();
      const gmKey = document.getElementById('modalGeminiKey').value.trim();
      const gmModel = document.getElementById('modalGeminiModel').value.trim() || 'gemini-3.5-flash';

      localStorage.setItem(KEY_GROQ, gKey);
      localStorage.setItem(KEY_GEMINI, gmKey);
      localStorage.setItem(KEY_GEMINI_MODEL, gmModel);

      document.getElementById('radioGeminiLabel').innerText = gmModel;
      closeSettingsModal();
      setStatus("Settings မှတ်သားပြီးပါပြီ!", 3000);
    }

    function setStatus(text, duration = 0) {
      const pill = document.getElementById('statusPill');
      document.getElementById('statusText').innerText = text;
      pill.classList.remove('hidden');
      if (duration > 0) setTimeout(() => pill.classList.add('hidden'), duration);
    }

    function clearStatus() {
      document.getElementById('statusPill').classList.add('hidden');
    }

    document.getElementById('fileInput').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const ext = file.name.split('.').pop().toLowerCase();

      if (['mp4', 'mov', 'webm'].includes(ext)) {
        const vidUrl = URL.createObjectURL(file);
        const vid = document.getElementById('videoElement');
        vid.src = vidUrl;
        document.getElementById('videoContainer').classList.remove('hidden');
      }

      if (ext === 'srt') {
        const txt = await file.text();
        parseSRT(txt);
        setStatus("SRT file loaded!", 3000);
      } else {
        const groqKey = localStorage.getItem(KEY_GROQ);
        if (!groqKey) {
          showCustomAlert('Audio/Video transcribe လုပ်ရန် Groq API Key လိုအပ်ပါသည်။ Settings တွင် Key ထည့်သွင်းပေးပါခင်ဗျာ။');
          return;
        }

        setStatus("Groq Whisper ဖြင့် မူရင်းစာတန်းထိုး ထုတ်ယူနေပါသည် (ခဏစောင့်ပါ)...");
        const fd = new FormData();
        fd.append('file', file);
        fd.append('apiKey', groqKey);

        try {
          const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
          const data = await res.json();
          if (data.error) throw new Error(data.error);
          subtitles = data.subtitles;
          renderList();
          setStatus("Original Subtitles ထုတ်ယူပြီးပါပြီ!", 3000);
        } catch(err) {
          showCustomAlert(err.message, "Transcription Error");
          clearStatus();
        }
      }
    });

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
        box.innerHTML = `
          <div class="romantic-card rounded-2xl p-10 text-center border border-rose-900/40 text-rose-300/80 my-2">
            <h3 class="text-sm font-semibold text-rose-100 mb-1">No Subtitles Loaded</h3>
            <p class="text-xs text-rose-400/80">Choose File နှိပ်၍ SRT ဖိုင် သို့မဟုတ် ဗီဒီယို/အသံဖိုင် တင်ပါ</p>
          </div>`;
        return;
      }

      box.innerHTML = subtitles.map((s, idx) => `
        <div class="romantic-card rounded-2xl p-3.5 border border-rose-900/30 shadow-sm transition hover:border-rose-700/50">
          <div class="flex items-center justify-between text-[10px] font-mono text-rose-300/80 border-b border-rose-900/30 pb-1.5 mb-2">
            <div class="flex items-center gap-1.5">
              <span class="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300 font-semibold">#${s.id}</span>
              <span>${s.startTime.split(',')[0]} ➔ ${s.endTime.split(',')[0]}</span>
            </div>
            <div class="flex items-center gap-1">
              <button onclick="adjustTimestamp(${idx}, -0.5)" class="px-1.5 py-0.5 rounded bg-rose-950 border border-rose-800 hover:text-white" title="-0.5s">-0.5s</button>
              <button onclick="adjustTimestamp(${idx}, 0.5)" class="px-1.5 py-0.5 rounded bg-rose-950 border border-rose-800 hover:text-white" title="+0.5s">+0.5s</button>
            </div>
          </div>
          
          <div class="text-xs text-rose-200/90 mb-2 leading-relaxed flex justify-between gap-2">
            <span>${s.originalText}</span>
            <button onclick="retranslateSingle(${idx})" class="text-[11px] text-pink-400 hover:text-pink-300 p-1" title="AI ဖြင့် ပြန်ဆိုမည်">🔄</button>
          </div>

          <div>
            <textarea 
              placeholder="ဘာသာပြန်စာသား..."
              onchange="subtitles[${idx}].translatedText = this.value"
              rows="2"
              class="w-full bg-rose-950/80 border border-rose-800/50 focus:border-rose-400 rounded-xl p-2 text-xs text-rose-50 placeholder-rose-400/40 focus:outline-none resize-none leading-relaxed"
            >${s.translatedText}</textarea>
          </div>
        </div>
      `).join('');
    }

    // SAFE & RESILIENT TRANSLATION (Auto Resume + Quota Safe)
    async function translateAllSafe() {
      const selectedEngine = document.querySelector('input[name="transEngine"]:checked')?.value || 'gemini';
      const groqKey = localStorage.getItem(KEY_GROQ) || '';
      const geminiKey = localStorage.getItem(KEY_GEMINI) || '';
      const geminiModel = localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.5-flash';

      const apiKey = (selectedEngine === 'gemini') ? geminiKey : groqKey;
      const modelName = (selectedEngine === 'gemini') ? geminiModel : 'llama-3.3-70b-versatile';

      if (!apiKey) {
        showCustomAlert(`${selectedEngine.toUpperCase()} API Key မရှိသေးပါ။ Settings တွင် ထည့်သွင်းပေးပါခင်ဗျာ။`);
        return;
      }
      if (!subtitles.length) {
        showCustomAlert("ဘာသာပြန်ရန် Subtitle မရှိသေးပါ");
        return;
      }

      // Chunk Size 45 (Request အကြိမ်ရေ အနည်းဆုံးဖြစ်အောင် ချိန်ညှိထားခြင်း)
      const chunkSize = 45;
      const targetLang = document.getElementById('targetLang').value;
      const toneStyle = document.getElementById('toneStyle').value;

      // ဘာသာမပြန်ရသေးသော အကြောင်းများကိုသာ စစ်ဆေးလုပ်ဆောင်မည်
      const remainingItems = subtitles.filter(s => !s.translatedText);
      if (remainingItems.length === 0) {
        showCustomAlert("စာကြောင်းများ အားလုံး ဘာသာပြန်ပြီးသား ဖြစ်ပါသည်");
        return;
      }

      for (let i = 0; i < remainingItems.length; i += chunkSize) {
        const chunk = remainingItems.slice(i, i + chunkSize);
        setStatus(`ဘာသာပြန်နေပါသည်: စာကြောင်း ${chunk[0].id} မှ ${chunk[chunk.length - 1].id} အထိ...`);

        let success = false;
        let attempt = 0;

        while (!success && attempt < 5) {
          try {
            const res = await fetch('/api/translate', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                subtitles: chunk,
                targetLang: targetLang,
                toneStyle: toneStyle,
                provider: selectedEngine,
                modelName: modelName,
                apiKey: apiKey
              })
            });

            const data = await res.json();
            if (data.error) throw new Error(data.error);

            data.translations.forEach(t => {
              const item = subtitles.find(x => x.id === t.id);
              if (item) item.translatedText = t.translatedText;
            });

            renderList();
            success = true;
            await sleep(2000); // Request တစ်ခုပြီးတိုင်း ၂ စက္ကန့် pause ပေးခြင်း

          } catch(err) {
            attempt++;
            const errMsg = err.message || "";
            // Quota limit သို့မဟုတ် High demand ဖြစ်ပါက Google ကန့်သတ်ချက်ကျော်လွန်အောင် စောင့်ပေးခြင်း
            let waitSeconds = 5;
            const matchWait = errMsg.match(/retry in ([0-9.]+)s/);
            if (matchWait && matchWait[1]) {
              waitSeconds = Math.ceil(parseFloat(matchWait[1])) + 2;
            }

            if (attempt < 5) {
              setStatus(`Rate Limit ကျော်လွန်ရန် ${waitSeconds} စက္ကန့် ခေတ္တနားပြီး အလိုအလျောက် ဆက်လုပ်ပါမည် (${attempt}/5)...`);
              await sleep(waitSeconds * 1000);
            } else {
              showCustomAlert(`Error ဖြစ်ပေါ်ခဲ့ပါသည်- ${err.message}\nကျန်ရှိသော အကြောင်းများကို ထပ်မံနှိပ်၍ ဆက်လက်ဘာသာပြန်နိုင်ပါသည်`, "ခေတ္တရပ်နားပါသည်");
              clearStatus();
              return;
            }
          }
        }
      }

      setStatus("စာကြောင်းအားလုံး ဘာသာပြန်ဆိုပြီးပါပြီ!", 4000);
    }

    function downloadOriginalSRT() {
      if (!subtitles.length) return showCustomAlert("Subtitle မရှိသေးပါ");
      let out = "";
      subtitles.forEach((s, i) => {
        out += `${i + 1}\\n${s.startTime} --> ${s.endTime}\\n${s.originalText}\\n\\n`;
      });
      const blob = new Blob([out], { type: 'text/plain;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = "original_subtitles.srt";
      a.click();
    }

    function downloadTranslatedSRT() {
      if (!subtitles.length) return showCustomAlert("Subtitle မရှိသေးပါ");
      let out = "";
      subtitles.forEach((s, i) => {
        const text = s.translatedText || s.originalText;
        out += `${i + 1}\\n${s.startTime} --> ${s.endTime}\\n${text}\\n\\n`;
      });
      const blob = new Blob([out], { type: 'text/plain;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = "translated_subtitles.srt";
      a.click();
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

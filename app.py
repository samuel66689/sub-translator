import os
import re
import json
import requests
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
        files = {'file': (file.filename, file.read(), file.content_type)}
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
            err_msg = res.json().get('error', {}).get('message', 'Transcription Failed')
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
    provider = req.get('provider', 'gemini')
    api_key = req.get('apiKey', '')
    model_name = req.get('modelName', 'gemini-1.5-flash')

    if not api_key:
        return jsonify({"error": "API Key ထည့်သွင်းပေးပါ (Settings တွင် စစ်ဆေးပါ)"}), 400

    system_prompt = (
        f"You are an expert subtitle translator. Translate the following subtitles into {target_lang}. "
        f"Output strictly a valid JSON array of objects with keys 'id' and 'translatedText'. "
        f"Keep the translation natural, concise, and conversational for subtitles. "
        f"Example: [{{\"id\": 1, \"translatedText\": \"မင်္ဂလာပါ\"}}]"
    )

    payload_data = [{"id": s["id"], "text": s["originalText"]} for s in subtitles]

    try:
        if provider == 'gemini':
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            body = {
                "contents": [{"parts": [{"text": system_prompt}, {"text": json.dumps(payload_data)}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}
            }
            res = requests.post(url, headers=headers, json=body, timeout=120)
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
                "temperature": 0.2
            }
            res = requests.post(url, headers=headers, json=body, timeout=120)
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
  <title>Thiri's Koko — Subtitle Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Padauk:wght@400;700&display=swap" rel="stylesheet">
  <style>
    body {
      background: radial-gradient(circle at 50% 0%, #280d1e 0%, #12060d 100%);
      font-family: 'Plus Jakarta Sans', 'Padauk', sans-serif;
      -webkit-tap-highlight-color: transparent;
    }
    .romantic-card {
      background: rgba(36, 14, 26, 0.75);
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
<body class="text-rose-100 min-h-full flex flex-col selection:bg-rose-500 selection:text-white pb-28">

  <!-- Main Top Bar -->
  <header class="sticky top-0 z-40 romantic-card border-b border-rose-900/40 px-4 py-3 flex items-center justify-between">
    <div class="flex items-center gap-2.5">
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-rose-600 to-pink-500 flex items-center justify-center text-white font-bold shadow-md shadow-rose-900/40">
        ❤
      </div>
      <div>
        <h1 class="text-sm font-bold text-white tracking-tight">Thiri's Koko</h1>
        <p class="text-[10px] text-rose-300/80">AI Subtitle Studio</p>
      </div>
    </div>

    <div class="flex items-center gap-2">
      <!-- Target Language -->
      <select id="targetLang" class="bg-rose-950/80 border border-rose-800/60 text-xs text-rose-100 rounded-xl px-2.5 py-1.5 focus:outline-none focus:border-rose-400">
        <option value="Burmese">မြန်မာစာ (Burmese)</option>
        <option value="English">English</option>
        <option value="Thai">ภาษาไทย (Thai)</option>
        <option value="Japanese">日本語 (Japanese)</option>
        <option value="Chinese">中文 (Chinese)</option>
      </select>

      <!-- 3-Line Hamburger Menu Button -->
      <button onclick="toggleMenu(true)" class="p-2 rounded-xl bg-rose-900/40 border border-rose-800/50 text-rose-200 hover:text-white hover:bg-rose-800/60 transition active:scale-95" aria-label="Menu">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>
    </div>
  </header>

  <!-- Status Notification Banner -->
  <div id="statusPill" class="hidden mx-4 mt-2.5 p-2.5 rounded-xl text-xs romantic-card border border-rose-500/40 text-rose-200 flex items-center justify-center gap-2 shadow-sm">
    <div class="w-2 h-2 rounded-full bg-rose-400 animate-ping"></div>
    <span id="statusText">Processing...</span>
  </div>

  <!-- Main Subtitle Editor Container -->
  <main class="flex-1 px-4 py-3 max-w-3xl mx-auto w-full flex flex-col gap-3">
    
    <!-- Quick Upload Bar -->
    <div class="romantic-card rounded-2xl p-3 border border-rose-800/40 flex items-center justify-between gap-3">
      <div class="flex items-center gap-2.5">
        <div class="w-9 h-9 rounded-xl bg-rose-500/10 text-rose-300 flex items-center justify-center">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
          </svg>
        </div>
        <div>
          <h2 class="text-xs font-semibold text-rose-100">Upload File (.srt, .mp4, .mp3)</h2>
          <p class="text-[10px] text-rose-400">Total: <span id="subCount">0</span> subtitles</p>
        </div>
      </div>
      
      <input type="file" id="fileInput" accept=".srt,video/*,audio/*" class="hidden"/>
      <label for="fileInput" class="cursor-pointer bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white text-xs font-semibold py-2 px-3.5 rounded-xl shadow transition active:scale-95">
        Choose File
      </label>
    </div>

    <!-- Subtitle Cards List -->
    <div id="subList" class="space-y-2.5">
      <div class="romantic-card rounded-2xl p-10 text-center border border-rose-900/40 text-rose-300/80 my-4">
        <div class="w-12 h-12 rounded-full bg-rose-500/10 text-rose-400 mx-auto flex items-center justify-center mb-3">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
        </div>
        <h3 class="text-sm font-semibold text-rose-100 mb-1">No Subtitles Loaded</h3>
        <p class="text-xs text-rose-400/80 max-w-xs mx-auto mb-3">
          ဖုန်းထဲက SRT စာတန်းထိုးဖိုင် (သို့မဟုတ် ဗီဒီယို/အသံဖိုင်) ကို Choose File နှိပ်ပြီး စတင်ထည့်သွင်းပါ
        </p>
        <button onclick="openSettingsModal()" class="text-xs text-rose-300 bg-rose-900/40 border border-rose-800/60 px-3.5 py-1.5 rounded-xl hover:bg-rose-800/50 transition">
          ⚙ API Key ထည့်ရန် နှိပ်ပါ
        </button>
      </div>
    </div>
  </main>

  <!-- Sticky Bottom Action Bar -->
  <footer class="fixed bottom-0 left-0 right-0 z-30 romantic-card border-t border-rose-900/40 p-3 flex items-center gap-2 max-w-md mx-auto">
    <button onclick="translateAll()" id="btnTranslate" class="flex-1 bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white text-xs font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition romantic-glow active:scale-[0.98]">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"/>
      </svg>
      <span>Translate All</span>
    </button>

    <button onclick="downloadSRT()" class="bg-rose-950/90 hover:bg-rose-900 text-rose-200 border border-rose-800/60 text-xs font-medium py-3 px-4 rounded-xl flex items-center gap-1.5 transition active:scale-[0.98]">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
      </svg>
      <span>Export .SRT</span>
    </button>
  </footer>

  <!-- ================= RIGHT SIDE DRAWER MENU ================= -->
  <div id="menuOverlay" onclick="toggleMenu(false)" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 hidden transition-opacity"></div>
  <aside id="menuDrawer" class="fixed top-0 right-0 bottom-0 w-80 max-w-[85vw] bg-rose-950 border-l border-rose-800/60 z-50 transform translate-x-full drawer-transition flex flex-col p-5 shadow-2xl">
    <div class="flex items-center justify-between border-b border-rose-900/60 pb-4 mb-4">
      <div class="flex items-center gap-2">
        <span class="text-rose-400 font-bold text-lg">☰</span>
        <h2 class="text-sm font-bold text-white tracking-wide">Menu</h2>
      </div>
      <button onclick="toggleMenu(false)" class="p-1 rounded-lg text-rose-400 hover:text-white">
        ✕
      </button>
    </div>

    <div class="flex-1 overflow-y-auto space-y-4 text-xs">
      
      <!-- Settings Tab Button in Menu -->
      <button onclick="toggleMenu(false); openSettingsModal();" class="w-full bg-rose-900/50 hover:bg-rose-900/80 border border-rose-800/60 rounded-xl p-3 text-left flex items-center justify-between text-rose-100 transition">
        <div class="flex items-center gap-2.5">
          <span class="text-base">⚙</span>
          <div>
            <div class="font-semibold text-white">Settings</div>
            <div class="text-[10px] text-rose-300/70">Groq & Gemini API Keys ထည့်ရန်</div>
          </div>
        </div>
        <span class="text-rose-400 font-bold">➔</span>
      </button>

      <!-- Guide Section: API Key ယူနည်း -->
      <div class="romantic-card rounded-2xl p-4 border border-rose-900/60 space-y-3">
        <div class="flex items-center gap-2 text-rose-300 font-semibold border-b border-rose-900/60 pb-2">
          <span>🔑</span>
          <span>API Key ယူနည်း (အခမဲ့)</span>
        </div>

        <!-- Groq Key Link & Guide -->
        <div class="space-y-1.5">
          <a href="https://console.groq.com/keys" target="_blank" class="block font-bold text-pink-300 hover:underline flex items-center justify-between bg-rose-900/30 p-2 rounded-lg border border-rose-800/40">
            <span>👉 Groq API Key ယူရန်နှိပ်ပါ</span>
            <span class="text-[10px] bg-pink-500/20 text-pink-200 px-1.5 py-0.5 rounded">Website</span>
          </a>
          <p class="text-[11px] text-rose-200/80 leading-relaxed pl-1">
            <strong>ယူနည်း:</strong> Link ကို နှိပ်ပြီး Google Account ဖြင့် Sign In ဝင်ပါ။ <strong>"Create API Key"</strong> ကို နှိပ်ပြီး Key ကို Copy ကူးကာ ယူပါ။ (အလွန်မြန်ပြီး အသံဖိုင် transcribe လုပ်ရန် မဖြစ်မနေ လိုအပ်ပါသည်)
          </p>
        </div>

        <div class="border-t border-rose-900/40 pt-2 space-y-1.5">
          <!-- Gemini Key Link & Guide -->
          <a href="https://aistudio.google.com/app/apikey" target="_blank" class="block font-bold text-indigo-300 hover:underline flex items-center justify-between bg-indigo-950/40 p-2 rounded-lg border border-indigo-800/40">
            <span>👉 Gemini API Key ယူရန်နှိပ်ပါ</span>
            <span class="text-[10px] bg-indigo-500/20 text-indigo-200 px-1.5 py-0.5 rounded">Google AI</span>
          </a>
          <p class="text-[11px] text-rose-200/80 leading-relaxed pl-1">
            <strong>ယူနည်း:</strong> Link ကို နှိပ်ပြီး Google Account ဖြင့် ဝင်ပါ။ <strong>"Create API key in new project"</strong> ကို နှိပ်၍ Key ကို Copy ကူးယူပါ။
          </p>
        </div>
      </div>

      <div class="text-[10px] text-rose-400/60 text-center pt-2">
        Thiri's Koko Studio • Designed with ❤
      </div>
    </div>
  </aside>

  <!-- ================= SETTINGS MODAL (AUTO-SAVE) ================= -->
  <div id="settingsModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
    <div class="bg-rose-950 border border-rose-800/80 rounded-2xl w-full max-w-sm p-5 shadow-2xl space-y-4">
      
      <div class="flex items-center justify-between border-b border-rose-900/80 pb-2.5">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <span>⚙</span> API Settings & Engine
        </h3>
        <button onclick="closeSettingsModal()" class="text-rose-400 hover:text-white text-base">✕</button>
      </div>

      <div class="space-y-3.5 text-xs">
        <!-- AI Engine Choice -->
        <div>
          <label class="block text-[11px] font-medium text-rose-300 mb-1">Active AI Engine (ဘာသာပြန်ရာတွင် သုံးမည့် Engine)</label>
          <select id="modalProvider" onchange="handleModalProviderChange()" class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2.5 text-rose-100 focus:outline-none">
            <option value="gemini">Google Gemini</option>
            <option value="groq">Groq (Llama 3.3)</option>
          </select>
        </div>

        <!-- Model Name -->
        <div>
          <label class="block text-[11px] font-medium text-rose-300 mb-1">Model Name</label>
          <input id="modalModelName" type="text" class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>

        <!-- Groq API Key Input Field -->
        <div class="pt-1 border-t border-rose-900/60">
          <div class="flex justify-between items-center mb-1">
            <label class="text-[11px] font-medium text-pink-300">Groq API Key</label>
            <a href="https://console.groq.com/keys" target="_blank" class="text-[10px] text-pink-400 underline">Key ယူရန် ↗</a>
          </div>
          <input id="modalGroqKey" type="password" placeholder="gsk_..." class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
          <p class="text-[10px] text-rose-400/70 mt-1">* Audio/Video မှ စာတန်းထိုးထုတ်ရန် မဖြစ်မနေ လိုအပ်ပါသည်</p>
        </div>

        <!-- Gemini API Key Input Field -->
        <div class="pt-1 border-t border-rose-900/60">
          <div class="flex justify-between items-center mb-1">
            <label class="text-[11px] font-medium text-indigo-300">Gemini API Key</label>
            <a href="https://aistudio.google.com/app/apikey" target="_blank" class="text-[10px] text-indigo-400 underline">Key ယူရန် ↗</a>
          </div>
          <input id="modalGeminiKey" type="password" placeholder="AIzaSy..." class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>
      </div>

      <div class="pt-2">
        <button onclick="saveSettings()" class="w-full bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white font-semibold py-2.5 rounded-xl text-xs shadow transition active:scale-95">
          Save Keys & Close (အမြဲမှတ်ထားမည်)
        </button>
      </div>

    </div>
  </div>

  <script>
    let subtitles = [];

    // Local Storage Keys
    const KEY_GROQ = 'thiri_koko_groq_key';
    const KEY_GEMINI = 'thiri_koko_gemini_key';
    const KEY_PROVIDER = 'thiri_koko_provider';
    const KEY_MODEL = 'thiri_koko_model';

    // Load saved settings on startup
    window.addEventListener('DOMContentLoaded', () => {
      const savedGroq = localStorage.getItem(KEY_GROQ) || '';
      const savedGemini = localStorage.getItem(KEY_GEMINI) || '';
      const savedProvider = localStorage.getItem(KEY_PROVIDER) || 'gemini';
      const savedModel = localStorage.getItem(KEY_MODEL) || 'gemini-1.5-flash';

      document.getElementById('modalGroqKey').value = savedGroq;
      document.getElementById('modalGeminiKey').value = savedGemini;
      document.getElementById('modalProvider').value = savedProvider;
      document.getElementById('modalModelName').value = savedModel;
    });

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

    function handleModalProviderChange() {
      const p = document.getElementById('modalProvider').value;
      document.getElementById('modalModelName').value = (p === 'gemini') ? 'gemini-1.5-flash' : 'llama-3.3-70b-versatile';
    }

    function saveSettings() {
      const groqKey = document.getElementById('modalGroqKey').value.trim();
      const geminiKey = document.getElementById('modalGeminiKey').value.trim();
      const provider = document.getElementById('modalProvider').value;
      const model = document.getElementById('modalModelName').value.trim();

      localStorage.setItem(KEY_GROQ, groqKey);
      localStorage.setItem(KEY_GEMINI, geminiKey);
      localStorage.setItem(KEY_PROVIDER, provider);
      localStorage.setItem(KEY_MODEL, model);

      closeSettingsModal();
      setStatus("Settings & API Keys သိမ်းဆည်းပြီးပါပြီ!", 3000);
    }

    function setStatus(text, duration = 0) {
      const pill = document.getElementById('statusPill');
      document.getElementById('statusText').innerText = text;
      pill.classList.remove('hidden');
      if (duration > 0) {
        setTimeout(() => pill.classList.add('hidden'), duration);
      }
    }

    function clearStatus() {
      document.getElementById('statusPill').classList.add('hidden');
    }

    // File Upload Handler
    document.getElementById('fileInput').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const ext = file.name.split('.').pop().toLowerCase();

      if (ext === 'srt') {
        const txt = await file.text();
        parseSRT(txt);
        setStatus("SRT file loaded!", 3000);
      } else {
        const groqKey = localStorage.getItem(KEY_GROQ);
        if (!groqKey) {
          alert('Audio/Video transcribe လုပ်ရန် Groq API Key လိုအပ်ပါသည်။ Menu -> Settings တွင် Groq API Key အရင်ထည့်ပေးပါခင်ဗျာ။');
          openSettingsModal();
          return;
        }

        setStatus("Whisper AI ဖြင့် အသံဖိုင်မှ စာတန်းထိုး ထုတ်ယူနေပါသည် (ခဏစောင့်ပါ)...");
        const fd = new FormData();
        fd.append('file', file);
        fd.append('apiKey', groqKey);

        try {
          const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
          const data = await res.json();
          if (data.error) throw new Error(data.error);
          subtitles = data.subtitles;
          renderList();
          setStatus("Transcription ပြီးစီးပါပြီ!", 3000);
        } catch(err) {
          alert("Error: " + err.message);
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
          <div class="romantic-card rounded-2xl p-10 text-center border border-rose-900/40 text-rose-300/80 my-4">
            <h3 class="text-sm font-semibold text-rose-100 mb-1">No Subtitles Loaded</h3>
            <p class="text-xs text-rose-400/80">Choose File နှိပ်၍ SRT ဖိုင် သို့မဟုတ် ဗီဒီယို/အသံဖိုင် တင်ပါ</p>
          </div>`;
        return;
      }

      box.innerHTML = subtitles.map((s, idx) => `
        <div class="romantic-card rounded-2xl p-3.5 border border-rose-900/30 shadow-sm transition hover:border-rose-700/50">
          <div class="flex items-center justify-between text-[10px] font-mono text-rose-300/80 border-b border-rose-900/30 pb-1.5 mb-2">
            <span class="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300 font-semibold">#${s.id}</span>
            <span>${s.startTime.split(',')[0]} ➔ ${s.endTime.split(',')[0]}</span>
          </div>
          
          <div class="text-xs text-rose-200/90 mb-2 leading-relaxed">
            ${s.originalText}
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

    async function translateAll() {
      const provider = localStorage.getItem(KEY_PROVIDER) || 'gemini';
      const groqKey = localStorage.getItem(KEY_GROQ) || '';
      const geminiKey = localStorage.getItem(KEY_GEMINI) || '';
      const model = localStorage.getItem(KEY_MODEL) || (provider === 'gemini' ? 'gemini-1.5-flash' : 'llama-3.3-70b-versatile');

      const activeKey = provider === 'gemini' ? geminiKey : groqKey;

      if (!activeKey) {
        alert(`${provider.toUpperCase()} API Key မရှိသေးပါ။ Menu -> Settings ထဲတွင် API Key အရင်ထည့်ပေးပါခင်ဗျာ။`);
        openSettingsModal();
        return;
      }
      if (!subtitles.length) {
        alert("Translate လုပ်ရန် Subtitle မရှိသေးပါ");
        return;
      }

      setStatus("AI ဖြင့် ဘာသာပြန်ဆိုနေပါသည်...");
      const chunkSize = 20;

      for (let i = 0; i < subtitles.length; i += chunkSize) {
        const chunk = subtitles.slice(i, i + chunkSize);
        setStatus(`ဘာသာပြန်နေပါသည်: ${i + 1} မှ ${Math.min(i + chunkSize, subtitles.length)} အထိ...`);

        try {
          const res = await fetch('/api/translate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              subtitles: chunk,
              targetLang: document.getElementById('targetLang').value,
              provider: provider,
              modelName: model,
              apiKey: activeKey
            })
          });
          const data = await res.json();
          if (data.error) throw new Error(data.error);

          data.translations.forEach(t => {
            const item = subtitles.find(x => x.id === t.id);
            if (item) item.translatedText = t.translatedText;
          });
          renderList();
        } catch(err) {
          alert("Error: " + err.message);
          break;
        }
      }
      setStatus("ဘာသာပြန်ဆိုခြင်း ပြီးစီးပါပြီ!", 4000);
    }

    function downloadSRT() {
      if (!subtitles.length) {
        alert("Export လုပ်ရန် Subtitle မရှိသေးပါ");
        return;
      }
      let out = "";
      subtitles.forEach((s, i) => {
        const text = s.translatedText || s.originalText;
        out += `${i + 1}\\n${s.startTime} --> ${s.endTime}\\n${text}\\n\\n`;
      });
      const blob = new Blob([out], { type: 'text/plain;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = "Thiri_Koko_subtitles.srt";
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

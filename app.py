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
        
        res = requests.post('https://api.groq.com/openai/v1/audio/transcriptions', 
                            headers=headers, files=files, data=data, timeout=300)
        
        if res.status_code != 200:
            return jsonify({"error": res.json().get('error', {}).get('message', 'Audio Transcription Failed')}), 400

        result = res.json()
        items = []
        for idx, seg in enumerate(result.get('segments', []), 1):
            def fmt(s):
                h, m, sec = int(s // 3600), int((s % 3600) // 60), s % 60
                return f"{h:02}:{m:02}:{int(sec):02},{int((sec % 1) * 1000):03}"
            
            items.append({
                "id": idx,
                "startTime": fmt(seg['start']),
                "endTime": fmt(seg['end']),
                "originalText": seg['text'].strip(),
                "translatedText": ""
            })
        return jsonify({"subtitles": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    req = request.json
    subtitles = req.get('subtitles', [])
    target_lang = req.get('targetLang', 'Burmese')
    provider = req.get('provider', 'gemini')
    api_key = req.get('apiKey', '')
    model_name = req.get('modelName', 'gemini-3.6-flash')

    if not api_key:
        return jsonify({"error": "API Key ထည့်သွင်းပေးပါ"}), 400

    system_prompt = f"""You are a professional subtitle translator.
Translate the following subtitle items into {target_lang}.
Output STRICTLY a valid JSON array of objects with keys: "id" (number) and "translatedText" (string).
Do NOT modify IDs. Keep translations concise, natural, and accurately matched to spoken tone for subtitles.
Example format: [{{"id": 1, "translatedText": "မင်္ဂလာပါ"}}]"""

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="my" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Thiri's Koko — Subtitle Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Padauk:wght@400;700&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['"Plus Jakarta Sans"', 'Padauk', 'sans-serif'],
          },
          colors: {
            rosewine: {
              950: '#14080e',
              900: '#230f19',
              850: '#321625',
              800: '#431d33',
              700: '#67284d',
              500: '#b83b74',
              400: '#df5894',
              300: '#f392bd',
              200: '#fbcfe3',
              100: '#fdf2f7',
            }
          }
        }
      }
    }
  </script>
  <style>
    body {
      background: radial-gradient(circle at 50% 0%, #2b1120 0%, #14080e 100%);
      -webkit-tap-highlight-color: transparent;
    }
    .romantic-glass {
      background: rgba(35, 15, 25, 0.7);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid rgba(243, 146, 189, 0.15);
    }
    .romantic-glow {
      box-shadow: 0 4px 20px -2px rgba(184, 59, 116, 0.35);
    }
  </style>
</head>
<body class="text-rose-100 min-h-full flex flex-col font-sans selection:bg-rose-500 selection:text-white pb-24">

  <!-- Romantic Header -->
  <header class="sticky top-0 z-40 romantic-glass border-b border-rose-900/40 px-4 py-3 flex items-center justify-between">
    <div class="flex items-center gap-2.5">
      <div class="w-9 h-9 rounded-2xl bg-gradient-to-tr from-rose-600 via-rose-500 to-pink-400 flex items-center justify-center text-white shadow-lg shadow-rose-600/30">
        <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 fill-current" viewBox="0 0 24 24">
          <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
        </svg>
      </div>
      <div>
        <h1 class="text-base font-bold tracking-tight text-white flex items-center gap-1.5">
          Thiri's Koko
          <span class="text-[9px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30">Studio</span>
        </h1>
        <p class="text-[11px] text-rose-300/70">Pure Subtitle Magic with AI</p>
      </div>
    </div>

    <!-- Target Language Selection -->
    <div class="flex items-center">
      <select id="targetLang" class="bg-rosewine-900/80 border border-rose-700/40 text-xs text-rose-100 rounded-xl px-2.5 py-1.5 focus:outline-none focus:border-rose-400">
        <option value="Burmese">မြန်မာစာ (Burmese)</option>
        <option value="English">English</option>
        <option value="Thai">ภาษาไทย (Thai)</option>
        <option value="Japanese">日本語 (Japanese)</option>
      </select>
    </div>
  </header>

  <!-- Mobile Segmented Tabs -->
  <div class="px-4 pt-3 pb-1 md:hidden">
    <div class="grid grid-cols-2 p-1 rounded-xl bg-rosewine-900/60 border border-rose-800/40 text-xs font-medium text-rose-300">
      <button onclick="switchTab('subtitles')" id="tabBtnSubtitles" class="py-2 rounded-lg bg-rose-600 text-white shadow-sm font-semibold transition">
        Subtitles List (<span id="subCountBadge">0</span>)
      </button>
      <button onclick="switchTab('config')" id="tabBtnConfig" class="py-2 rounded-lg text-rose-300 hover:text-white transition">
        AI Settings & Upload
      </button>
    </div>
  </div>

  <!-- Status Notification Pill -->
  <div id="statusPill" class="hidden mx-4 mt-2 p-2.5 rounded-xl text-xs flex items-center justify-between romantic-glass border border-rose-500/30 text-rose-200">
    <div class="flex items-center gap-2">
      <div class="w-2 h-2 rounded-full bg-rose-400 animate-ping"></div>
      <span id="statusText">Processing...</span>
    </div>
  </div>

  <!-- Main Responsive Container -->
  <div class="flex-1 px-4 py-3 max-w-5xl mx-auto w-full grid grid-cols-1 md:grid-cols-12 gap-4 items-start">
    
    <!-- LEFT PANEL: Settings & File Upload (Tab on Mobile) -->
    <section id="configSection" class="hidden md:flex md:col-span-4 flex-col gap-3.5 romantic-glass rounded-2xl p-4 border border-rose-800/40">
      <div class="flex items-center justify-between border-b border-rose-800/40 pb-2">
        <span class="text-xs font-semibold uppercase tracking-wider text-rose-300">Configuration</span>
        <span class="text-[10px] text-rose-400">BYOK Safe</span>
      </div>

      <!-- File Upload Card -->
      <div class="border border-dashed border-rose-500/40 hover:border-rose-400 rounded-xl p-3.5 text-center bg-rosewine-900/30 transition">
        <input type="file" id="fileInput" accept=".srt,video/*,audio/*" class="hidden"/>
        <label for="fileInput" class="cursor-pointer flex flex-col items-center gap-1.5">
          <div class="w-10 h-10 rounded-full bg-rose-500/10 flex items-center justify-center text-rose-400">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
          </div>
          <span class="text-xs font-medium text-rose-100">Upload SRT, Video, or Audio</span>
          <span class="text-[10px] text-rose-400/80">.srt, .mp4, .mp3, .wav (Max 500MB)</span>
        </label>
      </div>

      <!-- AI Provider -->
      <div>
        <label class="block text-[11px] font-medium text-rose-300 mb-1">AI Engine</label>
        <select id="provider" onchange="toggleProvider()" class="w-full bg-rosewine-900 border border-rose-700/40 text-xs text-rose-100 rounded-xl p-2.5 focus:outline-none focus:border-rose-400">
          <option value="gemini">Google Gemini (Recommended)</option>
          <option value="groq">Groq (Ultra-Fast Llama & Whisper)</option>
        </select>
      </div>

      <!-- Model Name -->
      <div>
        <label class="block text-[11px] font-medium text-rose-300 mb-1">Model Name</label>
        <input id="modelName" type="text" value="gemini-3.6-flash" class="w-full bg-rosewine-900 border border-rose-700/40 text-xs text-rose-100 rounded-xl p-2.5 focus:outline-none focus:border-rose-400 font-mono">
      </div>

      <!-- API Key -->
      <div>
        <label class="block text-[11px] font-medium text-rose-300 mb-1">API Key</label>
        <input id="apiKey" type="password" placeholder="Paste Gemini / Groq API Key" class="w-full bg-rosewine-900 border border-rose-700/40 text-xs text-rose-100 rounded-xl p-2.5 focus:outline-none focus:border-rose-400 font-mono">
      </div>

      <div class="text-[10px] text-rose-400/70 leading-relaxed bg-rose-950/40 p-2.5 rounded-xl border border-rose-900/30">
        💡 ဗီဒီယို/အသံဖိုင် တင်မည်ဆိုပါက Groq Engine ကိုရွေးပြီး Groq API Key ထည့်ပေးပါ (Whisper AI ဖြင့် Subtitle ထုတ်ပေးပါသည်)။
      </div>
    </section>

    <!-- RIGHT PANEL: Subtitles Cards List (Tab on Mobile) -->
    <section id="subtitlesSection" class="md:col-span-8 flex flex-col gap-2.5">
      <div id="subList" class="space-y-2.5">
        <!-- Empty State -->
        <div class="romantic-glass rounded-2xl p-8 text-center border border-rose-800/30 text-rose-300/80 my-4">
          <div class="w-12 h-12 rounded-full bg-rose-500/10 text-rose-400 mx-auto flex items-center justify-center mb-2.5">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
            </svg>
          </div>
          <h3 class="text-sm font-semibold text-rose-100 mb-1">No Subtitles Loaded</h3>
          <p class="text-xs text-rose-400/80 max-w-xs mx-auto mb-3">
            ဖုန်းထဲက SRT စာတန်းထိုးဖိုင် (သို့မဟုတ် Audio/Video) ကို Upload လုပ်ပြီး စတင်လိုက်ပါ
          </p>
          <button onclick="switchTab('config')" class="md:hidden text-xs bg-rose-600 hover:bg-rose-500 text-white font-medium px-4 py-2 rounded-xl transition">
            Go to Upload & Settings
          </button>
        </div>
      </div>
    </section>
  </div>

  <!-- Mobile Sticky Bottom Action Bar -->
  <footer class="fixed bottom-0 left-0 right-0 z-40 romantic-glass border-t border-rose-900/40 p-3 flex items-center gap-2 max-w-lg mx-auto md:max-w-none">
    <button onclick="translateAll()" id="btnTranslate" class="flex-1 bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white text-xs font-semibold py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition romantic-glow active:scale-[0.98]">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"/>
      </svg>
      <span>Translate All</span>
    </button>

    <button onclick="downloadSRT()" class="bg-rosewine-900/90 hover:bg-rosewine-800 text-rose-200 border border-rose-700/50 text-xs font-medium py-3 px-4 rounded-xl flex items-center gap-1.5 transition active:scale-[0.98]">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
      </svg>
      <span>Export .SRT</span>
    </button>
  </footer>

  <script>
    let subtitles = [];

    function switchTab(tab) {
      const subSection = document.getElementById('subtitlesSection');
      const configSection = document.getElementById('configSection');
      const tabSub = document.getElementById('tabBtnSubtitles');
      const tabCfg = document.getElementById('tabBtnConfig');

      if (tab === 'subtitles') {
        subSection.classList.remove('hidden');
        configSection.classList.add('hidden');
        tabSub.classList.add('bg-rose-600', 'text-white', 'font-semibold');
        tabSub.classList.remove('text-rose-300');
        tabCfg.classList.remove('bg-rose-600', 'text-white', 'font-semibold');
        tabCfg.classList.add('text-rose-300');
      } else {
        configSection.classList.remove('hidden');
        subSection.classList.add('hidden');
        tabCfg.classList.add('bg-rose-600', 'text-white', 'font-semibold');
        tabCfg.classList.remove('text-rose-300');
        tabSub.classList.remove('bg-rose-600', 'text-white', 'font-semibold');
        tabSub.classList.add('text-rose-300');
      }
    }

    function toggleProvider() {
      const p = document.getElementById('provider').value;
      const modelInput = document.getElementById('modelName');
      modelInput.value = (p === 'gemini') ? 'gemini-3.6-flash' : 'llama-3.3-70b-versatile';
    }

    function showStatus(text, showPing = true) {
      const pill = document.getElementById('statusPill');
      const txt = document.getElementById('statusText');
      txt.innerText = text;
      pill.classList.remove('hidden');
    }

    function hideStatus() {
      document.getElementById('statusPill').classList.add('hidden');
    }

    // File Upload Handler
    document.getElementById('fileInput').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const ext = file.name.split('.').pop().toLowerCase();

      if (ext === 'srt') {
        const text = await file.text();
        parseSRTClient(text);
        switchTab('subtitles');
      } else {
        const apiKey = document.getElementById('apiKey').value;
        const provider = document.getElementById('provider').value;
        if (provider !== 'groq' || !apiKey) {
          alert('Audio/Video ဖိုင်များအတွက် AI Engine နေရာတွင် Groq ကိုရွေးပြီး Groq API Key ထည့်ပေးပါ');
          switchTab('config');
          return;
        }

        showStatus("Whisper AI ဖြင့် အသံဖိုင်မှ စာတန်းထိုး ထုတ်ယူနေပါသည်...");

        const formData = new FormData();
        formData.append('file', file);
        formData.append('apiKey', apiKey);

        try {
          const res = await fetch('/api/transcribe', { method: 'POST', body: formData });
          const data = await res.json();
          if (data.error) throw new Error(data.error);
          subtitles = data.subtitles;
          renderList();
          showStatus("Transcription အောင်မြင်ပါသည်!");
          setTimeout(hideStatus, 3000);
          switchTab('subtitles');
        } catch(err) {
          alert("Error: " + err.message);
          hideStatus();
        }
      }
    });

    function parseSRTClient(text) {
      const blocks = text.trim().replace(/\\r\\n/g, '\\n').split(/\\n\\s*\\n/);
      subtitles = [];
      blocks.forEach((b, i) => {
        const lines = b.trim().split('\\n');
        if (lines.length >= 2) {
          const timeIdx = /^\\d+$/.test(lines[0].trim()) ? 1 : 0;
          if (lines[timeIdx] && lines[timeIdx].includes('-->')) {
            const [s, e] = lines[timeIdx].split('-->').map(x => x.trim());
            subtitles.push({
              id: i + 1,
              startTime: s,
              endTime: e,
              originalText: lines.slice(timeIdx + 1).join('\\n'),
              translatedText: ''
            });
          }
        }
      });
      renderList();
    }

    function renderList() {
      const box = document.getElementById('subList');
      document.getElementById('subCountBadge').innerText = subtitles.length;

      if (!subtitles.length) {
        box.innerHTML = `
          <div class="romantic-glass rounded-2xl p-8 text-center border border-rose-800/30 text-rose-300/80 my-4">
            <h3 class="text-sm font-semibold text-rose-100 mb-1">No Subtitles Loaded</h3>
            <p class="text-xs text-rose-400/80">Upload ပြုလုပ်ရန် Settings & Upl

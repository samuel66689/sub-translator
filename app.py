import json
import requests
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    file = request.files.get('file')
    api_key = request.form.get('apiKey')
    if not file or not api_key:
        return jsonify({"error": "Video/Audio transcribe လုပ်ရန် Groq API Key လိုအပ်ပါသည်"}), 400

    try:
        files = {'file': (file.filename, file.read(), file.content_type or 'application/octet-stream')}
        data = {'model': 'whisper-large-v3-turbo', 'response_format': 'verbose_json'}
        headers = {'Authorization': f'Bearer {api_key}'}
        
        res = requests.post(
            'https://api.groq.com/openai/v1/audio/transcriptions',
            headers=headers,
            files=files,
            data=data,
            timeout=180
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
    api_key = req.get('apiKey', '')
    model_name = req.get('modelName', 'gemini-3.8-flash').strip()

    if not api_key:
        return jsonify({"error": "Gemini API Key လိုအပ်ပါသည်"}), 400

    tone_descriptions = {
        'natural': 'natural spoken conversational style suitable for movie/drama subtitles (သဘာဝကျကျ စကားပြောဟန်)',
        'formal': 'polite, elegant literary style for documentaries (ယဉ်ကျေးသပ်ရပ်သော စာဟန်ပေဟန်)',
        'explaining': 'clear, educational style (နားလည်လွယ်အောင် ရှင်းပြဟန်)',
        'casual': 'relaxed youthful casual style with modern slangs (ပေါ့ပေါ့ပါးပါး လူငယ်သုံး)'
    }
    chosen_tone = tone_descriptions.get(tone_style, tone_descriptions['natural'])

    system_prompt = (
        f"You are a professional audiovisual subtitle translator. "
        f"Translate the following subtitles into {target_lang}. "
        f"Style: {chosen_tone}. Output STRICTLY a valid JSON array of objects with keys 'id' and 'translatedText'. "
        f"Do NOT omit any IDs. Example: [{{\"id\": 1, \"translatedText\": \"မင်္ဂလာပါ\"}}]"
    )

    payload_data = [{"id": s["id"], "text": s["originalText"]} for s in subtitles]

    try:
        clean_model = model_name.replace('models/', '')
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        body = {
            "contents": [{"parts": [{"text": system_prompt}, {"text": json.dumps(payload_data)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.25
            }
        }
        res = requests.post(url, headers=headers, json=body, timeout=90)
        if res.status_code != 200:
            err_data = res.json().get('error', {})
            return jsonify({"error": err_data.get('message', f'Gemini Error ({res.status_code})')}), res.status_code
        
        res_json = res.json()
        raw_text = res_json['candidates'][0]['content']['parts'][0]['text']
        translations = json.loads(raw_text)

        return jsonify({"translations": translations if isinstance(translations, list) else []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

HTML_PAGE = """<!DOCTYPE html>
<html lang="my" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Thiri's Koko — AI Subtitle Studio Pro</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Padauk:wght@400;700&display=swap" rel="stylesheet">
  <style>
    body {
      background: radial-gradient(circle at 50% 0%, #290d1f 0%, #11050c 100%);
      font-family: 'Plus Jakarta Sans', 'Padauk', sans-serif;
    }
    .romantic-card {
      background: rgba(36, 14, 26, 0.85);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(243, 146, 189, 0.18);
    }
    .romantic-glow {
      box-shadow: 0 4px 20px -2px rgba(219, 39, 119, 0.35);
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
      <button onclick="toggleMenu(true)" class="p-2 rounded-xl bg-rose-900/40 border border-rose-800/50 text-rose-200 hover:text-white transition" aria-label="Menu">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>
    </div>
  </header>

  <!-- Progress Bar -->
  <div id="progressContainer" class="hidden px-4 pt-3 max-w-3xl mx-auto w-full">
    <div class="w-full bg-rose-950/80 rounded-full h-2 border border-rose-900/60 overflow-hidden">
      <div id="progressBar" class="bg-gradient-to-r from-rose-500 to-pink-400 h-full transition-all duration-300" style="width: 0%"></div>
    </div>
    <div class="flex justify-between text-[10px] text-rose-300/80 mt-1">
      <span id="progressText">0%</span>
      <span id="progressCount">0 / 0</span>
    </div>
  </div>

  <!-- Status Banner with Cooldown Timer -->
  <div id="statusPill" class="hidden mx-4 mt-2.5 p-2.5 rounded-xl text-xs romantic-card border border-rose-500/40 text-rose-200 flex items-center justify-between shadow-sm max-w-3xl md:mx-auto">
    <div class="flex items-center gap-2">
      <div id="statusIndicator" class="w-2 h-2 rounded-full bg-rose-400 animate-ping"></div>
      <span id="statusText">Processing...</span>
    </div>
    <button id="btnStop" onclick="stopTranslation()" class="hidden px-2.5 py-1 rounded-lg bg-rose-800 hover:bg-rose-700 text-white text-[10px] font-semibold transition">
      Stop ⏹
    </button>
  </div>

  <!-- Main Container -->
  <main class="flex-1 px-4 py-3 max-w-3xl mx-auto w-full flex flex-col gap-3">
    
    <!-- Upload Section -->
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
            <p class="text-[10px] text-rose-400">Total: <span id="subCount" class="font-bold text-pink-300">0</span> items</p>
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
          <select id="targetLang" class="w-full bg-rose-950/90 border border-rose-800/60 text-xs text-rose-100 rounded-xl p-2 focus:outline-none">
            <option value="Burmese">မြန်မာစာ (Burmese)</option>
            <option value="English">English</option>
            <option value="Thai">ภาษาไทย (Thai)</option>
            <option value="Japanese">日本語 (Japanese)</option>
          </select>
        </div>

        <div>
          <label class="block text-[11px] font-medium text-rose-300 mb-1">စကားပြောပုံစံ / အသုံးအနှုန်းဟန်</label>
          <select id="toneStyle" class="w-full bg-rose-950/90 border border-rose-800/60 text-xs text-rose-100 rounded-xl p-2 focus:outline-none">
            <option value="natural">🗣️ သဘာဝကျ စကားပြောဟန် (ရုပ်ရှင်/ဇာတ်လမ်း)</option>
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
      <span class="text-[11px] text-rose-300 font-medium">Selected Model:</span>
      <span id="footerModelName" class="text-xs text-pink-300 font-semibold font-mono">gemini-3.8-flash</span>
    </div>

    <div class="flex items-center gap-2">
      <button onclick="startTranslation()" id="btnTranslate" class="flex-1 bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white text-xs font-semibold py-3 px-3 rounded-xl flex items-center justify-center gap-1.5 transition romantic-glow active:scale-[0.98]">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
        </svg>
        <span>Translate All</span>
      </button>

      <button onclick="downloadOriginalSRT()" class="bg-rose-950/90 hover:bg-rose-900 text-rose-200 border border-rose-800/60 text-xs font-medium py-3 px-3 rounded-xl transition active:scale-[0.98]">
        <span>Original .SRT</span>
      </button>

      <button onclick="downloadTranslatedSRT()" class="bg-rose-900/80 hover:bg-rose-800 text-white border border-rose-700/60 text-xs font-medium py-3 px-3 rounded-xl transition active:scale-[0.98]">
        <span>Translated .SRT</span>
      </button>
    </div>
  </footer>

  <!-- Drawer Menu -->
  <div id="menuOverlay" onclick="toggleMenu(false)" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 hidden transition-opacity"></div>
  <aside id="menuDrawer" class="fixed top-0 right-0 bottom-0 w-80 max-w-[85vw] bg-rose-950 border-l border-rose-800/60 z-50 transform translate-x-full transition-transform duration-300 flex flex-col p-5 shadow-2xl">
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
            <div class="font-semibold text-white">Settings & Models</div>
            <div class="text-[10px] text-rose-300/70">Official Gemini Models ရွေးချယ်ရန်</div>
          </div>
        </div>
        <span class="text-rose-400 font-bold">➔</span>
      </button>
    </div>
  </aside>

  <!-- Settings Modal (Only 6 Requested Models) -->
  <div id="settingsModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
    <div class="bg-rose-950 border border-rose-800/80 rounded-2xl w-full max-w-sm p-5 shadow-2xl space-y-4">
      <div class="flex items-center justify-between border-b border-rose-900/80 pb-2.5">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <span>⚙</span> Model & API Key
        </h3>
        <button onclick="closeSettingsModal()" class="text-rose-400 hover:text-white text-base">✕</button>
      </div>

      <div class="space-y-3 text-xs">
        <div>
          <label class="block text-[11px] font-medium text-pink-300 mb-1">Official Gemini Model ရွေးပါ</label>
          <select id="modalGeminiSelect" class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2.5 text-rose-100 text-xs focus:outline-none font-mono">
            <option value="gemini-3.8-flash">Gemini 3.8 Flash (gemini-3.8-flash)</option>
            <option value="gemini-3.6-flash">Gemini 3.6 Flash (gemini-3.6-flash)</option>
            <option value="gemini-3.5-flash">Gemini 3.5 Flash (gemini-3.5-flash)</option>
            <option value="gemini-3-flash">Gemini 3 Flash (gemini-3-flash)</option>
            <option value="gemini-3.5-flash-lite">Gemini 3.5 Flash Lite (gemini-3.5-flash-lite)</option>
            <option value="gemini-3.1-flash-lite">Gemini 3.1 Flash Lite (gemini-3.1-flash-lite)</option>
          </select>
        </div>

        <div>
          <label class="block text-[11px] font-medium text-indigo-300 mb-1">Gemini API Key</label>
          <input id="modalGeminiKey" type="password" placeholder="AIzaSy..." class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>

        <div>
          <label class="block text-[11px] font-medium text-rose-400 mb-1">Groq API Key (Audio Transcribe အတွက်သာ)</label>
          <input id="modalGroqKey" type="password" placeholder="gsk_..." class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>
      </div>

      <div class="pt-2">
        <button onclick="saveSettings()" class="w-full bg-gradient-to-r from-rose-600 to-pink-500 text-white font-semibold py-2.5 rounded-xl text-xs shadow transition active:scale-95">
          Save Settings & Close
        </button>
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

    const sleep = (ms) => new Promise(res => setTimeout(res, ms));

    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('modalGroqKey').value = localStorage.getItem(KEY_GROQ) || '';
      document.getElementById('modalGeminiKey').value = localStorage.getItem(KEY_GEMINI) || '';
      
      const savedModel = localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.8-flash';
      const select = document.getElementById('modalGeminiSelect');
      select.value = savedModel;
      document.getElementById('footerModelName').innerText = savedModel;

      document.getElementById('fileInput').addEventListener('change', handleFileSelected);
    });

    function toggleMenu(show) {
      const drawer = document.getElementById('menuDrawer');
      const overlay = document.getElementById('menuOverlay');
      if (show) {
        overlay.classList.remove('hidden');
        drawer.classList.remove('translate-x-full');
      } else {
        drawer.classList.add('translate-x-full');
        overlay.classList.add('hidden');
      }
    }

    function openSettingsModal() { document.getElementById('settingsModal').classList.remove('hidden'); }
    function closeSettingsModal() { document.getElementById('settingsModal').classList.add('hidden'); }

    function saveSettings() {
      const gKey = document.getElementById('modalGroqKey').value.trim();
      const gmKey = document.getElementById('modalGeminiKey').value.trim();
      const selectedModel = document.getElementById('modalGeminiSelect').value;

      localStorage.setItem(KEY_GROQ, gKey);
      localStorage.setItem(KEY_GEMINI, gmKey);
      localStorage.setItem(KEY_GEMINI_MODEL, selectedModel);

      document.getElementById('footerModelName').innerText = selectedModel;
      closeSettingsModal();
      setStatus("Model & Settings အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!", 3000);
    }

    function setStatus(text, duration = 0) {
      const pill = document.getElementById('statusPill');
      document.getElementById('statusText').innerText = text;
      pill.classList.remove('hidden');
      if (duration > 0) setTimeout(() => pill.classList.add('hidden'), duration);
    }

    function updateProgress(done, total) {
      const pContainer = document.getElementById('progressContainer');
      pContainer.classList.remove('hidden');
      const pct = Math.round((done / total) * 100);
      document.getElementById('progressBar').style.width = `${pct}%`;
      document.getElementById('progressText').innerText = `${pct}%`;
      document.getElementById('progressCount').innerText = `${done} / ${total}`;
      if (done >= total) setTimeout(() => pContainer.classList.add('hidden'), 3000);
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
        const groqKey = localStorage.getItem(KEY_GROQ);
        if (!groqKey) {
          alert("Video/Audio မှ Subtitle ထုတ်ယူရန် Groq API Key လိုအပ်ပါသည်။ Settings တွင် ထည့်သွင်းပေးပါခင်ဗျာ။");
          openSettingsModal();
          return;
        }

        setStatus("Whisper AI ဖြင့် အသံဖိုင်မှ Subtitle ထုတ်နေပါသည်...");
        const fd = new FormData();
        fd.append('file', file);
        fd.append('apiKey', groqKey);

        try {
          const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
          const data = await res.json();
          if (data.error) throw new Error(data.error);
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
        box.innerHTML = '<div class="romantic-card rounded-2xl p-10 text-center text-rose-300/80">No Subtitles Loaded</div>';
        return;
      }

      box.innerHTML = subtitles.map((s, idx) => `
        <div class="romantic-card rounded-2xl p-3.5 border border-rose-900/30">
          <div class="flex items-center justify-between text-[10px] font-mono text-rose-300/80 border-b border-rose-900/30 pb-1.5 mb-2">
            <span class="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300 font-semibold">#${s.id}</span>
            <span>${s.startTime.split(',')[0]} ➔ ${s.endTime.split(',')[0]}</span>
          </div>
          <div class="text-xs text-rose-200/90 mb-2 leading-relaxed">${escapeHtml(s.originalText)}</div>
          <textarea rows="2" class="w-full bg-rose-950/80 border border-rose-800/50 rounded-xl p-2 text-xs text-rose-50 resize-none focus:outline-none"
            onchange="subtitles[${idx}].translatedText = this.value">${escapeHtml(s.translatedText)}</textarea>
        </div>
      `).join('');
    }

    function stopTranslation() {
      isTranslating = false;
      document.getElementById('btnStop').classList.add('hidden');
      setStatus("ရပ်တန့်လိုက်ပါပြီ", 3000);
    }

    async function startTranslation() {
      if (isTranslating) return;
      const geminiKey = localStorage.getItem(KEY_GEMINI) || '';
      const modelName = localStorage.getItem(KEY_GEMINI_MODEL) || 'gemini-3.8-flash';

      if (!geminiKey) {
        alert("Gemini API Key ထည့်သွင်းပေးပါ (Settings တွင် ထည့်နိုင်ပါသည်)");
        openSettingsModal();
        return;
      }
      if (!subtitles.length) return alert("Subtitle မရှိသေးပါ");

      const pending = subtitles.filter(s => !s.translatedText);
      if (!pending.length) return alert("စာကြောင်းအားလုံး ဘာသာပြန်ပြီးပါပြီ");

      isTranslating = true;
      document.getElementById('btnStop').classList.remove('hidden');

      // Block 50 Items per request
      const chunkSize = 50; 
      let completedCount = subtitles.length - pending.length;
      updateProgress(completedCount, subtitles.length);

      for (let i = 0; i < pending.length; i += chunkSize) {
        if (!isTranslating) break;
        const chunk = pending.slice(i, i + chunkSize);

        setStatus(`[${modelName}] ဘာသာပြန်နေပါသည်: #${chunk[0].id} မှ #${chunk[chunk.length - 1].id} (စာကြောင်း ${chunk.length} ကြောင်း)...`);

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
            const data = await res.json();
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

            // ၁ မိနစ်လျှင် ၂ ကြိမ်သာ ဖြစ်စေရန် (30 Seconds Cooldown Timer)
            const isLastChunk = (i + chunkSize) >= pending.length;
            if (!isLastChunk && isTranslating) {
              for (let sec = 30; sec > 0; sec--) {
                if (!isTranslating) break;
                setStatus(`နောက်တစ်ကြိမ် မပို့မီ စောင့်ဆိုင်းနေပါသည် (Rate Limit Safe): ${sec} စက္ကန့်...`);
                await sleep(1000);
              }
            }

          } catch(e) {
            retries++;
            const waitSec = retries * 10;
            setStatus(`Error: ${e.message} — ${waitSec}s စောင့်ဆိုင်းနေပါသည် (${retries}/3)...`);
            await sleep(waitSec * 1000);
          }
        }
      }

      isTranslating = false;
      document.getElementById('btnStop').classList.add('hidden');
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

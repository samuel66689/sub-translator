import os
import re
import json
import requests
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

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
        return jsonify({"error": "Video/Audio အတွက် Groq API Key လိုအပ်ပါသည်"}), 400

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
            err_msg = res.json().get('error', {}).get('message', 'Audio Transcription Failed')
            return jsonify({"error": err_msg}), 400

        result = res.json()
        items = []
        for idx, seg in enumerate(result.get('segments', []), 1):
            s = seg['start']
            e = seg['end']
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
    model_name = req.get('modelName', 'gemini-3.6-flash')

    if not api_key:
        return jsonify({"error": "API Key ထည့်သွင်းပေးပါ"}), 400

    system_prompt = (
        f"You are a professional subtitle translator. "
        f"Translate the following subtitle items into {target_lang}. "
        f"Output strictly a JSON array of objects with 'id' and 'translatedText'. "
        f"Example: [{{\"id\": 1, \"translatedText\": \"မင်္ဂလာပါ\"}}]"
    )

    payload_data = [{"id": s["id"], "text": s["originalText"]} for s in subtitles]

    try:
        if provider == 'gemini':
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            body = {
                "contents": [
                    {"parts": [{"text": system_prompt}, {"text": json.dumps(payload_data)}]}
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.2
                }
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
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
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
  <title>Thiri's Koko</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {
      background: radial-gradient(circle at 50% 0%, #2b1120 0%, #14080e 100%);
      -webkit-tap-highlight-color: transparent;
    }
    .romantic-card {
      background: rgba(35, 15, 25, 0.75);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(243, 146, 189, 0.2);
    }
  </style>
</head>
<body class="text-rose-100 min-h-full flex flex-col font-sans pb-28">

  <header class="sticky top-0 z-40 romantic-card border-b border-rose-900/40 px-4 py-3 flex items-center justify-between">
    <div class="flex items-center gap-2">
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-rose-600 to-pink-400 flex items-center justify-center text-white font-bold shadow-md">
        ❤
      </div>
      <div>
        <h1 class="text-sm font-bold text-white tracking-tight">Thiri's Koko</h1>
        <p class="text-[10px] text-rose-300/80">Romantic Subtitle Studio</p>
      </div>
    </div>
    <select id="targetLang" class="bg-rose-950 border border-rose-800 text-xs text-rose-100 rounded-lg px-2 py-1.5 focus:outline-none">
      <option value="Burmese">Burmese (မြန်မာ)</option>
      <option value="English">English</option>
      <option value="Thai">Thai (ไทย)</option>
      <option value="Japanese">Japanese</option>
    </select>
  </header>

  <div class="px-4 pt-3 pb-1 md:hidden">
    <div class="grid grid-cols-2 p-1 rounded-xl bg-rose-950/60 border border-rose-900 text-xs font-medium">
      <button onclick="switchTab('subtitles')" id="tabBtnSub" class="py-2 rounded-lg bg-rose-600 text-white font-semibold">
        Subtitles (<span id="subCount">0</span>)
      </button>
      <button onclick="switchTab('config')" id="tabBtnCfg" class="py-2 rounded-lg text-rose-300">
        Settings & Upload
      </button>
    </div>
  </div>

  <div id="statusPill" class="hidden mx-4 mt-2 p-2 rounded-xl text-xs romantic-card border border-rose-500/40 text-rose-200 text-center animate-pulse">
    <span id="statusText">Processing...</span>
  </div>

  <div class="flex-1 px-4 py-3 max-w-4xl mx-auto w-full grid grid-cols-1 md:grid-cols-12 gap-4">
    <section id="configSection" class="hidden md:flex md:col-span-4 flex-col gap-3 romantic-card rounded-2xl p-4">
      <h2 class="text-xs font-semibold uppercase tracking-wider text-rose-300 border-b border-rose-900 pb-2">Settings</h2>

      <div class="border border-dashed border-rose-500/40 rounded-xl p-4 text-center bg-rose-950/30">
        <input type="file" id="fileInput" accept=".srt,video/*,audio/*" class="hidden"/>
        <label for="fileInput" class="cursor-pointer flex flex-col items-center gap-1">
          <span class="text-xs font-medium text-rose-100">Upload SRT / Video / Audio</span>
          <span class="text-[10px] text-rose-400">Tap here to choose file</span>
        </label>
      </div>

      <div>
        <label class="block text-[11px] text-rose-300 mb-1">AI Engine</label>
        <select id="provider" onchange="toggleProvider()" class="w-full bg-rose-950 border border-rose-800 text-xs text-rose-100 rounded-lg p-2">
          <option value="gemini">Google Gemini</option>
          <option value="groq">Groq (Llama / Whisper)</option>
        </select>
      </div>

      <div>
        <label class="block text-[11px] text-rose-300 mb-1">Model Name</label>
        <input id="modelName" type="text" value="gemini-3.6-flash" class="w-full bg-rose-950 border border-rose-800 text-xs text-rose-100 rounded-lg p-2">
      </div>

      <div>
        <label class="block text-[11px] text-rose-300 mb-1">API Key</label>
        <input id="apiKey" type="password" placeholder="Enter API Key" class="w-full bg-rose-950 border border-rose-800 text-xs text-rose-100 rounded-lg p-2">
      </div>
    </section>

    <section id="subtitlesSection" class="md:col-span-8 flex flex-col gap-2.5">
      <div id="subList" class="space-y-2.5">
        <div class="romantic-card rounded-2xl p-8 text-center text-rose-300/80">
          <p class="text-xs">No Subtitles Loaded. Upload a file to start.</p>
        </div>
      </div>
    </section>
  </div>

  <footer class="fixed bottom-0 left-0 right-0 z-40 romantic-card border-t border-rose-900/40 p-3 flex items-center gap-2 max-w-lg mx-auto md:max-w-none">
    <button onclick="translateAll()" class="flex-1 bg-gradient-to-r from-rose-600 to-pink-500 text-white text-xs font-semibold py-3 rounded-xl shadow-lg active:scale-95 transition">
      Translate All
    </button>
    <button onclick="downloadSRT()" class="bg-rose-950 border border-rose-800 text-rose-200 text-xs font-medium py-3 px-4 rounded-xl active:scale-95 transition">
      Export .SRT
    </button>
  </footer>

  <script>
    let subtitles = [];

    function switchTab(t) {
      const subSec = document.getElementById('subtitlesSection');
      const cfgSec = document.getElementById('configSection');
      const tabSub = document.getElementById('tabBtnSub');
      const tabCfg = document.getElementById('tabBtnCfg');

      if (t === 'subtitles') {
        subSec.classList.remove('hidden');
        cfgSec.classList.add('hidden');
        tabSub.classList.add('bg-rose-600', 'text-white');
        tabCfg.classList.remove('bg-rose-600', 'text-white');
      } else {
        cfgSec.classList.remove('hidden');
        subSec.classList.add('hidden');
        tabCfg.classList.add('bg-rose-600', 'text-white');
        tabSub.classList.remove('bg-rose-600', 'text-white');
      }
    }

    function toggleProvider() {
      const p = document.getElementById('provider').value;
      document.getElementById('modelName').value = (p === 'gemini') ? 'gemini-3.6-flash' : 'llama-3.3-70b-versatile';
    }

    function setStatus(t) {
      const p = document.getElementById('statusPill');
      document.getElementById('statusText').innerText = t;
      p.classList.remove('hidden');
    }

    function clearStatus() {
      document.getElementById('statusPill').classList.add('hidden');
    }

    document.getElementById('fileInput').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const ext = file.name.split('.').pop().toLowerCase();
      if (ext === 'srt') {
        const txt = await file.text();
        parseSRT(txt);
        switchTab('subtitles');
      } else {
        const apiKey = document.getElementById('apiKey').value;
        const prov = document.getElementById('provider').value;
        if (prov !== 'groq' || !apiKey) {
          alert('Video/Audio အတွက် Engine ကို Groq ရွေးပြီး Groq API Key ထည့်ပေးပါ');
          switchTab('config');
          return;
        }

        setStatus("Whisper AI ဖြင့် စာတန်းထိုး ထုတ်ယူနေပါသည်...");
        const fd = new FormData();
        fd.append('file', file);
        fd.append('apiKey', apiKey);

        try {
          const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
          const data = await res.json();
          if (data.error) throw new Error(data.error);
          subtitles = data.subtitles;
          renderList();
          setStatus("Complete!");
          setTimeout(clearStatus, 3000);
          switchTab('subtitles');
        } catch(err) {
          alert(err.message);
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
        box.innerHTML = '<div class="romantic-card rounded-2xl p-8 text-center text-rose-300/80"><p class="text-xs">No Subtitles Loaded.</p></div>';
        return;
      }

      box.innerHTML = subtitles.map((s, idx) => `
        <div class="romantic-card rounded-xl p-3 border border-rose-900/40">
          <div class="flex justify-between text-[10px] text-rose-400 mb-1 font-mono">
            <span>#${s.id}</span>
            <span>${s.startTime.split(',')[0]} ➔ ${s.endTime.split(',')[0]}</span>
          </div>
          <div class="text-xs text-rose-100 mb-2 leading-relaxed">${s.originalText}</div>
          <textarea rows="2" class="w-full bg-rose-950 border border-rose-800 text-xs p-2 rounded-lg text-rose-100 focus:outline-none"
            placeholder="Translation..."
            onchange="subtitles[${idx}].translatedText = this.value">${s.translatedText}</textarea>
        </div>
      `).join('');
    }

    async function translateAll() {
      const apiKey = document.getElementById('apiKey').value;
      if (!apiKey) {
        alert("API Key ထည့်ပေးပါ");
        switchTab('config');
        return;
      }
      if (!subtitles.length) return alert("Subtitle မရှိသေးပါ");

      setStatus("Translating...");
      const chunkSize = 20;

      for (let i = 0; i < subtitles.length; i += chunkSize) {
        const chunk = subtitles.slice(i, i + chunkSize);
        setStatus(`Translating: ${i + 1} to ${Math.min(i + chunkSize, subtitles.length)}...`);

        try {
          const res = await fetch('/api/translate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              subtitles: chunk,
              targetLang: document.getElementById('targetLang').value,
              provider: document.getElementById('provider').value,
              modelName: document.getElementById('modelName').value.trim(),
              apiKey: apiKey.trim()
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
      setStatus("Finished!");
      setTimeout(clearStatus, 3000);
    }

    function downloadSRT() {
      if (!subtitles.length) return;
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


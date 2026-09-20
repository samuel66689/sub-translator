import os
import re
import json
import requests
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# ----------------- SRT Helper Functions -----------------
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

def to_srt_string(items):
    out = []
    for idx, it in enumerate(items, 1):
        text = it.get('translatedText') or it.get('originalText', '')
        out.append(f"{idx}\n{it['startTime']} --> {it['endTime']}\n{text}\n")
    return "\n".join(out)

# ----------------- API Endpoints -----------------
@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    file = request.files.get('file')
    api_key = request.form.get('apiKey')
    if not file or not api_key:
        return jsonify({"error": "File နှင့် Groq API Key လိုအပ်ပါသည်"}), 400

    try:
        # Audio/Video -> Whisper via Groq
        files = {'file': (file.filename, file.read(), file.content_type)}
        data = {'model': 'whisper-large-v3', 'response_format': 'verbose_json'}
        headers = {'Authorization': f'Bearer {api_key}'}
        
        res = requests.post('https://api.groq.com/openai/v1/audio/transcriptions', 
                            headers=headers, files=files, data=data, timeout=120)
        
        if res.status_code != 200:
            return jsonify({"error": res.json().get('error', {}).get('message', 'Transcription error')}), 400

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
    model_name = req.get('modelName', 'gemini-2.5-flash')

    if not api_key:
        return jsonify({"error": "API Key ထည့်သွင်းပေးပါ"}), 400

    system_prompt = f"""You are a professional subtitle translator.
Translate the following subtitle items into {target_lang}.
Output STRICTLY a valid JSON array of objects with keys: "id" (number) and "translatedText" (string).
Do NOT modify IDs. Keep translations concise and natural for subtitles.
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
            res = requests.post(url, headers=headers, json=body, timeout=60)
            if res.status_code != 200:
                return jsonify({"error": res.json().get('error', {}).get('message', 'Gemini API Error')}), 400
            
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
            res = requests.post(url, headers=headers, json=body, timeout=60)
            if res.status_code != 200:
                return jsonify({"error": res.json().get('error', {}).get('message', 'Groq API Error')}), 400
            
            parsed = json.loads(res.json()['choices'][0]['message']['content'])
            translations = parsed if isinstance(parsed, list) else parsed.get('translations', parsed.get('subtitles', []))

        return jsonify({"translations": translations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------- Frontend Interface -----------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Subtitle Translator Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-zinc-950 text-zinc-100 min-h-screen flex flex-col font-sans">

  <!-- Header -->
  <header class="border-b border-zinc-800 bg-zinc-900/80 px-6 py-3 flex items-center justify-between sticky top-0 z-30">
    <div class="flex items-center gap-3">
      <span class="w-8 h-8 rounded bg-indigo-600 flex items-center justify-center font-bold text-white">S</span>
      <h1 class="text-base font-semibold">SubCraft AI Studio</h1>
    </div>
    <div class="flex items-center gap-3">
      <select id="targetLang" class="bg-zinc-800 border border-zinc-700 text-xs px-3 py-2 rounded text-zinc-200">
        <option value="Burmese">Burmese (မြန်မာစာ)</option>
        <option value="English">English</option>
        <option value="Thai">Thai (ไทย)</option>
        <option value="Japanese">Japanese</option>
      </select>
      <button onclick="translateAll()" id="btnTranslate" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium px-4 py-2 rounded transition">
        Translate Subtitles
      </button>
      <button onclick="downloadSRT()" class="bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-200 text-xs px-3 py-2 rounded">
        Export .SRT
      </button>
    </div>
  </header>

  <!-- Main Content -->
  <div class="flex-1 flex flex-col md:flex-row overflow-hidden">
    <!-- Left Panel: Settings & Upload -->
    <div class="w-full md:w-80 border-r border-zinc-800 p-5 flex flex-col gap-4 bg-zinc-900/40">
      <h2 class="text-xs font-semibold text-zinc-400 uppercase tracking-wider">AI Configuration</h2>
      
      <div>
        <label class="block text-xs mb-1 text-zinc-300">Engine</label>
        <select id="provider" onchange="toggleProvider()" class="w-full bg-zinc-950 border border-zinc-800 text-xs p-2 rounded">
          <option value="gemini">Google Gemini</option>
          <option value="groq">Groq (Llama / Whisper)</option>
        </select>
      </div>

      <div>
        <label class="block text-xs mb-1 text-zinc-300">Model Name</label>
        <input id="modelName" type="text" value="gemini-2.5-flash" class="w-full bg-zinc-950 border border-zinc-800 text-xs p-2 rounded text-zinc-200" placeholder="e.g. gemini-2.5-flash / gemini-3.5-flash">
      </div>

      <div>
        <label class="block text-xs mb-1 text-zinc-300">API Key</label>
        <input id="apiKey" type="password" placeholder="Enter API Key" class="w-full bg-zinc-950 border border-zinc-800 text-xs p-2 rounded text-zinc-200">
      </div>

      <div class="mt-3">
        <label class="block text-xs mb-2 text-zinc-300">Upload File (.srt, .mp4, .mp3, .wav)</label>
        <input type="file" id="fileInput" accept=".srt,video/*,audio/*" class="w-full text-xs text-zinc-400 file:mr-2 file:py-2 file:px-3 file:rounded file:border-0 file:text-xs file:bg-indigo-600 file:text-white cursor-pointer"/>
      </div>

      <div id="status" class="text-xs text-indigo-400 mt-2 hidden">Processing...</div>
    </div>

    <!-- Right Panel: Side-by-Side Editor -->
    <div class="flex-1 flex flex-col bg-zinc-950 overflow-hidden">
      <div class="grid grid-cols-12 px-4 py-2 border-b border-zinc-800 text-xs font-semibold text-zinc-400 bg-zinc-900/60">
        <div class="col-span-2">Time</div>
        <div class="col-span-5">Original</div>
        <div class="col-span-5">Translated</div>
      </div>
      <div id="subList" class="flex-1 overflow-y-auto divide-y divide-zinc-800/60 p-2 space-y-1">
        <div class="text-center text-zinc-600 text-xs py-20">No subtitles loaded. Upload a video, audio, or .srt file to start.</div>
      </div>
    </div>
  </div>

  <script>
    let subtitles = [];

    function toggleProvider() {
      const p = document.getElementById('provider').value;
      const modelInput = document.getElementById('modelName');
      modelInput.value = (p === 'gemini') ? 'gemini-2.5-flash' : 'llama-3.3-70b-versatile';
    }

    // Load file
    document.getElementById('fileInput').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const ext = file.name.split('.').pop().toLowerCase();
      const status = document.getElementById('status');

      if (ext === 'srt') {
        const text = await file.text();
        parseSRTClient(text);
      } else {
        const apiKey = document.getElementById('apiKey').value;
        if (!apiKey) {
          alert('Audio/Video transcribe ရန် Groq API key အရင်ထည့်ပေးပါ');
          return;
        }
        status.innerText = "Transcribing with Whisper AI...";
        status.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);
        formData.append('apiKey', apiKey);

        try {
          const res = await fetch('/api/transcribe', { method: 'POST', body: formData });
          const data = await res.json();
          if (data.error) throw new Error(data.error);
          subtitles = data.subtitles;
          renderList();
          status.innerText = "Transcription Complete!";
        } catch(err) {
          alert(err.message);
          status.classList.add('hidden');
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
      if (!subtitles.length) {
        box.innerHTML = '<div class="text-center text-zinc-600 text-xs py-20">No subtitles loaded.</div>';
        return;
      }
      box.innerHTML = subtitles.map((s, idx) => `
        <div class="grid grid-cols-12 px-4 py-2 gap-3 text-xs items-start">
          <div class="col-span-2 font-mono text-zinc-500">${s.startTime.split(',')[0]}<br>${s.endTime.split(',')[0]}</div>
          <div class="col-span-5 text-zinc-300">${s.originalText}</div>
          <div class="col-span-5">
            <textarea onchange="subtitles[${idx}].translatedText = this.value" 
              class="w-full bg-zinc-900 border border-zinc-800 rounded p-1.5 text-zinc-100 text-xs resize-none focus:outline-none focus:border-indigo-500" 
              rows="2">${s.translatedText}</textarea>
          </div>
        </div>
      `).join('');
    }

    async function translateAll() {
      const apiKey = document.getElementById('apiKey').value;
      if (!apiKey) return alert("API Key ထည့်သွင်းပေးပါ");
      if (!subtitles.length) return alert("Translate လုပ်ရန် Subtitle မရှိသေးပါ");

      const status = document.getElementById('status');
      status.innerText = "Translating...";
      status.classList.remove('hidden');

      const chunkSize = 25;
      for (let i = 0; i < subtitles.length; i += chunkSize) {
        const chunk = subtitles.slice(i, i + chunkSize);
        try {
          const res = await fetch('/api/translate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              subtitles: chunk,
              targetLang: document.getElementById('targetLang').value,
              provider: document.getElementById('provider').value,
              modelName: document.getElementById('modelName').value,
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
        } catch(err) {
          alert("Error: " + err.message);
          break;
        }
      }
      status.innerText = "Finished!";
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
      a.download = "translated_subtitles.srt";
      a.click();
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("Server running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
  

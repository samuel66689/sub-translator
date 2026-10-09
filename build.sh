#!/bin/bash
# ffmpeg extracts audio from big video files for transcription
# (Groq only accepts <= 25MB). Download a static build if not present.

set -e

if [ "$(id -u)" = "0" ]; then
    apt-get update && apt-get install -y ffmpeg unzip
elif [ -x "bin/ffmpeg" ] && [ -x "bin/ffprobe" ]; then
    echo "ffmpeg already present in ./bin, skipping download"
else
    # Download static ffmpeg build for Linux amd64
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar xf ffmpeg-release-amd64-static.tar.xz
    mkdir -p bin
    cp ffmpeg-*-amd64-static/ffmpeg bin/
    cp ffmpeg-*-amd64-static/ffprobe bin/
    chmod +x bin/ffmpeg bin/ffprobe
    # Clean up the large extracted dir + tarball to save disk
    rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-amd64-static
fi

# deno: the ONLY JS runtime that solves YouTube's PO-token challenge for the
# android player client (node does NOT work — verified 2026-10-09). Best-effort
# install here; the app also self-heals a missing deno at runtime via
# _ensure_deno() (the Render runtime network can download it), so a failed
# build-time install must NOT fail the deploy.
if [ ! -x "bin/deno" ]; then
    python3 -c "
import urllib.request, zipfile, os
url = 'https://github.com/denoland/deno/releases/download/v2.9.7/deno-x86_64-unknown-linux-gnu.zip'
os.makedirs('bin', exist_ok=True)
print('downloading deno...', flush=True)
urllib.request.urlretrieve(url, '/tmp/deno.zip')
print('extracting deno...', flush=True)
with zipfile.ZipFile('/tmp/deno.zip') as z:
    z.extractall('bin')
os.chmod('bin/deno', 0o755)
os.remove('/tmp/deno.zip')
print('deno bytes:', os.path.getsize('bin/deno'), flush=True)
" || echo "WARNING: deno build-time install failed — app will self-heal at runtime"
fi

pip install -r requirements.txt

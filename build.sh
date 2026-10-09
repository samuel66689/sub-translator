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

# deno: JS runtime that yt-dlp needs to solve YouTube's signature challenge
# when using the android player client (bypasses datacenter bot-checks).
if [ ! -x "bin/deno" ]; then
    DENO_VERSION="v2.9.7"
    wget -q "https://dl.deno.land/release/${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip" -O /tmp/deno.zip
    unzip -o -q /tmp/deno.zip -d bin/
    chmod +x bin/deno
    rm -f /tmp/deno.zip
else
    echo "deno already present in ./bin, skipping download"
fi

pip install -r requirements.txt

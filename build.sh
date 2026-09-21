#!/bin/bash
# Install ffmpeg which is required for pydub to process audio
# Use apt-get if running as root (e.g., in Docker), otherwise we may need an alternative.
# Render's native python environment runs as an unprivileged user and doesn't support apt-get directly.
# Let's ensure this script downloads static ffmpeg if apt fails.

if [ "$(id -u)" = "0" ]; then
    apt-get update && apt-get install -y ffmpeg
else
    # Download static ffmpeg build for Linux amd64
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar xvf ffmpeg-release-amd64-static.tar.xz
    mkdir -p bin
    cp ffmpeg-*-amd64-static/ffmpeg bin/
    cp ffmpeg-*-amd64-static/ffprobe bin/
    export PATH=$PATH:$(pwd)/bin
fi

pip install -r requirements.txt

#!/bin/bash
# Install ffmpeg which is required for pydub to process audio
# Use apt-get if running as root (e.g., in Docker), otherwise we may need an alternative.
# Render's native python environment runs as an unprivileged user and doesn't support apt-get directly.
# Let's ensure this script downloads static ffmpeg if apt fails.

set -e

if [ "$(id -u)" = "0" ]; then
    apt-get update && apt-get install -y ffmpeg
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
    export PATH=$PATH:$(pwd)/bin
fi

pip install -r requirements.txt

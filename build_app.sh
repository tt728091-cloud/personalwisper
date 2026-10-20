#!/bin/zsh
# Build PersonalWisper.app in this folder. Re-run after moving the project folder.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT_DIR="$PWD"
APP="$PROJECT_DIR/PersonalWisper.app"
BUILD="$PROJECT_DIR/app/build"

rm -rf "$APP" "$BUILD"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$BUILD" logs

clang -O2 -Wall -DPROJECT_DIR="\"$PROJECT_DIR\"" -o "$APP/Contents/MacOS/PersonalWisper" app/launcher.c
cp app/Info.plist "$APP/Contents/Info.plist"
# Records other apps' sound for meeting transcripts (macOS 14.2+).
clang -O2 -Wall -fobjc-arc -mmacosx-version-min=14.2 -framework Foundation -framework CoreAudio -framework AVFoundation \
  -o "$APP/Contents/MacOS/systemaudio" app/systemaudio.m

.venv/bin/python app/make_icon.py "$BUILD/AppIcon.iconset"
iconutil -c icns "$BUILD/AppIcon.iconset" -o "$APP/Contents/Resources/AppIcon.icns"

# Ad-hoc signature gives macOS a stable identity to attach permissions to.
codesign --force --sign - --identifier local.personalwisper.systemaudio "$APP/Contents/MacOS/systemaudio"
codesign --force --sign - --identifier local.personalwisper "$APP"
echo "Built $APP"

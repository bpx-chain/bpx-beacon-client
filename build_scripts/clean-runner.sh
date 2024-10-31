#!/bin/bash
# Cleans up files/directories that may be left over from previous runs for a clean slate before starting a new build

set -o errexit

PWD=$(pwd)

rm -rf ../venv || true
rm -rf venv || true
rm -rf bpx_blockchain.egg-info || true
rm -rf build_scripts/final_installer || true
rm -rf build_scripts/dist || true
rm -rf build_scripts/pyinstaller || true
rm -rf bpx-gui/build || true
rm -rf bpx-gui/daemon || true
rm -rf bpx-gui/node_modules || true
rm bpx-gui/temp.json || true
( cd "$PWD/bpx-gui" && git checkout HEAD -- package-lock.json ) || true
cd "$PWD" || true

# Clean up old globally installed node_modules that might conflict with the current build
rm -rf /opt/homebrew/lib/node_modules || true

# Clean up any installed versions of node so we can start fresh
brew list | grep "^node\@\|^node$" | xargs -L1 brew uninstall || true

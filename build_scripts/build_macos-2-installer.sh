#!/bin/bash

set -o errexit -o nounset

git status
git submodule

if [ ! "$BPX_INSTALLER_VERSION" ]; then
	echo "WARNING: No environment variable BPX_INSTALLER_VERSION set. Using 0.0.0."
	BPX_INSTALLER_VERSION="0.0.0"
fi
echo "BPX Installer Version is: $BPX_INSTALLER_VERSION"

echo "Installing npm utilities"
cd npm_macos || exit 1
npm ci
PATH=$(npm bin):$PATH
cd .. || exit 1

echo "Create dist/"
sudo rm -rf dist
mkdir dist

echo "Create executables with pyinstaller"
SPEC_FILE=$(python -c 'import bpx; print(bpx.PYINSTALLER_SPEC_PATH)')
pyinstaller --log-level=INFO "$SPEC_FILE"
LAST_EXIT_CODE=$?
if [ "$LAST_EXIT_CODE" -ne 0 ]; then
	echo >&2 "pyinstaller failed!"
	exit $LAST_EXIT_CODE
fi
cp -r dist/daemon ../bpx-gui/packages/gui

# Change to the gui package
cd ../bpx-gui/packages/gui || exit 1

# sets the version for bpx-beacon-client in package.json
brew install jq
cp package.json package.json.orig
jq --arg VER "$BPX_INSTALLER_VERSION" '.version=$VER' package.json > temp.json && mv temp.json package.json

echo "Building macOS Electron app"
OPT_ARCH="--x64"
if [ "$(arch)" = "arm64" ]; then
  OPT_ARCH="--arm64"
fi
PRODUCT_NAME="BPX Beacon Client"
echo electron-builder build --mac "${OPT_ARCH}" --config.productName="$PRODUCT_NAME"
electron-builder build --mac "${OPT_ARCH}" --config.productName="$PRODUCT_NAME"
LAST_EXIT_CODE=$?
ls -l dist/mac*/bpx.app/Contents/Resources/app.asar

# reset the package.json to the original
mv package.json.orig package.json

if [ "$LAST_EXIT_CODE" -ne 0 ]; then
	echo >&2 "electron-builder failed!"
	exit $LAST_EXIT_CODE
fi

mv dist/* ../../../build_scripts/dist/
cd ../../../build_scripts || exit 1

mkdir final_installer
DMG_NAME="bpx-gui-${BPX_INSTALLER_VERSION}.dmg"
if [ "$(arch)" = "arm64" ]; then
  mv dist/"${DMG_NAME}" dist/bpx-beacon-client-"${BPX_INSTALLER_VERSION}"-arm64.dmg
  DMG_NAME=bpx-${BPX_INSTALLER_VERSION}-arm64.dmg
fi
mv dist/"$DMG_NAME" final_installer/

ls -lh final_installer
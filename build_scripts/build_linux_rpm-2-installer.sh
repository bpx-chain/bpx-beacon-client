#!/bin/bash

set -o errexit

git status
git submodule

if [ ! "$1" ]; then
  echo "This script requires either amd64 of arm64 as an argument"
	exit 1
elif [ "$1" = "amd64" ]; then
	export REDHAT_PLATFORM="x86_64"
else
	export REDHAT_PLATFORM="arm64"
fi

if [ ! "$BPX_INSTALLER_VERSION" ]; then
	echo "WARNING: No environment variable BPX_INSTALLER_VERSION set. Using 0.0.0."
	BPX_INSTALLER_VERSION="0.0.0"
fi
echo "BPX Installer Version is: $BPX_INSTALLER_VERSION"

echo "Installing npm and electron packagers"
cd npm_linux || exit 1
npm ci
PATH=$(npm bin):$PATH
cd .. || exit 1

echo "Create dist/"
rm -rf dist
mkdir dist

echo "Create executables with pyinstaller"
SPEC_FILE=$(python -c 'import bpx; print(bpx.PYINSTALLER_SPEC_PATH)')
pyinstaller --log-level=INFO "$SPEC_FILE"
LAST_EXIT_CODE=$?
if [ "$LAST_EXIT_CODE" -ne 0 ]; then
	echo >&2 "pyinstaller failed!"
	exit $LAST_EXIT_CODE
fi

# Builds CLI only rpm
CLI_RPM_BASE="bpx-beacon-client-cli-$BPX_INSTALLER_VERSION-1.$REDHAT_PLATFORM"
mkdir -p "dist/$CLI_RPM_BASE/opt/bpx-beacon-client"
mkdir -p "dist/$CLI_RPM_BASE/usr/bin"
cp -r dist/daemon/* "dist/$CLI_RPM_BASE/opt/bpx-beacon-client/"
ln -s ../../opt/bpx-beacon-client/bpx "dist/$CLI_RPM_BASE/usr/bin/bpx"
# This is built into the base build image
# shellcheck disable=SC1091
. /etc/profile.d/rvm.sh
rvm use ruby-3
# /usr/lib64/libcrypt.so.1 is marked as a dependency specifically because newer versions of fedora bundle
# libcrypt.so.2 by default, and the libxcrypt-compat package needs to be installed for the other version
# Marking as a dependency allows yum/dnf to automatically install the libxcrypt-compat package as well
fpm -s dir -t rpm \
  -C "dist/$CLI_RPM_BASE" \
  --directories "/opt/bpx-beacon-client" \
  -p "dist/$CLI_RPM_BASE.rpm" \
  --name bpx-beacon-client-cli \
  --license Apache-2.0 \
  --version "$BPX_INSTALLER_VERSION" \
  --architecture "$REDHAT_PLATFORM" \
  --description "BPX is an EVM-compatible blockchain based on Proof of Space and Time consensus." \
  --depends /usr/lib64/libcrypt.so.1 \
  --before-install=assets/rpm/before-install.sh \
  --rpm-tag 'Requires(pre): findutils' \
  .
# CLI only rpm done

cp -r dist/daemon ../bpx-gui/packages/gui

# Change to the gui package
cd ../bpx-gui/packages/gui || exit 1

# sets the version for bpx-beacon-client in package.json
cp package.json package.json.orig
jq --arg VER "$BPX_INSTALLER_VERSION" '.version=$VER' package.json > temp.json && mv temp.json package.json

echo "Building Linux(rpm) Electron app"
OPT_ARCH="--x64"
if [ "$REDHAT_PLATFORM" = "arm64" ]; then
  OPT_ARCH="--arm64"
fi
PRODUCT_NAME="bpx-beacon-client"
echo electron-builder build --linux rpm "${OPT_ARCH}" \
  --config.extraMetadata.name=bpx-beacon-client \
  --config.productName="${PRODUCT_NAME}" --config.linux.desktop.Name="BPX Beacon Client" \
  --config.rpm.packageName="bpx-beacon-client"
electron-builder build --linux rpm "${OPT_ARCH}" \
  --config.extraMetadata.name=bpx-beacon-client \
  --config.productName="${PRODUCT_NAME}" --config.linux.desktop.Name="BPX Beacon Client" \
  --config.rpm.packageName="bpx-beacon-client"
LAST_EXIT_CODE=$?
ls -l dist/linux*-unpacked/resources

# reset the package.json to the original
mv package.json.orig package.json

if [ "$LAST_EXIT_CODE" -ne 0 ]; then
	echo >&2 "electron-builder failed!"
	exit $LAST_EXIT_CODE
fi

GUI_RPM_NAME="bpx-beacon-client-${BPX_INSTALLER_VERSION}-1.${REDHAT_PLATFORM}.rpm"
mv "dist/${PRODUCT_NAME}-${BPX_INSTALLER_VERSION}.rpm" "../../../build_scripts/dist/${GUI_RPM_NAME}"
cd ../../../build_scripts || exit 1

echo "Create final installer"
rm -rf final_installer
mkdir final_installer

mv "dist/${GUI_RPM_NAME}" final_installer/
# Move the cli only rpm into final installers as well, so it gets uploaded as an artifact
mv "dist/$CLI_RPM_BASE.rpm" final_installer/

ls -l final_installer/

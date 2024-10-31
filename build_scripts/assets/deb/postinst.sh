#!/usr/bin/env bash
# Post install script for the UI .deb to place symlinks in places to allow the CLI to work similarly in both versions

set -e

ln -s /opt/bpx-beacon-client/resources/app.asar.unpacked/daemon/bpx /usr/bin/bpx || true
ln -s /opt/bpx-beacon-client/bpx-gui /usr/bin/bpx-gui || true

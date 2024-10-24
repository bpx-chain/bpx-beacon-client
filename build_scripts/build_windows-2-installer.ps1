# $env:path should contain a path to editbin.exe

$ErrorActionPreference = "Stop"

mkdir build_scripts\win_build

git status
git submodule

if (-not (Test-Path env:BPX_INSTALLER_VERSION)) {
  $env:BPX_INSTALLER_VERSION = '0.0.0'
  Write-Output "WARNING: No environment variable BPX_INSTALLER_VERSION set. Using 0.0.0"
}
Write-Output "BPX Version is: $env:BPX_INSTALLER_VERSION"
Write-Output "   ---"

Write-Output "   ---"
Write-Output "Use pyinstaller to create bpx .exe's"
Write-Output "   ---"
$SPEC_FILE = (python -c 'import bpx; print(bpx.PYINSTALLER_SPEC_PATH)') -join "`n"
pyinstaller --log-level INFO $SPEC_FILE

Write-Output "   ---"
Write-Output "Copy bpx executables to bpx-gui\"
Write-Output "   ---"
Copy-Item "dist\daemon" -Destination "..\bpx-gui\packages\gui\" -Recurse

Write-Output "   ---"
Write-Output "Setup npm packager"
Write-Output "   ---"
Set-Location -Path ".\npm_windows" -PassThru
npm ci
$Env:Path = $(npm bin) + ";" + $Env:Path

Set-Location -Path "..\..\" -PassThru

Write-Output "   ---"
Write-Output "Prepare Electron packager"
Write-Output "   ---"
$Env:NODE_OPTIONS = "--max-old-space-size=3000"

# Change to the GUI directory
Set-Location -Path "bpx-gui\packages\gui" -PassThru

Write-Output "   ---"
Write-Output "Increase the stack for bpx command for (bpx plots create) chiapos limitations"
# editbin.exe needs to be in the path
editbin.exe /STACK:8000000 daemon\bpx.exe
Write-Output "   ---"

$packageVersion = "$env:BPX_INSTALLER_VERSION"
$packageName = "bpx-gui-$packageVersion"

Write-Output "packageName is $packageName"

Write-Output "   ---"
Write-Output "fix version in package.json"
choco install jq
cp package.json package.json.orig
jq --arg VER "$env:BPX_INSTALLER_VERSION" '.version=$VER' package.json > temp.json
rm package.json
mv temp.json package.json
Write-Output "   ---"

Write-Output "   ---"
Write-Output "electron-builder"
electron-builder build --win --x64 --config.productName="BPX Beacon Client"
Get-ChildItem dist\win-unpacked\resources
Write-Output "   ---"

Write-Output "   ---"
Write-Output "Moving final installers to expected location"
Write-Output "   ---"
Copy-Item ".\dist\win-unpacked" -Destination "$env:GITHUB_WORKSPACE\bpx-gui\bpx-gui-win32-x64" -Recurse
mkdir "$env:GITHUB_WORKSPACE\bpx-gui\release-builds\windows-installer" -ea 0
Copy-Item ".\dist\bpx-beacon-client-$packageVersion.exe" -Destination "$env:GITHUB_WORKSPACE\bpx-gui\release-builds\windows-installer\bpx-beacon-client_${packageVersion}_amd64.exe"

Write-Output "   ---"
Write-Output "Windows Installer complete"
Write-Output "   ---"

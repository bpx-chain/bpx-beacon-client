# -*- mode: python ; coding: utf-8 -*-
import importlib
import os
import pathlib
import platform
import sysconfig

from pkg_resources import get_distribution

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

THIS_IS_WINDOWS = platform.system().lower().startswith("win")
THIS_IS_MAC = platform.system().lower().startswith("darwin")

ROOT = pathlib.Path(importlib.import_module("bpx").__file__).absolute().parent.parent


def solve_name_collision_problem(analysis):
    """
    There is a collision between the `bpx` file name (which is the executable)
    and the `bpx` directory, which contains non-code resources like `english.txt`.
    We move all the resources in the zipped area so there is no
    need to create the `bpx` directory, since the names collide.

    Fetching data now requires going into a zip file, so it will be slower.
    It's best if files that are used frequently are cached.

    A sample large compressible file (1 MB of `/dev/zero`), seems to be
    about eight times slower.

    Note that this hack isn't documented, but seems to work.
    """

    zipped = []
    datas = []
    for data in analysis.datas:
        if str(data[0]).startswith("bpx/"):
            zipped.append(data)
        else:
            datas.append(data)

    # items in this field are included in the binary
    analysis.zipped_data = zipped

    # these items will be dropped in the root folder uncompressed
    analysis.datas = datas


keyring_imports = collect_submodules("keyring.backends")

# keyring uses entrypoints to read keyring.backends from metadata file entry_points.txt.
keyring_datas = copy_metadata("keyring")[0]

version_data = copy_metadata(get_distribution("bpx-beacon-client"))[0]

block_cipher = None

SERVERS = [
    "beacon",
    "harvester",
    "farmer",
    "introducer",
    "timelord",
]

# TODO: collapse all these entry points into one `chia_exec` entrypoint that accepts the server as a parameter

entry_points = ["bpx.cmds.bpx"] + [f"bpx.server.start_{s}" for s in SERVERS]

hiddenimports = []
hiddenimports.extend(entry_points)
hiddenimports.extend(keyring_imports)

binaries = []

if os.path.exists(f"{ROOT}/madmax/chia_plot"):
    binaries.extend([
        (
            f"{ROOT}/madmax/chia_plot",
            "madmax"
        )
    ])

if os.path.exists(f"{ROOT}/madmax/chia_plot_k34",):
    binaries.extend([
        (
            f"{ROOT}/madmax/chia_plot_k34",
            "madmax"
        )
    ])

if os.path.exists(f"{ROOT}/bladebit/bladebit"):
    binaries.extend([
        (
            f"{ROOT}/bladebit/bladebit",
            "bladebit"
        )
    ])

if os.path.exists(f"{ROOT}/bladebit/bladebit_cuda"):
    binaries.extend([
        (
            f"{ROOT}/bladebit/bladebit_cuda",
            "bladebit"
        )
    ])

if THIS_IS_WINDOWS:
    hiddenimports.extend(["win32timezone", "win32cred", "pywintypes", "win32ctypes.pywin32"])

# this probably isn't necessary
if THIS_IS_WINDOWS:
    entry_points.extend(["aiohttp", "bpx.util.bip39"])

if THIS_IS_WINDOWS:
    bpx_mod = importlib.import_module("bpx")
    dll_paths = pathlib.Path(sysconfig.get_path("platlib")) / "*.dll"

    binaries = [
        (
            dll_paths,
            ".",
        ),
        (
            "C:\\Windows\\System32\\msvcp140.dll",
            ".",
        ),
        (
            "C:\\Windows\\System32\\vcruntime140_1.dll",
            ".",
        ),
        (
            f"{ROOT}\\madmax\\chia_plot.exe",
            "madmax"
        ),
        (
            f"{ROOT}\\madmax\\chia_plot_k34.exe",
            "madmax"
        ),
        (
            f"{ROOT}\\bladebit\\bladebit.exe",
            "bladebit"
        ),
    ]
    
    if os.path.exists(f"{ROOT}\\bladebit\\bladebit_cuda.exe"):
        binaries.extend([
            (
                f"{ROOT}\\bladebit\\bladebit_cuda.exe",
                "bladebit"
            )
        ])

datas = []

datas.append((f"{ROOT}/bpx/util/english.txt", "bpx/util"))
datas.append((f"{ROOT}/bpx/util/initial-config.yaml", "bpx/util"))
datas.append((f"{ROOT}/bpx/ssl/*", "bpx/ssl"))
datas.append((f"{ROOT}/mozilla-ca/*", "mozilla-ca"))
datas.append(version_data)

pathex = []


def add_binary(name, path_to_script, collect_args):
    analysis = Analysis(
        [path_to_script],
        pathex=pathex,
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        runtime_hooks=[],
        excludes=[],
        win_no_prefer_redirects=False,
        win_private_assemblies=False,
        cipher=block_cipher,
        noarchive=False,
    )

    solve_name_collision_problem(analysis)

    binary_pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

    binary_exe = EXE(
        binary_pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name=name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
    )

    collect_args.extend(
        [
            binary_exe,
            analysis.binaries,
            analysis.zipfiles,
            analysis.datas,
        ]
    )


COLLECT_ARGS = []

add_binary("bpx", f"{ROOT}/bpx/cmds/bpx.py", COLLECT_ARGS)
add_binary("daemon", f"{ROOT}/bpx/daemon/server.py", COLLECT_ARGS)

for server in SERVERS:
    add_binary(f"start_{server}", f"{ROOT}/bpx/server/start_{server}.py", COLLECT_ARGS)

add_binary("start_crawler", f"{ROOT}/bpx/seeder/start_crawler.py", COLLECT_ARGS)
add_binary("start_seeder", f"{ROOT}/bpx/seeder/dns_server.py", COLLECT_ARGS)

COLLECT_KWARGS = dict(
    strip=False,
    upx_exclude=[],
    name="daemon",
)

coll = COLLECT(*COLLECT_ARGS, **COLLECT_KWARGS)

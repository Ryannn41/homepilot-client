# PyInstaller spec：在 Windows 上于仓库根目录执行
#   pyinstaller packaging/homepilot-desktop.spec

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent.parent

a = Analysis(
    [str(ROOT / "desktop" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "desktop" / "ui"), "desktop/ui"),
        (str(ROOT / "app" / "assets"), "app/assets"),
    ],
    hiddenimports=["webview"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HomePilot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="HomePilot")
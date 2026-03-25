# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for NanoGS GUI v1.0
# Build with:  .venv\Scripts\pyinstaller.exe nanogs_gui.spec

block_cipher = None

a = Analysis(
    ['nanogs_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # scipy: the cKDTree import is inside a try block, so PyInstaller
        # may not detect it during static analysis.
        'scipy.spatial',
        'scipy.spatial.ckdtree',
        'scipy._lib.messagestream',
        'scipy._lib._util',
        # PIL sub-modules loaded lazily
        'PIL._tkinter_finder',
        'PIL.Image',
        'PIL.ImageFile',
        'PIL.WebPImagePlugin',
        'PIL.PngImagePlugin',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude optional heavy packages that are not needed for CPU-only use.
    # Users who want GPU support must run from source with CuPy installed.
    excludes=['cupy', 'cupy_cuda11x', 'cupy_cuda12x', 'pyOpenSSL'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NanoGS_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # windowed = no console window; the GUI's log panel captures all output.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='NanoGS_GUI',
)

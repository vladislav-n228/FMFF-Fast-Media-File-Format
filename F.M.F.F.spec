# -*- mode: python ; coding: utf-8 -*-
import glob
import os
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('tkinterdnd2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pymupdf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# jpeglib's cjpeglib_*.pyd files are picked at import time by listing its
# own package directory on disk (jpeglib._bind.Cjpeglib._versions ->
# os.listdir(cjpeglib.__path__[0])), not by a normal `import` PyInstaller's
# static analysis can see -- collect_all/collect_dynamic_libs both miss
# them entirely (confirmed: 0 binaries found either way), which used to
# make jpeglib crash at startup in the packaged .exe with "the system
# cannot find the path specified" for the (never-bundled) cjpeglib
# directory. Globbing and adding them as binaries by hand, into that same
# jpeglib/cjpeglib destination, is what actually gets them extracted to
# _MEIPASS in a directory _versions() can list.
import jpeglib
_cjpeglib_dir = os.path.join(os.path.dirname(jpeglib.__file__), 'cjpeglib')
for _f in glob.glob(os.path.join(_cjpeglib_dir, 'cjpeglib_*.pyd')):
    binaries.append((_f, 'jpeglib/cjpeglib'))


a = Analysis(
    ['F.M.F.F.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='F.M.F.F',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.ico'],
)

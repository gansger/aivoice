# PyInstaller build description. Run through build.bat.
# A folder build, not one file: it starts faster and trips antivirus heuristics less often.

a = Analysis(
    ["aivoice_app.py"],
    datas=[("aivoice/assets", "aivoice/assets")],
    excludes=["tkinter", "unittest"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="AIVoice",
    icon="aivoice/assets/icon.ico",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="AIVoice", upx=False)

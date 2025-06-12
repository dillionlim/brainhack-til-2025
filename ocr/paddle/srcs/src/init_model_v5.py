#!/usr/bin/env python3
import os
import pathlib
import sys

#
# 1) Determine where paddlex will look for fonts.
#    If PADDLE_PDX_CACHE_HOME is set, use that; otherwise default to ~/.paddlex
#
env_cache = os.environ.get("PADDLE_PDX_CACHE_HOME", None)
if env_cache:
    cache_dir = pathlib.Path(env_cache).expanduser()
else:
    cache_dir = pathlib.Path.home() / ".paddlex"

fonts_dir = cache_dir / "fonts"

#
# 2) List of font filenames that paddlex will attempt to load at import-time.
#
expected_fonts = [
    "PingFang-SC-Regular.ttf",
    "simfang.ttf",
]

#
# 3) Check for each file under fonts_dir. If any are missing, exit(1).
#
missing = []
for fname in expected_fonts:
    fpath = fonts_dir / fname
    if not fpath.is_file():
        missing.append(str(fpath))

if missing:
    print(f"ERROR: missing font files under: {fonts_dir}", file=sys.stderr)
    for p in missing:
        print(f"  • {p}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"✔ All required font files found in: {fonts_dir}")
    sys.exit(0)

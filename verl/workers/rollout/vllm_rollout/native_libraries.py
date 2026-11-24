"""Load the experiment's NIXL dependencies without exporting OpenSSL globally.

Putting these libraries on LD_LIBRARY_PATH crashes the installed PyArrow at
import. RTLD_LOCAL keeps their symbols within the NIXL dependency group.
"""

import ctypes
import hashlib
import os
from pathlib import Path

_handles = []
_HASHES = {
    "libcrypto.so.3": "57eaf849df1f25e3612efae40c81c396ae813eb921c64745cb0b8f2da6076a06",
    "libssl.so.3": "342bd15411863e62b8fd7f0c98a4398daef71fb8c4a3fe253623fc02df425580",
}


def load_nixl_dependencies():
    directory = os.environ.get("VERL_V17_NATIVE_LIBRARY_DIR")
    if not directory or _handles:
        return
    root = Path(directory)
    paths = [root / name for name in _HASHES]
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != _HASHES[path.name]:
            raise RuntimeError(f"NIXL dependency content differs: {path.name}")
    for path in paths:
        _handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL))

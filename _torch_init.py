"""
_torch_init.py
==============
Windows DLL path bootstrap for PyTorch on Python 3.12+.

PyTorch on Windows requires its internal DLLs to be discoverable before
`import torch` is called. On Python 3.8+ Windows, `os.add_dll_directory`
is the correct mechanism.

Import this module BEFORE importing torch, or place it in sitecustomize.py.

Usage (automatic via conftest import pattern):
    import _torch_init  # sets DLL paths
    import torch         # now works
"""

import os
import sys
from pathlib import Path


def _add_torch_dll_directories() -> None:
    """
    Add the torch/lib directory to the Windows DLL search path.
    This fixes 'DLL load failed while importing _C' on Python 3.12+.
    """
    if sys.platform != "win32":
        return  # Only needed on Windows

    # Find the torch package location
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            return

        torch_root = Path(spec.origin).parent
        lib_dir    = torch_root / "lib"

        if lib_dir.is_dir():
            os.add_dll_directory(str(lib_dir))

            # Also add the parent Scripts directory (for CUDA DLLs if present)
            scripts_dir = Path(sys.prefix) / "Scripts"
            if scripts_dir.is_dir():
                os.add_dll_directory(str(scripts_dir))

    except Exception:
        # Silently skip — worst case torch raises its own helpful error
        pass


_add_torch_dll_directories()

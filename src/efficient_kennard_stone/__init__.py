"""Fast, memory-efficient exact Kennard-Stone sample selection."""


def _preload_tbb():
    """Make a pip-installed TBB visible to numba.

    The ``tbb`` wheel puts ``libtbb.so.12`` in ``<prefix>/lib``, which is not on
    the loader path of a plain virtualenv, so numba silently falls back to its
    much slower ``workqueue`` threading layer. Loading the library globally
    before numba looks for it fixes that. No-op where not applicable.
    """
    import ctypes
    import os
    import sys

    if not sys.platform.startswith("linux"):
        return
    for name in ("libtbb.so.12", "libtbb.so.2"):
        path = os.path.join(sys.prefix, "lib", name)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            return


_preload_tbb()

from ._kennard_stone import kennard_stone, train_test_split  # noqa: E402

__all__ = ["kennard_stone", "train_test_split"]
__version__ = "0.1.0"

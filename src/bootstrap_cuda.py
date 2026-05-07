"""Pre-load NVIDIA CUDA DLLs on Windows.

ctranslate2 lazy-loads cuBLAS at first inference via LoadLibrary, which doesn't
always honour os.add_dll_directory paths. Pre-loading the DLLs ourselves keeps
them resident in the process; later LoadLibrary("cublas64_12.dll") calls then
return the existing handle by ref-count.

Handles must outlive the call to survive GC, so they're cached at module level.
"""
import os
import sys
import glob


_dll_dir_handles = []
_loaded_dlls = []


def setup_cuda_dlls():
    if sys.platform != 'win32':
        return

    import ctypes

    try:
        import nvidia
    except ImportError:
        return

    bin_dirs = []
    for nvidia_root in nvidia.__path__:
        bin_dirs.extend(sorted(glob.glob(os.path.join(nvidia_root, '*', 'bin'))))
    if not bin_dirs:
        return

    for d in bin_dirs:
        try:
            _dll_dir_handles.append(os.add_dll_directory(d))
        except (AttributeError, OSError):
            pass
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

    # cuDNN libs depend on each other (graph -> ops -> heuristic -> ...), so retry
    # the load loop until no new DLL succeeds — handles ordering implicitly.
    seen = set()
    for _ in range(4):
        progress = False
        for d in bin_dirs:
            for dll_path in glob.glob(os.path.join(d, '*.dll')):
                if dll_path in seen:
                    continue
                try:
                    _loaded_dlls.append(ctypes.WinDLL(dll_path))
                    seen.add(dll_path)
                    progress = True
                except OSError:
                    pass
        if not progress:
            break

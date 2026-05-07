import os
import sys
from dotenv import load_dotenv


# Keep DLL directory cookies AND pre-loaded library handles alive for the process lifetime.
_dll_dir_handles = []
_loaded_dlls = []


def _setup_cuda_dlls():
    """Make cuBLAS / cuDNN DLLs (from pip packages) findable by ctranslate2 on Windows.

    ctranslate2 lazy-loads cuBLAS at first inference via LoadLibrary, which doesn't
    always honor os.add_dll_directory paths. The robust fix is to ctypes.WinDLL the
    DLLs ourselves at startup — once they're loaded into the process, subsequent
    LoadLibrary("cublas64_12.dll") calls return the existing handle by ref-count.
    """
    if sys.platform != 'win32':
        return

    import ctypes
    import glob

    # Find all nvidia/* packages with a bin/ directory containing DLLs.
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

    # Pre-load every .dll in the bin dirs so they live in the process forever.
    # cuDNN's libs depend on each other (graph -> ops -> heuristic ...) so we
    # try repeatedly until no new ones load (handles ordering implicitly).
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
                    pass  # may fail if dependency not yet loaded; retry next pass
        if not progress:
            break


print('Starting WhisperWriter...')
load_dotenv()
_setup_cuda_dlls()

# Make src/ importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Initialize config and pre-load Whisper model BEFORE importing PyQt5.
# (PyQt5's bundled Qt5 DLLs conflict with ctranslate2/CUDA on Windows;
# loading Qt first causes an access-violation crash inside WhisperModel.__init__.)
from utils import ConfigManager
ConfigManager.initialize()

_model_options = ConfigManager.get_config_section('model_options')
_preloaded = None
if not _model_options.get('use_api'):
    from transcription import create_local_model
    _preloaded = create_local_model()

# Now safe to import the Qt-based app.
from main import WhisperWriterApp
_app = WhisperWriterApp(preloaded_model=_preloaded)
_app.run()

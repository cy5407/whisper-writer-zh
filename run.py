import os
import sys
from dotenv import load_dotenv


print('Starting WhisperWriter...')
load_dotenv()

# Make src/ importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from bootstrap_cuda import setup_cuda_dlls
setup_cuda_dlls()

# Initialize config and pre-load Whisper model BEFORE importing PyQt5.
# (PyQt5's bundled Qt5 DLLs conflict with ctranslate2/CUDA on Windows;
# loading Qt first causes an access-violation crash inside WhisperModel.__init__.)
from utils import ConfigManager
ConfigManager.initialize()

_model_options = ConfigManager.get_config_section('model_options')
_preloaded = None
# Only preload when a real config exists AND we're not in API mode. First-run users
# (no config.yaml) shouldn't be forced to download a multi-GB model before the
# settings window opens; if preload fails, let the GUI come up so the user can fix it.
if ConfigManager.config_file_exists() and not _model_options.get('use_api'):
    from transcription import create_local_model
    try:
        _preloaded = create_local_model()
    except Exception as e:
        print(f'[whisper-writer] preload failed ({type(e).__name__}: {e}); '
              f'continuing — adjust model settings in the settings window.',
              file=sys.stderr)

# Now safe to import the Qt-based app.
from main import WhisperWriterApp
_app = WhisperWriterApp(preloaded_model=_preloaded)
_app.run()

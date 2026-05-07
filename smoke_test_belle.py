"""Verify a local Chinese-tuned Whisper model (e.g. BELLE-zh) loads and runs.

Resolution order for the model path:
  1. CLI arg:  python smoke_test_belle.py <model_path>
  2. env var:  BELLE_MODEL_PATH=...
  3. config:   model_options.local.model_path in src/config.yaml
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bootstrap_cuda import setup_cuda_dlls
setup_cuda_dlls()


def resolve_model_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', default=None)
    args = parser.parse_args()
    if args.model_path:
        return args.model_path
    env = os.environ.get('BELLE_MODEL_PATH')
    if env:
        return env
    try:
        from utils import ConfigManager
        ConfigManager.initialize()
        return ConfigManager.get_config_value('model_options', 'local', 'model_path')
    except Exception:
        return None


path = resolve_model_path()
if not path:
    sys.exit('No model path provided. Pass as CLI arg, set BELLE_MODEL_PATH, or '
             'configure model_options.local.model_path in src/config.yaml.')

import numpy as np
from faster_whisper import WhisperModel

print(f'Loading model from {path}...')
m = WhisperModel(path, device='cuda', compute_type='float16')
print('  loaded')

print('Running inference on 3s silent audio (sanity)...')
audio = np.zeros(16000 * 3, dtype=np.float32)
segs, info = m.transcribe(audio, language='zh')
print(f'  result: {repr("".join(s.text for s in segs))}')
print(f'  detected language: {info.language}')
print('BELLE OK')

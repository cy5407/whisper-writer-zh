"""Verify Whisper inference (not just model load) works on this machine."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bootstrap_cuda import setup_cuda_dlls
setup_cuda_dlls()

import numpy as np
from faster_whisper import WhisperModel

print('Loading large-v3 on cuda/float16...')
m = WhisperModel('large-v3', device='cuda', compute_type='float16')
print('  loaded')

# Generate 3 seconds of dummy silent audio (16kHz mono, float32 [-1, 1])
print('Running inference on 3s silent audio...')
audio = np.zeros(16000 * 3, dtype=np.float32)
segments, info = m.transcribe(audio, language='zh')
text = ''.join(s.text for s in segments)
print(f'  result: {repr(text)}')
print(f'  detected language: {info.language}')
print('INFERENCE OK')

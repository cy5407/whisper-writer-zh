"""Verify Whisper inference (not just model load) works on this machine."""
import os, sys, glob

# Replicate run.py CUDA DLL setup
import nvidia, ctypes
bin_dirs = []
for nvidia_root in nvidia.__path__:
    bin_dirs.extend(sorted(glob.glob(os.path.join(nvidia_root, '*', 'bin'))))
_handles = []
_libs = []
for d in bin_dirs:
    _handles.append(os.add_dll_directory(d))
    os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
seen = set()
for _ in range(4):
    progress = False
    for d in bin_dirs:
        for p in glob.glob(os.path.join(d, '*.dll')):
            if p in seen:
                continue
            try:
                _libs.append(ctypes.WinDLL(p))
                seen.add(p)
                progress = True
            except OSError:
                pass
    if not progress:
        break

print(f'Pre-loaded {len(_libs)} DLLs from {len(bin_dirs)} bin dirs')
print(f'  bin_dirs: {bin_dirs}')

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

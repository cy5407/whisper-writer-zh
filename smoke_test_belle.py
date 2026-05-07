"""Verify BELLE Chinese Whisper model loads and runs inference."""
import os, sys, glob, ctypes
import nvidia
bin_dirs = []
for r in nvidia.__path__:
    bin_dirs.extend(sorted(glob.glob(os.path.join(r, '*', 'bin'))))
_h, _l = [], []
for d in bin_dirs:
    _h.append(os.add_dll_directory(d))
    os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
seen = set()
for _ in range(4):
    progress = False
    for d in bin_dirs:
        for p in glob.glob(os.path.join(d, '*.dll')):
            if p in seen: continue
            try: _l.append(ctypes.WinDLL(p)); seen.add(p); progress = True
            except OSError: pass
    if not progress: break

import numpy as np
from faster_whisper import WhisperModel

path = r'C:\Users\cy5407\whisper-writer\models\belle-whisper-zh'
print(f'Loading BELLE-zh from {path}...')
m = WhisperModel(path, device='cuda', compute_type='float16')
print('  loaded')

print('Running inference on 3s silent audio (sanity)...')
audio = np.zeros(16000 * 3, dtype=np.float32)
segs, info = m.transcribe(audio, language='zh')
print(f'  result: {repr("".join(s.text for s in segs))}')
print(f'  detected language: {info.language}')
print('BELLE OK')

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bootstrap_cuda import setup_cuda_dlls
setup_cuda_dlls()

from opencc import OpenCC
cc = OpenCC('s2twp')
print('[OpenCC] 软件 ->', cc.convert('软件'))
print('[OpenCC] 数据库服务器 ->', cc.convert('数据库服务器'))
print('[OpenCC] 我用React写了一个组件 ->', cc.convert('我用React写了一个组件'))

print('[CTranslate2] checking CUDA...')
import ctranslate2
print('  cuda device count:', ctranslate2.get_cuda_device_count())

print('[faster-whisper] loading large-v3 on cuda/float16 (first run will download ~3GB)...')
from faster_whisper import WhisperModel
m = WhisperModel('large-v3', device='cuda', compute_type='float16')
print('  model loaded OK')

print('SMOKE TEST PASSED')

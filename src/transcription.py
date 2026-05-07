import io
import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from openai import OpenAI
from opencc import OpenCC

from utils import ConfigManager

_cc_tw = OpenCC('s2twp')

_LLM_SYSTEM_PROMPT = (
    "你是語音轉譯後處理助手。輸入是台灣使用者的繁體中文語音轉譯文字，"
    "可能包含中英混雜的技術詞彙、遊戲術語、人名地名，以及同音字選錯、漏字、"
    "標點不當的問題。\n\n"
    "任務：\n"
    "1. 修正同音字、近音字、字序錯誤（例如：紙的→紫的、健兵→劍兵、認知更→認知裡）\n"
    "2. 修正不通順的標點符號（移除孤立的全形冒號『：』除非語意需要）\n"
    "3. 補上明顯漏失的字，但不要擴寫、不要縮寫、不要改變語意\n"
    "4. 保留原本的口語特徵（『呢、喔、啦、的』之類語助詞）\n"
    "5. 中英文混雜時保持原樣，不翻譯\n\n"
    "嚴格要求：只輸出修正後的文字本身，不要加任何前綴、解釋、引號、markdown。"
)


def _llm_polish(text):
    """Send transcription through an LLM to fix homophones and punctuation. Returns original on any failure."""
    if not text or not text.strip():
        return text
    pp = ConfigManager.get_config_section('post_processing')
    if not pp.get('llm_polish_enabled'):
        return text

    provider = pp.get('llm_provider', 'anthropic')
    model = pp.get('llm_model', 'claude-haiku-4-5')
    timeout = float(pp.get('llm_timeout', 5.0))

    try:
        if provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                ConfigManager.console_print('[LLM polish] ANTHROPIC_API_KEY not set, skipping.')
                return text
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key, timeout=timeout)
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                system=_LLM_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': text}],
            )
            polished = ''.join(b.text for b in resp.content if hasattr(b, 'text')).strip()
        elif provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                ConfigManager.console_print('[LLM polish] OPENAI_API_KEY not set, skipping.')
                return text
            client = OpenAI(api_key=api_key, timeout=timeout)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[
                    {'role': 'system', 'content': _LLM_SYSTEM_PROMPT},
                    {'role': 'user', 'content': text},
                ],
            )
            polished = (resp.choices[0].message.content or '').strip()
        else:
            ConfigManager.console_print(f'[LLM polish] unknown provider {provider!r}, skipping.')
            return text
    except Exception as e:
        ConfigManager.console_print(f'[LLM polish] failed ({type(e).__name__}: {e}); using raw output.')
        return text

    if not polished:
        return text
    ConfigManager.console_print(f'[LLM polish] {text!r} -> {polished!r}')
    return polished

def create_local_model():
    """
    Create a local model using the faster-whisper library.
    """
    ConfigManager.console_print('Creating local model...')
    local_model_options = ConfigManager.get_config_section('model_options')['local']
    compute_type = local_model_options['compute_type']
    model_path = local_model_options.get('model_path')

    if compute_type == 'int8':
        device = 'cpu'
        ConfigManager.console_print('Using int8 quantization, forcing CPU usage.')
    else:
        device = local_model_options['device']

    try:
        if model_path:
            ConfigManager.console_print(f'Loading model from: {model_path}')
            model = WhisperModel(model_path,
                                 device=device,
                                 compute_type=compute_type,
                                 download_root=None)  # Prevent automatic download
        else:
            model = WhisperModel(local_model_options['model'],
                                 device=device,
                                 compute_type=compute_type)
    except Exception as e:
        ConfigManager.console_print(f'Error initializing WhisperModel: {e}')
        ConfigManager.console_print('Falling back to CPU.')
        model = WhisperModel(model_path or local_model_options['model'],
                             device='cpu',
                             compute_type=compute_type,
                             download_root=None if model_path else None)

    ConfigManager.console_print('Local model created.')
    return model

def transcribe_local(audio_data, local_model=None):
    """
    Transcribe an audio file using a local model.
    """
    if not local_model:
        local_model = create_local_model()
    model_options = ConfigManager.get_config_section('model_options')

    # Convert int16 to float32
    audio_data_float = audio_data.astype(np.float32) / 32768.0

    # Use multi-temperature fallback when config temp is 0.0 (i.e. user didn't override).
    # faster-whisper retries higher temps if compression-ratio / log-prob thresholds fail —
    # important for unclear speech. Passing a single 0.0 DISABLES this safety net.
    cfg_temp = model_options['common']['temperature']
    temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if cfg_temp == 0.0 else cfg_temp

    response = local_model.transcribe(
        audio=audio_data_float,
        language=model_options['common']['language'],
        initial_prompt=model_options['common']['initial_prompt'],
        condition_on_previous_text=model_options['local']['condition_on_previous_text'],
        temperature=temperature,
        vad_filter=model_options['local']['vad_filter'],
        beam_size=10,             # default 5 — wider beam helps unclear speech
        patience=2.0,             # default 1.0 — let beam search explore longer
        no_speech_threshold=0.5,  # default 0.6 — less aggressive silence classification
    )
    return ''.join([segment.text for segment in list(response[0])])

def transcribe_api(audio_data):
    """
    Transcribe an audio file using the OpenAI API.
    """
    model_options = ConfigManager.get_config_section('model_options')
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY') or None,
        base_url=model_options['api']['base_url'] or 'https://api.openai.com/v1'
    )

    # Convert numpy array to WAV file
    byte_io = io.BytesIO()
    sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate') or 16000
    sf.write(byte_io, audio_data, sample_rate, format='wav')
    byte_io.seek(0)

    response = client.audio.transcriptions.create(
        model=model_options['api']['model'],
        file=('audio.wav', byte_io, 'audio/wav'),
        language=model_options['common']['language'],
        prompt=model_options['common']['initial_prompt'],
        temperature=model_options['common']['temperature'],
    )
    return response.text

def post_process_transcription(transcription):
    """
    Apply post-processing to the transcription.
    Pipeline: strip -> OpenCC s2twp 繁化 -> optional LLM polish -> trailing tweaks.
    """
    transcription = transcription.strip()
    transcription = _cc_tw.convert(transcription)
    transcription = _llm_polish(transcription)
    post_processing = ConfigManager.get_config_section('post_processing')
    if post_processing['remove_trailing_period'] and transcription.endswith('.'):
        transcription = transcription[:-1]
    if post_processing['add_trailing_space']:
        transcription += ' '
    if post_processing['remove_capitalization']:
        transcription = transcription.lower()

    return transcription

def transcribe(audio_data, local_model=None):
    """
    Transcribe audio date using the OpenAI API or a local model, depending on config.
    """
    if audio_data is None:
        return ''

    if ConfigManager.get_config_value('model_options', 'use_api'):
        transcription = transcribe_api(audio_data)
    else:
        transcription = transcribe_local(audio_data, local_model)

    return post_process_transcription(transcription)


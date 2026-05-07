"""Lightweight i18n for WhisperWriter UI. Lookup is identity-on-miss so partial translations work."""
from utils import ConfigManager

_TRANSLATIONS = {
    'en': {},
    'zh_TW': {
        # Buttons
        'Start': '開始',
        'Settings': '設定',
        'Save': '儲存',
        'Browse': '瀏覽',
        'Reset to saved settings': '還原為已儲存的設定',

        # Status window
        'Recording...': '錄音中...',
        'Transcribing...': '轉譯中...',
        'WhisperWriter Status': 'WhisperWriter 狀態',

        # Dialogs
        'Description': '說明',
        'Settings Saved': '設定已儲存',
        'Settings have been saved. The application will now restart.':
            '設定已儲存，應用程式即將重新啟動。',
        'Close without saving?': '不儲存就關閉？',
        'Are you sure you want to close without saving?':
            '確定要在未儲存的情況下關閉嗎？',
        'Select Whisper Model File': '選擇 Whisper 模型檔',
        'Model Files (*.bin);;All Files (*)': '模型檔 (*.bin);;所有檔案 (*)',

        # System tray
        'WhisperWriter Main Menu': 'WhisperWriter 主選單',
        'Open Settings': '開啟設定',
        'Exit': '結束',

        # Settings tab names (derived from schema category keys via .replace('_',' ').capitalize())
        'Model options': '模型',
        'Recording options': '錄音',
        'Post processing': '後處理',
        'Misc': '其他',

        # Settings field labels (from schema keys via .replace('_',' ').capitalize())
        'Use api': '使用 API',
        'Common': '共通',
        'Api': 'API',
        'Local': '本地',
        'Language': '語言',
        'Temperature': 'Temperature',
        'Initial prompt': '初始提示',
        'Model': '模型',
        'Base url': 'Base URL',
        'Api key': 'API 金鑰',
        'Device': '裝置',
        'Compute type': '計算精度',
        'Condition on previous text': '參考上一段文字',
        'Vad filter': 'VAD 靜音過濾',
        'Model path': '模型路徑',
        'Activation key': '啟動快捷鍵',
        'Input backend': '輸入後端',
        'Recording mode': '錄音模式',
        'Sound device': '音訊裝置',
        'Sample rate': '取樣率',
        'Silence duration': '靜音判定時間 (ms)',
        'Min duration': '最短錄音時間 (ms)',
        'Writing key press delay': '輸出按鍵間隔 (s)',
        'Remove trailing period': '移除句尾句點',
        'Add trailing space': '結尾加空白',
        'Remove capitalization': '全部轉小寫',
        'Input method': '輸入模擬方式',
        'Print to terminal': '在終端機印出狀態',
        'Hide status window': '隱藏狀態視窗',
        'Noise on completion': '完成時播放提示音',
    },
}

_lang_cache = None


def get_lang():
    global _lang_cache
    if _lang_cache is None:
        _lang_cache = ConfigManager.get_config_value('misc', 'language') or 'zh_TW'
    return _lang_cache


def reset_lang_cache():
    global _lang_cache
    _lang_cache = None


def tr(text):
    if text is None:
        return text
    return _TRANSLATIONS.get(get_lang(), {}).get(text, text)

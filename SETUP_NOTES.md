# WhisperWriter 安裝筆記（cy5407 客製版）

在原版 [savbell/whisper-writer](https://github.com/savbell/whisper-writer) 之上做了三項修改，目的：**繁體中文 + 中英混合 + 右 Ctrl PTT，用在 Claude Code / Discord / 任何文字輸入視窗**。

## 環境

- Windows 11、Python 3.12.10
- NVIDIA RTX 5070 Ti（Blackwell, 16GB VRAM）
- 模型：large-v3，cuda + float16

## 套件

走 pip 最新版（原 `requirements.txt` 版本太舊，且檔案是 UTF-16 LE，無法直接用）：

```
faster-whisper 1.2.1
ctranslate2 4.7.1            # 支援 Blackwell sm_120
nvidia-cublas-cu12 12.9.2.10
nvidia-cudnn-cu12 9.21.1.3
opencc-python-reimplemented  # 繁化
PyQt5 5.15.11
pynput 1.8.1
sounddevice / soundfile / pyperclip / PyYAML / openai
audioplayer / webrtcvad-wheels
```

## 踩到的坑與解法

### 1. PyQt5 + CUDA DLL 載入衝突（Windows）

**症狀**：`Creating local model...` 後 access violation segfault，stack trace 指到 `faster_whisper/transcribe.py:689`。

**原因**：PyQt5 5.15 自帶的 Qt5 DLL 在 ctranslate2 載入 CUDA library 之前先佔用了某些 CRT/OpenMP 符號，造成 native crash。

**解法**：在 import PyQt5 之前先把 Whisper 模型載完。重寫 `run.py`：
- 先做 CUDA DLL path 設定
- 先 `from transcription import create_local_model` 並 instantiate model
- 最後才 `from main import WhisperWriterApp`

對應 main.py 改了：`WhisperWriterApp.__init__(self, preloaded_model=None)`，`initialize_components` 優先用 preloaded。

### 2. CUDA DLL 找不到

**症狀**：純 venv 跑 ctranslate2 找不到 cuBLAS / cuDNN。

**解法**：`run.py` 啟動時呼叫 `_setup_cuda_dlls()`，從 `nvidia.cublas.lib` / `nvidia.cudnn.lib` 取出路徑，呼叫 `os.add_dll_directory()` 並塞進 `PATH`。

### 3. CUDA DLL 在 inference 時找不到（cublas64_12.dll not found）

**症狀**：模型 `Local model created.` 看似成功，但第一次 transcribe 時噴 `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`。

**原因**（三層）：

1. **`os.add_dll_directory()` 回傳的 handle 被 GC 後路徑就失效**——必須把 handle 保存在某個全域變數，否則只在當下一行有效。
2. **ctranslate2 的 cuBLAS 是 lazy load**：model __init__ 不碰 cuBLAS，第一次 `encode()` 才動態載入。所以即使 cuDNN 在初始化時拿得到，cuBLAS 載入時 search path 已失效。
3. **少裝了 `nvidia-cuda-runtime-cu12`**——cuBLAS 依賴 `cudart64_12.dll`，但 GPU driver 不會帶這個 runtime DLL，得透過 pip 套件補。

**解法**：

```python
# run.py（也同步到 src/main.py）
_dll_dir_handles = []   # ← 必須保留 handle 物件，避免 GC 移除路徑
_loaded_dlls = []       # ← 用 ctypes.WinDLL 強制 pre-load 所有 DLL，process 持有 ref-count

def _setup_cuda_dlls():
    if sys.platform != 'win32':
        return
    import ctypes, glob
    import nvidia
    bin_dirs = []
    for nvidia_root in nvidia.__path__:   # nvidia 是 namespace package，沒有 __file__
        bin_dirs.extend(sorted(glob.glob(os.path.join(nvidia_root, '*', 'bin'))))
    for d in bin_dirs:
        _dll_dir_handles.append(os.add_dll_directory(d))
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
    # Brute-force pre-load every DLL with retry loop (handles inter-DLL deps)
    seen = set()
    for _ in range(4):
        progress = False
        for d in bin_dirs:
            for p in glob.glob(os.path.join(d, '*.dll')):
                if p in seen: continue
                try:
                    _loaded_dlls.append(ctypes.WinDLL(p))
                    seen.add(p)
                    progress = True
                except OSError:
                    pass
        if not progress: break
```

加裝套件：
```
pip install nvidia-cuda-runtime-cu12
```

驗證：跑 `smoke_test_v2.py`，會在 silent audio 上做 inference 拿到結果（Whisper 對靜音常 hallucinate「字幕志愿者 楊茜茜」字串，這是正常的）。

### 4. 中文 config 被當 cp950 解析

**症狀**：config.yaml 含 `initial_prompt` 中文字，啟動時 `UnicodeDecodeError: 'cp950' codec...`。

**原因**：`src/utils.py` 三個 `open(...)` 沒指定 encoding，Windows 中文 locale 預設 cp950。

**解法**：三處都改成 `open(..., encoding='utf-8')`，`yaml.dump` 加 `allow_unicode=True`。

## OpenCC 繁化注入

Whisper / faster-whisper 對中文預設輸出簡體（即使 initial_prompt 偏置繁體仍會偷渡簡中）。仿 Will 保哥 ZeroType 的做法，在 post-processing 加 OpenCC `s2twp`（簡 → 台灣正體含詞彙轉換）。

`src/transcription.py`：

```python
from opencc import OpenCC
_cc_tw = OpenCC('s2twp')

def post_process_transcription(transcription):
    transcription = transcription.strip()
    transcription = _cc_tw.convert(transcription)   # ← 注入點
    ...
```

效果：`軟件 → 軟體`、`组件 → 元件`、`数据库服务器 → 資料庫伺服器`。

## 設定（src/config.yaml）

| 項目 | 值 | 備註 |
|---|---|---|
| `model_options.local.model` | `large-v3` | 16GB VRAM 跑得動 |
| `model_options.local.device` | `cuda` | |
| `model_options.local.compute_type` | `float16` | |
| `model_options.common.language` | `zh` | |
| `model_options.common.initial_prompt` | （繁體中文提示句） | 偏置模型輸出繁體 |
| `recording_options.activation_key` | `ctrl_right` | 右 Ctrl |
| `recording_options.recording_mode` | `hold_to_record` | 按住說話 |
| `recording_options.input_backend` | `auto` | 不要設 `pynput`，會在模型載入時就啟動 keyboard hook |

## 啟動方式

雙擊 `launch.bat`，或：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\venv\Scripts\python.exe run.py
```

第一次啟動：在 WhisperWriter 主視窗按 **Start** 啟用 keyboard listener。
之後在任何視窗按住右 Ctrl 講話，放開 → 文字自動 typewrite 進 focused 視窗。

## 適用情境

任何能輸入文字的地方：

- Claude Code TUI（Windows Terminal / WezTerm）
- Discord 桌面版（含 openab ACP bridge 路徑）
- 瀏覽器（Claude.ai / ChatGPT / Gemini）
- VS Code、Cursor、Slack、Notion、Email

不要用在密碼欄位。部分反作弊遊戲會擋合成按鍵事件。

---

## 初步使用觀察與待修正方向

第一輪實測的觀察記錄，作為日後調整的依據。狀態欄位：✅ 已處理 / 🟡 已調整待驗證 / 🔴 待解決。

### 準確度問題

**情境**：使用者表示自身發音不算清楚，初版設定下辨識率不足。

| 旋鈕 | 原值 | 調後 | 狀態 |
|---|---|---|---|
| `temperature` | `0.0`（單一值，**會關掉 fallback**） | `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` 多溫度 fallback | 🟡 |
| `beam_size` | 5（預設） | 10 | 🟡 |
| `patience` | 1.0（預設） | 2.0 | 🟡 |
| `no_speech_threshold` | 0.6（預設） | 0.5 | 🟡 |
| `condition_on_previous_text` | `true`（GUI 存檔時被覆蓋回預設） | `false`（避免短 PTT 模式 cascade hallucination） | 🟡 |

實作位置：`src/transcription.py` 的 `transcribe_local()`，`src/config.yaml` 的 `local.condition_on_previous_text`。

**代價**：每次 inference 慢 1.5–2 倍（短句感覺不到）。

**未來可再嘗試**：
- 麥克風前端做 noise gate / volume normalize
- 用 LLM 做後處理（送 transcription 給 GPT-4 / Claude 校稿）
- 訓練個人化 LoRA（ASR fine-tune），但成本太高，先不考慮
- 替使用者常講的人名、術語客製 `initial_prompt`

**已執行**：換成中文 fine-tuned 模型 **BELLE-2/Belle-whisper-large-v3-zh**。

  - 架構與 `large-v3` 完全一樣，可無痛 swap（透過 `model_path` 指定）
  - 步驟（記錄供日後重做或換其他模型）：
    ```powershell
    pip install transformers accelerate torch
    ct2-transformers-converter `
      --model BELLE-2/Belle-whisper-large-v3-zh `
      --output_dir C:\Users\cy5407\whisper-writer\models\belle-whisper-zh `
      --quantization float16 `
      --copy_files tokenizer.json preprocessor_config.json
    ```
  - config.yaml：`model_options.local.model_path: C:\...\models\belle-whisper-zh`（要切回原版 large-v3 把 model_path 設回 null 即可）
  - 已知特徵：對靜音幻覺出「请不吝点赞订阅打赏支持...」之類字幕用語（fine-tune 訓練資料含大量 YouTube/B站 字幕）。不影響實際語音的辨識，且 OpenCC 會繁化掉。
  - 狀態：🟡 待驗證實際使用準確度

**已執行**：建立 LLM 後處理層（預設 OFF，待你填 API key 後開啟）。

  - Pipeline：Whisper 輸出 → OpenCC 繁化 → **LLM polish（修同音字 + 補標點）** → typewrite
  - 實作位置：`src/transcription.py` 的 `_llm_polish()` + `post_process_transcription()`
  - 預設 backend：**Claude Haiku 4.5**（`claude-haiku-4-5`）
    - 為什麼不是 Sonnet/Opus：這個 task 是「短文字、規則明確、不需要推理」，Haiku 足夠且快 3-5 倍、便宜 10-50 倍。互動式語音輸入對延遲很敏感。
  - 啟用步驟：
    1. 在 `.env` 加上：
       ```
       ANTHROPIC_API_KEY=sk-ant-...
       ```
    2. `src/config.yaml` 把 `post_processing.llm_polish_enabled` 改成 `true`
    3. 重啟 app
  - 切換到別的模型/廠牌：改 `config.yaml`：
    ```yaml
    post_processing:
      llm_provider: anthropic   # 或 openai
      llm_model: claude-haiku-4-5   # 或 claude-sonnet-4-6 / claude-opus-4-6 / gpt-4o-mini / gpt-4o
      llm_timeout: 5.0
    ```
  - **失敗 fallback**：API key 缺、API 呼叫失敗、timeout，都會自動退回原始 Whisper 輸出，不會卡住輸入流程
  - **Log 顯示**：每次 polish 會在 log 印出 `[LLM polish] '原文' -> '修正後'`，方便看效果與除錯
  - 狀態：🟢 已實作；🟡 待你填 API key 後驗證實際品質
  - 預估成本（Haiku 4.5）：每段 ~50 tokens in / 50 out → ~$0.0005／段，每天 100 段 ≈ 1.5 美分

### 標點符號問題：莫名出現的 `：`

**症狀**：轉譯結果常出現孤立的全形冒號 `：`，例如句首或段落中間，沒有對應的「說話/標題」語意。

**根因分析**：

1. **訓練資料偏差**：Whisper 訓練語料含大量字幕、對話、引述格式，模型內部 prior 偏好補出 `說：` / `題目：` / `說明：` 之類結構。當錄音突然開始或語氣停頓時，模型「腦補」一個前綴然後省略主詞，留下孤立冒號。
2. **`initial_prompt` 含冒號**：目前 prompt 是
   ```
   以下是普通話的句子，請使用繁體中文輸出，可能包含中英文混雜的技術詞彙，例如：軟體、程式、資料、伺服器、TypeScript、React、API、Claude Code、Discord。
   ```
   裡面 `：` 出現過一次（「例如：」），會讓模型微幅偏向產生 `：`。
3. **OpenCC `s2twp` 不會碰標點**：所以不是後處理引入的。

**狀態**：🔴 待解決。

**可能修法**（擇一或組合）：

- **Prompt 改寫**：把 `initial_prompt` 裡的 `：` 改成逗號或刪除；測試模型是否減少冒號輸出
- **Post-processing regex**：在 `post_process_transcription()` 加：
  ```python
  import re
  # 移除孤立冒號（前後沒文字緊貼）、行首冒號、句末冒號
  transcription = re.sub(r'^[：:]\s*', '', transcription)
  transcription = re.sub(r'\s*[：:]$', '', transcription)
  transcription = re.sub(r'(?<=[。！？\s])[：:](?=\s|$)', '', transcription)
  ```
- **送 LLM 修標點**：交給小型 LLM（gpt-4o-mini / claude-haiku）做 punctuation polish，順便修錯字

注：句號（`。` / `.`）目前正常，符合語料合理結尾，**不需要動**。

### 其他觀察（待累積）

- 結尾空白：`add_trailing_space: true`（目前的設定）每段轉譯後加一個空白；用在 Claude Code 連續輸入感受是「自然」還是「多餘」待觀察。
- VAD 邊界：`silence_duration: 900`（ms）— 講話中間停頓 < 0.9 秒就不會切。如果想斷句更積極，可降到 600；但太短會誤切。
- 啟動到第一次 inference 的延遲：模型載入 ~5 秒、第一次 transcribe 多花 2-3 秒做 lazy CUDA kernel JIT，第二次起就正常 <1 秒。

### 中英切換無法生效

**症狀**：講中文沒問題；嘗試講英文（例如想口述 markdown 語法 `# heading` 之類）時，輸出仍被強制轉成中文同音字，無法切到英文模式。

**根因**：`src/config.yaml` 的 `model_options.common.language: zh` 寫死語言為中文。Whisper 不論輸入是什麼語言，都會嘗試硬解碼成中文。BELLE-zh 的中文 fine-tune 又進一步加深這個 bias。

**解法**（按工程量排序）：

1. **Auto-detect**（最簡單）：把 `language` 改成 `null`，Whisper 會自動偵測每段 audio 的語言。代價：
   - 每段多 ~100-200 ms（語言偵測步驟）
   - 短句、噪音音訊偶爾會誤判
   - BELLE-zh 是中文 fine-tuned，對英文偵測能力可能弱於原版 large-v3

2. **雙熱鍵切換**：右 Ctrl = 中文（BELLE-zh），右 Alt = 英文（large-v3）。要改 WhisperWriter 的 `key_listener.py` 支援多組 activation key，並動態載入兩個模型（記憶體佔用 ~6GB）。

3. **Tray menu / 快速 toggle**：系統列加一個切換 zh/en 的選項，當下狀態存在 ConfigManager。改動小但每次手動切麻煩。

4. **Prompt 加雙語 hint**：保持 `language: null` + 在 `initial_prompt` 寫「以下可能是繁體中文或英文，請依音訊判斷」，讓模型同時兼顧。

**狀態**：🟡 已實作 #1（auto-detect）。如果英文偵測效果不夠好，再上 #2（雙模型雙熱鍵）。

實作位置：`src/config.yaml` `model_options.common.language: null`。

---

## 問題總結（截至本次 session）

整理所有遇到的痛點與當前狀態，作為日後複盤與優化的索引。

| # | 議題 | 狀態 | 處置方向 |
|---|---|---|---|
| 1 | 套件版本太舊不相容（requirements.txt UTF-16 + numpy 1.24 不支援 Py3.12） | ✅ | 改裝最新版套件 |
| 2 | PyQt5 Qt5 DLL 與 ctranslate2 CUDA 在啟動順序上衝突，access violation | ✅ | run.py 在 import PyQt5 前先 preload Whisper 模型 |
| 3 | Windows DLL 找不到 cudart/cuBLAS（lazy load + add_dll_directory handle GC + 缺套件） | ✅ | pip install nvidia-cuda-runtime-cu12 + ctypes.WinDLL pre-load + 保留 dll handle |
| 4 | 中文 config.yaml 被 cp950 解析失敗 | ✅ | utils.py 開檔指定 encoding='utf-8' |
| 5 | Whisper 輸出簡體中文 | ✅ | OpenCC `s2twp` 後處理繁化 |
| 6 | 對含糊發音準確度不夠 | 🟡 | 多溫度 fallback、beam_size=10、patience=2.0、no_speech_threshold=0.5、condition_on_previous_text=false |
| 7 | 大量同音字選錯（紙/紫、健兵/劍兵、刀/氘、驅毒/驅逐、飆三/標三 等） | 🟡 | 換 BELLE-zh 中文 fine-tuned 模型；上 LLM polish（待填 API key） |
| 8 | 句中孤立 `：` 莫名出現 | 🔴 | 待修：prompt 移除 `：` / regex 後處理 / LLM polish 順便修 |
| 9 | 介面英文不友善 | ✅ | 加 i18n module，預設繁體中文，可在設定 → 其他 → 語言切換 |
| 10 | 沒有檔案 log，stdout 視窗關掉就遺失 | ✅ | launch.bat 加上 `>>logs/whisper-writer.log` 並用 `python -u` 即時 flush |
| 11 | **中英文切換不過來，講英文輸出仍被當中文** | 🟡 | 已開 auto-detect (language=null)；觀察效果，必要時加雙模型雙熱鍵 |
| 12 | 對外放音訊（喇叭播 YouTube）辨識崩潰 | 📌 | 非 dictation 公平場景，建議用 headset 直接對著講；若要做 stream 轉譯需另一條 pipeline |

圖例：✅ 已處理 ／ 🟡 已處置但待驗證 ／ 🔴 未解 ／ 📌 設計上不在 scope 內

### 後續優先順序建議

1. 你填 ANTHROPIC_API_KEY 並打開 `llm_polish_enabled` → 解掉 #6/#7/#8 的大半（最高 CP）
2. 觀察 #11 auto-detect 效果，必要時上雙熱鍵
3. 累積一週實際使用 → 看哪些 issue 仍痛 → 決定下一波 LoRA / 客製 prompt / 雙模型 / 音訊前處理

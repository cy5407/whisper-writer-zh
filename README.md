# WhisperWriter 繁中客製版

[savbell/whisper-writer](https://github.com/savbell/whisper-writer) 的 Windows / 繁體中文 / 注音輸入法友好分支。

按住右 Ctrl 講話 → 放開 → 文字自動貼進當前 focused 視窗。

## 跟原版不一樣的地方

| 項目 | 原版 | 本分支 |
|---|---|---|
| 中文輸出 | 簡體 | OpenCC `s2twp` 繁化（含詞彙轉換：軟件→軟體、组件→元件） |
| 中英混合 | 需指定語言 | `language: null` auto-detect，純英文 / 中英混雜都能用 |
| Windows CUDA | 需手動裝 cuBLAS / cuDNN | 自動 pre-load `nvidia-*` pip 套件，支援 Blackwell sm_120 |
| 注音 IME 衝突 | pynput 模擬按鍵被輸入法攔截 | 剪貼簿模式繞過 IME（用 Win32 SendInput Ctrl+V 注入） |
| 終端機貼上 | 只支援 Ctrl+V | 可選 Ctrl+Shift+V（WezTerm / Windows Terminal）或 clipboard-only |
| LLM 後處理 | 無 | 選用 Claude Haiku / GPT 修同音字與標點 |
| UI 語言 | 英文 | 繁體中文 i18n |
| 中文模型 | 預設 large-v3 | 支援 BELLE-2/Belle-whisper-large-v3-zh fine-tuned |
| log 隱私 | 寫逐字稿全文 | 預設只記字數與耗時，逐字稿需開 debug flag |

## 環境

- Windows 11
- Python 3.12
- NVIDIA GPU（可選；CPU 也能跑但慢）

## 安裝

```powershell
git clone https://github.com/cy5407/whisper-writer-zh
cd whisper-writer-zh

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-cuda.txt   # 只有用 CUDA 才需要
```

### 用 BELLE-zh 中文 fine-tuned 模型（選用，準確度更好）

```powershell
pip install transformers accelerate torch
ct2-transformers-converter `
  --model BELLE-2/Belle-whisper-large-v3-zh `
  --output_dir models\belle-whisper-zh `
  --quantization float16 `
  --copy_files tokenizer.json preprocessor_config.json
```

之後在設定 → 模型 → 本地 → **模型路徑** 填 `models\belle-whisper-zh`。

## 啟動

雙擊 `launch.bat`（背景跑、log 寫到 `logs/whisper-writer.log`），或：

```powershell
.\venv\Scripts\python.exe run.py
```

主視窗按 **Start** 啟用 keyboard listener。之後在任何視窗按住右 Ctrl 講話 → 放開 → 文字自動出現。

第一次啟動會比較慢（模型載入 ~5 秒、首次 transcribe 多 ~3 秒做 CUDA kernel JIT），第二次起 < 1 秒。

## 推薦設定

打開 WhisperWriter 主視窗 → **設定**：

| 分頁 | 欄位 | 建議值 | 為什麼 |
|---|---|---|---|
| 模型 → 本地 | 模型 | `large-v3` 或 BELLE-zh 路徑 | BELLE-zh 中文準確度更好 |
| 模型 → 本地 | 裝置 | `cuda` | 有 GPU 就用 |
| 模型 → 本地 | 計算精度 | `float16` | 16GB VRAM 跑得動 |
| 模型 → 本地 | VAD 靜音過濾 | `false` | 開啟容易把停頓誤判成靜音砍掉 |
| 模型 → 共通 | 語言 | 留空 (`null`) | auto-detect 中英 |
| 錄音 | 啟動快捷鍵 | `ctrl_right` | 不擋一般打字 |
| 錄音 | 錄音模式 | `hold_to_record` | 按住說話最直覺 |
| 後處理 | 輸入模擬方式 | `clipboard` | 繞過注音 IME |
| 後處理 | 剪貼簿貼上熱鍵 | 看你主用什麼視窗 | 見下表 |

### 剪貼簿貼上熱鍵怎麼選

| 主要使用視窗 | 選 |
|---|---|
| Discord / Notepad / 瀏覽器 / Slack | `Ctrl+V` |
| WezTerm / Windows Terminal / 其他終端機 | `Ctrl+Shift+V` |
| 跨多種視窗、想自己決定貼到哪 | `none`（手動 paste） |

## LLM 後處理（選用，修同音字 + 補標點）

每段轉譯多 ~500ms 但會大幅改善「紙的→紫的」「健兵→劍兵」這類同音字錯誤。

1. 在專案根目錄建 `.env`：
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   （或 `OPENAI_API_KEY=...`，要用哪個 provider 就填哪個）

2. 設定 → 後處理 → **啟用 LLM polish** 打勾 → 儲存

預設用 Claude Haiku 4.5（一段 ~0.0005 USD，每天 100 段約 1.5 美分）。要換 Sonnet / Opus / GPT 改 `llm_provider` 與 `llm_model`。

## 已知問題

- **純靜音輸入會 hallucinate**：BELLE-zh 對沒講話的音訊會出「字幕志願者 楊茜茜」「請訂閱頻道」之類字串（訓練資料含大量 YouTube 字幕）。實際語音不受影響，OpenCC 會繁化掉。
- **反作弊遊戲擋合成按鍵**：clipboard 模式 SendInput 仍會被擋。改用 `none` 模式手動貼。
- **剪貼簿暫時被佔用 ~300ms**：自動貼上模式期間若手動複製別的東西會被洗掉。實務上很少遇到。
- **不還原圖片剪貼簿**：只還原文字。剛複製過圖片再講話，圖片會沒了。

## 啟動時的常見錯誤

- `cublas64_12.dll not found` → 沒裝 `nvidia-cuda-runtime-cu12`，跑 `pip install -r requirements-cuda.txt`
- `UnicodeDecodeError: 'cp950'` → 已修，更新到最新版即可
- 「Creating local model...」後 access violation → PyQt5 與 CUDA DLL 載入順序問題，用 `run.py` 啟動而非直接跑 `src/main.py`

## Credits

- [savbell/whisper-writer](https://github.com/savbell/whisper-writer) — 原專案
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 推論引擎
- [OpenCC](https://github.com/BYVoid/OpenCC) — 簡繁轉換
- [BELLE-2/Belle-whisper-large-v3-zh](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh) — 中文 fine-tuned 模型

## License

GPL（沿用原專案）。詳見 [LICENSE](LICENSE)。

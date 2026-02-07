# DIFFA 環境構築ガイド（Windows + uv）

公式READMEはconda + pip構成だが、本環境ではuvを使用して構築した。
Windows 11 + NVIDIA GPU（16GB VRAM）での推論までの手順と、遭遇した問題・回避策をまとめる。

## 動作確認済み環境

| 項目 | 値 |
|------|-----|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4070 Ti SUPER (16GB VRAM) |
| Python | 3.10 (uv管理) |
| CUDA | 12.8 (PyTorch同梱) |

## 1. uv のインストール

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. リポジトリのクローンとPython環境の作成

```bash
git clone https://github.com/NKU-HLT/DIFFA.git
cd DIFFA
uv python install 3.10
```

`.python-version` に `3.10` が記載されており、uvが自動的にこのバージョンを使用する。

## 3. pyproject.toml による依存関係管理

公式の `requirements.txt` の代わりに `pyproject.toml` で依存関係を管理している。
主なポイント:

- `transformers==4.49.0` — **バージョン固定必須**（新しいバージョンではモデルが動かない）
- `torch>=2.7.0`, `torchaudio>=2.7.0` — CUDA 12.8版をPyTorch公式インデックスから取得
- `deepspeed>=0.17.0 ; sys_platform == 'linux'` — Windowsでは不要のためLinux限定
- `gradio>=5.0.0` — Web UIデモ用

PyTorchのCUDA版を取得するため、`pyproject.toml` に以下のインデックス設定がある:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
```

## 4. 依存関係のインストール

```bash
uv sync
```

これだけで仮想環境の作成と全パッケージのインストールが完了する。

## 5. モデルのダウンロード

3つの事前学習モデルが必要。`./models/` ディレクトリに配置する。

### 注意: huggingface-cli は使えない

Windows日本語ロケール環境では `huggingface-cli download` がUnicodeエラーで失敗する（非推奨警告に含まれる絵文字が原因）。
代わりにPython APIの `snapshot_download` を使う:

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('GSAI-ML/LLaDA-8B-Instruct', local_dir='./models/LLaDA-8B-Instruct')
snapshot_download('openai/whisper-small', local_dir='./models/whisper-small')
snapshot_download('zhoujiaming777/DIFFA', local_dir='./models/DIFFA')
"
```

ダウンロード後のディレクトリ構成:

```
models/
  LLaDA-8B-Instruct/    # ~16GB (bfloat16)
  whisper-small/         # ~461MB
  DIFFA/                 # アダプタ重み
```

## 6. コード修正（必須）

### DIFFAModel のトークナイザパス

`src/modeling_DIFFA.py` の `__init__` にトークナイザパスが `/path/to/models/LLaDA-8B-Instruct` とハードコードされている。
推論スクリプト側で `DIFFAModel.from_pretrained(model_path, tokenizer=tokenizer)` のように `tokenizer=` キーワード引数を渡すことで回避している（`demo_inference.py` は対応済み）。

### 音声の読み込みに soundfile を使用

`torchaudio.load()` は新しいバージョンで `torchcodec` を要求するため、代わりに `soundfile` を使用:

```python
import soundfile as sf
audio, sr = sf.read(wav_path, dtype="float32")
```

## 7. 推論の実行

### CLI（コマンドライン）

```bash
uv run python demo_inference.py \
    --audio_path ./sample.wav \
    --question "What is the speaker saying?" \
    --steps 16 --block_length 16 --max_new_tokens 64
```

### Web UI（Gradio）

```bash
uv run python demo_gui.py
```

ブラウザで http://localhost:7860 を開き、音声ファイルをアップロードして質問を入力する。

## 8. VRAM 16GB でのメモリ管理

LLaDA-8B（bfloat16）は約16GBのVRAMを使用する。RTX 4070 Ti SUPER（16GB）ではギリギリなので、`device_map="auto"` を使ってGPUに載りきらない部分をCPUにオフロードしている:

```python
model.llm_model = AutoModel.from_pretrained(
    llm_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

`accelerate` パッケージが自動的にGPU/CPUの配置を決定する。

## 既知の警告（無視して問題ない）

| 警告メッセージ | 説明 |
|---------------|------|
| `Missing keys in state_dict: ['whisper_model.*']` | Whisper重みはDIFFAチェックポイントに含まれず、別途ロードされるため正常 |
| `The model weights are not tied` | LLaDAモデルの仕様。動作に影響なし |
| `Some parameters are on the meta device because they were offloaded to the cpu` | `device_map="auto"` によるCPUオフロード。正常動作 |

## 公式手順との差分まとめ

| 項目 | 公式 (README) | 本環境 |
|------|---------------|--------|
| パッケージ管理 | conda + pip | uv |
| 依存定義 | requirements.txt | pyproject.toml |
| PyTorch CUDA | 手動指定 | uv.sources でCU128インデックス指定 |
| モデルDL | huggingface-cli | Python API (`snapshot_download`) |
| 音声読み込み | torchaudio | soundfile |
| DeepSpeed | 必須 | Linux限定（Windows推論では不要） |
| トークナイザ | ハードコードパス | `tokenizer=` kwarg で渡す |

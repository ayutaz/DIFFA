# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

DIFFA（Large Language Diffusion Models Can Listen and Understand）は、拡散ベースの大規模音声言語モデル（LALM）の研究プロジェクト。凍結されたLLaDA-8B（拡散LLM）をバックボーンとし、Whisper-Smallエンコーダで音声特徴を抽出し、Dual Adapter（セマンティック＋アコースティック）で音声知覚を実現する。AAAI 2026採択済み。

## セットアップ

```bash
conda create -n diffa python=3.10
conda activate diffa
pip install -r requirements.txt
```

必要な事前学習モデル（パスはスクリプト内で設定）:
- LLaDA-8B-Instruct
- Whisper-Small (openai/whisper-small)
- DIFFAチェックポイント (huggingface: zhoujiaming777/DIFFA)

## 主要コマンド

### 学習
```bash
# Stage 1: ASR事前学習（LibriSpeech 960h）- GPU4枚、分散学習
bash train_stage1.sh

# Stage 2: 指示追従QAタスク - Stage 1チェックポイントから継続
bash train_stage2.sh
```

学習スクリプトの中で `llm_model`、`whisper_model`、`stage1_ckpt`（Stage 2のみ）のパスを設定する必要がある。

### 推論・評価
```bash
# MMSU ベンチマーク（推論＋評価を一括実行）
bash run_mmsu_inference.sh

# 個別の推論スクリプト
python inference_stage2_mmsu.py --model_path <path> --output_jsonl <path> --steps 4 --block_length 4 --max_new_tokens 4
python inference_stage2_mmau.py --model_path <path> --output_json <path>
python inference_stage2_voicebench.py --model_path <path> --output_json <path>

# 評価スクリプト
python mmsu/evaluate.py <result.jsonl>
python mmau/evaluate.py <result.json>
```

## アーキテクチャ

### 2段階学習パイプライン
- **Stage 1**: ASRタスク。WhisperProjector（セマンティックアダプタ）のみ学習。LibriSpeechの音声→転写テキスト。
- **Stage 2**: QA/指示追従タスク。QformerConnector（アコースティックアダプタ）を追加し、Dual Adapterで学習。Stage 1チェックポイントをロードして継続。

### コアモジュール（`src/`）
- **`modeling_DIFFA.py`**: メインモデルクラス `DIFFAModel`。WhisperEncoder→Adapter→LLaDA統合、学習ステージ分岐の制御。
- **`llm_generate.py`**: 拡散生成アルゴリズム。反復的なマスク→予測→リマスクで非自己回帰的にテキスト生成。CFG（Classifier-Free Guidance）とセミオートレグレッシブ（ブロック単位生成）に対応。
- **`dataloader.py`**: `DIFFADataset`クラスとカスタムcollator。JSON形式の学習データ読み込み、音声特徴抽出、チャットプロンプト構築、forward diffusion（ランダムマスキング）を担当。
- **`sft_trainer.py`**: `dLLMTrainer`（HuggingFace Trainerを継承）。absorbing state拡散損失 `loss = cross_entropy / t` を実装。
- **`modeling_whisper_encoder.py`**: Whisperエンコーダのラッパー。全隠れ層の状態を取得。
- **`subsample.py`**: 畳み込みベースの音声特徴ダウンサンプリング。
- **`configuration_DIFFA.py`**: モデル設定クラス（prompt_size=64、conv_kernel_sizes="5,5,5"等）。

### 重要な設計パターン
- 音声埋め込みはテキスト埋め込み中の`<audio>`マーカー位置に挿入される
- マスクトークンID: 126336（LLaDAの`[MASK]`トークン）
- チャットテンプレート: `<|startoftext|>`, `<|start_header_id|>`, `<|eot_id|>` を使用
- 分散学習: `torch.distributed.run` + DeepSpeed ZeRO Stage 1（`config/dp_config.json`）
- 推論は自己回帰モデルより遅い（KVキャッシュ非対応）

### データ形式
学習データはJSON形式。各エントリに `audio_id`, `audio_filepath`, `input`, `target`, `transcription`, `dataset`, `duration` を含む。音声は16kHz。最大30秒。

## 重要な依存関係の制約
- `transformers==4.49.0`（固定バージョン）
- `torch>=2.7.0`
- `deepspeed>=0.17.0`
- Python 3.10

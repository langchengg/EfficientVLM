<div align="center">

# 🔬 EfficientVLM: Visual Token Pruning for Lightweight Vision-Language Models

**Adaptive Visual Token Compression for Efficient VLM Inference**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?logo=pytorch)](https://pytorch.org)

</div>

---

## Overview

This project explores **visual token pruning** for lightweight Vision-Language Models (VLMs), built upon the [MiniMind-V](https://github.com/jingyaogong/minimind-v) architecture (26M parameters). We investigate how to reduce the computational cost of visual token processing in VLMs while maintaining image understanding quality, specifically tailored for resource-constrained environments like Apple Silicon (M4).

**Key Idea**: CLIP ViT-B/16 produces 196 visual tokens (14×14 grid) per image. Not all tokens carry equal information — background regions are often redundant. By compressing 196 tokens to a smaller set (e.g., 98 or 49) through attention pooling or score-based pruning, we significantly speed up VLM inference and reduce FLOPs.

---

## 📊 Experimental Results (Apple M4)

We benchmarked the model across different pruning ratios on an Apple M4 chip using the MPS backend. The `attention_pool` strategy yields the best efficiency-performance tradeoff.

### Inference Efficiency

<div align="center">

| Configuration | Visual Tokens | Speed (tok/s) | Speedup | FLOPs Reduction | Image Understanding |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 196 | 9.5 | 1.00x | 0.0% | Excellent |
| **Attention Pool (50%)** | **98** | **14.5** | **1.52x** | **51.5%** | **Strong** |
| Attention Pool (75%) | 147 | 12.4 | 1.30x | 26.1% | Excellent |
| Score Prune (25%) | 49 | 7.1* | 0.75x | 76.1% | Degraded |

</div>

> *Note: At extreme pruning (25%), the overhead of the pruning mechanism itself outweighs the savings in the extremely lightweight 26M MiniMind backbone, causing a drop in actual wall-clock speed despite theoretical FLOPs reduction.*

### Performance Visualization

<div align="center">
<img src="images/dashboard.png" width="90%" alt="Benchmark Dashboard"/>
</div>

### Qualitative Generation Example

**Prompt:** "Describe this image in detail." (Input: A busy city street with cars)

| Model | Response Snippet | Speed |
|-------|------------------|-------|
| Baseline (196 tok) | "A busy city street with multiple cars driving along the road. There are several vehicles parked on the side..." | 36.9 tok/s |
| **Pruned (98 tok)** | "A busy city street with cars driving and parked along the road. Several vehicles and pedestrians are visible..." | **54.4 tok/s** |

*The pruned model maintains spatial awareness and semantic understanding while generating tokens 47% faster.*

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              EfficientVLM Pipeline              │
                    └─────────────────────────────────────────────────┘

  Image ──► CLIP ViT-B/16 ──► 196 tokens ──► Token Pruning ──► K tokens ──► Linear Proj ──► LLM
            (frozen)           (14×14×768)     Module            (K×768)      (K×hidden)     (MiniMind)
                                              ▲
                                              │
                               ┌──────────────┴──────────────┐
                               │                             │
                        Attention Pool              Score-based Prune
                        (Perceiver-style            (Top-K importance
                         cross-attention)            selection)
```

### Two Pruning Strategies

| Strategy | Mechanism | Differentiable | Best For |
|----------|-----------|:-:|----------|
| `attention_pool` | Learnable queries compress tokens via cross-attention (Perceiver-style) | ✅ Fully | Training from scratch |
| `score_prune` | Lightweight scorer ranks token importance, keeps Top-K | ⚠️ Partial | Post-training pruning |

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/langchengg/EfficientVLM.git
cd EfficientVLM
pip install -r requirements.txt
```

> **Apple Silicon (M4/M3/M2)**: The code automatically detects the MPS backend and uses `torch.amp.autocast('mps', float16)`. Requires `transformers==4.45.2` to avoid macOS ARM compatibility issues.

### 2. Prepare Data & Weights

```bash
# LLM base weights
wget -P ./out/ https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch/resolve/main/llm_512.pth
# Vision encoder (CLIP ViT-B/16)
git clone https://huggingface.co/openai/clip-vit-base-patch16 ./model/vision_model/clip-vit-base-patch16
# Datasets
wget -P ./dataset/ https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_i2t.parquet
wget -P ./dataset/ https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_i2t.parquet
```

### 3. Training the Pruned Model

Train a model with 50% visual tokens using Attention Pooling:

```bash
cd trainer

# Stage 1: Pretrain (Vision-Language Alignment ~2 hours on M4)
python train_pretrain_vlm.py --epochs 4 --from_weight llm \
    --keep_ratio 0.5 --pruning_strategy attention_pool \
    --save_weight pretrain_vlm_pruned

# Stage 2: SFT (Instruction Tuning ~4 hours on M4)
python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm_pruned \
    --keep_ratio 0.5 --pruning_strategy attention_pool \
    --save_weight sft_vlm_pruned
```

### 4. Evaluation & Benchmarking

```bash
# Interactive Chat
python eval_vlm.py --weight sft_vlm_pruned --keep_ratio 0.5 --pruning_strategy attention_pool

# Automated Benchmark across all pruning ratios
python benchmark_pruning.py --weight sft_vlm_pruned --ratios 1.0 0.75 0.5 0.25 --strategies attention_pool score_prune
```

---

## Hardware Requirements

| Hardware | Pretraining | SFT | Inference (tok/s) |
|----------|:---:|:---:|:---:|
| **MacBook M4 (16GB)** | ✅ ~2-4h/epoch | ✅ ~3-5h/epoch | **✅ ~10-40** |
| NVIDIA RTX 3090 | ✅ ~1h/epoch | ✅ ~1.5h/epoch | ✅ ~30-60 |
| CPU only | ⚠️ Very slow | ⚠️ Very slow | ✅ ~2-5 |

---

## Related Work & Inspiration

This project aligns with edge-device AI efficiency research, drawing inspiration from:
- **FastVAR** (ICCV 2025) — Cached token pruning for visual autoregressive modeling.
- **SliM-LLM** (ICML 2025) — Mix-precision techniques for efficient LLMs.
- **MiniMind-V** — The base 26M VLM framework.

---

## Citation

```bibtex
@misc{efficientvlm2026,
  title={EfficientVLM: Adaptive Visual Token Pruning for Edge Devices},
  author={Lang Cheng},
  year={2026},
  url={https://github.com/langchengg/EfficientVLM}
}
```

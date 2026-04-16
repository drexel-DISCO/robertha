# Robertha

**Robust Eigenspectrum Regularized Transformer Architecture using Iterative Hopfield Attention**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ACL 2026](https://img.shields.io/badge/ACL-2026-green.svg)](https://2026.aclweb.org/)

Official implementation of "Robertha: Eigenspectrum Regularized Attention for Robust Natural Language Understanding", accepted at ACL 2026.

## Overview

Robertha addresses **asymmetric vulnerability** to embedding corruption in encoder-based language models, where low-magnitude embeddings (critical function words like *not*, *only*, *some* and sentiment words like *excellent*, *terrible*) degrade disproportionately under noise. Our approach combines three mechanisms:

1. **Iterative Hopfield Attention** — Progressive denoising through energy-minimizing attractor dynamics
2. **Eigenspectrum Regularization (ESR)** — Low-rank key structure enforcement creating strong, well-separated attractors with wide recovery basins
3. **Differential Recovery** — Adaptive computational allocation where heavily corrupted embeddings receive more iterations while clean embeddings converge quickly

**Key Results** across 13 GLUE and SuperGLUE tasks:

| | Robertha | Best Baseline |
|---|---|---|
| Degradation at low corruption (σ=0.5) | **0.5 pt** | 14.4 pt |
| Degradation at high corruption (σ=2.0) | **3.9 pt** | 15.6 pt |
| Advantage over baselines (σ=2.0) | **+7.6–9.3 pt** avg | — |
| Task-specific gains (QQP, σ=2.0) | **67.1** | 45.4 |

## Installation

**Requirements:** NVIDIA GPU with CUDA 12.1+ support.

```bash
# Create environment
conda create -n robertha python=3.11 -y
conda activate robertha

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

## Quick Start

**Train on a single task:**
```bash
python robertha.py \
    --regularization esr \
    --task sst2 \
    --train True \
    --epochs 30 \
    --device cuda:0 \
    --seed 42 \
    --noise_type absolute
```

**Evaluate a trained model under corruption:**
```bash
python robertha.py \
    --regularization esr \
    --task sst2 \
    --train False \
    --model_dir ./models \
    --seed 42
```

**Reproduce all results (5 seeds × 22 tasks):**
```bash
bash run_robertha.sh
```

## Supported Tasks

| Benchmark | Tasks |
|-----------|-------|
| GLUE (9) | `cola`, `sst2`, `mrpc`, `qqp`, `stsb`, `mnli`, `qnli`, `rte`, `wnli` |
| SuperGLUE (4) | `boolq`, `copa`, `wic`, `wsc` |
| AdvGLUE (6) | `adv_sst2`, `adv_qqp`, `adv_mnli`, `adv_qnli`, `adv_rte`, `adv_mnli_mismatched` |
| PAWS (3) | `paws`, `pawsx_en`, `pawsx_de` |

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--regularization` | `none` | Regularization type (`esr`, `none`) |
| `--task` | `sst2` | Task name (see table above) |
| `--train` | `False` | `True` to train, `False` to evaluate |
| `--model_size` | `tiny` | Model size (`tiny` for TinyBERT-scale) |
| `--beta` | `15.0` | Inverse temperature for Hopfield retrieval |
| `--max_iterations` | `50` | Maximum Hopfield iterations (T) |
| `--lambda_esr` | `0.05` | ESR regularization weight |
| `--esr_target_entropy_ratio` | `0.35` | Target normalized entropy (α) |
| `--noise_type` | `absolute` | Noise type (`absolute` or `percentage`) |
| `--epochs` | `30` | Training epochs |
| `--seed` | `42` | Random seed |

## Citation

```bibtex
@inproceedings{podasca_acl2026,
  title={Robertha: Eigenspectrum Regularized Attention for Robust Natural Language Understanding},
  author={Podasca, Andreia and Das, Anup},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This material is based upon work supported by the U.S. Department of Energy under Award No. DE-SC0022014 and the National Science Foundation under Grant No. CCF-1942697.

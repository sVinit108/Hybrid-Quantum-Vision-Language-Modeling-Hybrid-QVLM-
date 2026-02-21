# Hybrid-Quantum-Vision-Language-Modeling-Hybrid-QVLM-
manuscript submitted to Knowledge-Based Systems (Elsevier), 2026 

# Hybrid-QVLM: A Quantum-Classical Framework for Enhanced Vision-Language Modeling

[![Paper](https://img.shields.io/badge/Paper-Under_Review-orange)](link_to_paper)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red.svg)](https://pytorch.org/)

> **A hybrid quantum-classical multimodal transformer integrating parameterized quantum circuits for vision-language understanding.**

[Paper (Under Review)](link) | [Preprint](link) | [Project Page](link)

---

## 🔬 Overview

**Hybrid-QVLM** explores the integration of quantum computing into vision-language models by replacing standard linear projections with quantum-enhanced embeddings. Our framework achieves **49.6% parameter sparsity** (288M active from 571M total) while demonstrating competitive performance across multiple benchmarks.

### Key Innovations

- **Quantum CNN (QCNN)**: 12-qubit variational circuit for vision feature extraction with amplitude encoding
- **Position-Aware Quantum Mixing (PAQM)**: 6-qubit circuit encoding positional relationships through quantum entanglement
- **Adaptive Quantum Routing**: Learned residual gate dynamically balances quantum-classical contributions (44%→57% for complex tasks)
- **Sparse Hybrid Architecture**: Integration with Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE)

---

## 📊 Results

| Dataset | Val Accuracy | CIDEr | BLEU-4 | Epochs | Active Params |
|---------|--------------|-------|--------|--------|---------------|
| **VQA v2** | 72.20% | 0.6635 | 0.2157 | 3 | 288M |
| **Food-101** | 91.54% | 0.7665 | 0.4284 | 2 | 288M |
| **DiffusionDB** | 62.52% | 1.8057 | 0.3299 | 5 | 288M |

### Ablation Study Highlights
- **Removing quantum components** → CIDEr/BLEU-4 collapse to near-zero
- **DiffusionDB accuracy gap**: 17.64% drop (62.52% → 44.88%) without quantum embeddings
- **Adaptive routing**: PAQM gate α increases from 0.437 to 0.570 for compositionally complex tasks

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│  Vision Pipeline                 Language Pipeline          │
│  ┌──────────┐                    ┌──────────┐               │
│  │  Input   │                    │  Token   │               │
│  │  Image   │                    │ Embedding│               │
│  └────┬─────┘                    └────┬─────┘               │
│       │                               │                     │
│  ┌────▼─────┐                    ┌────▼─────┐               │
│  │ Patchify │                    │   PAQM   │ (Quantum)     │
│  │ (Conv2d) │                    │ (6-qubit)│               │
│  └────┬─────┘                    └────┬─────┘               │
│       │                               │                     │
│  ┌────▼─────┐                         │                     │
│  │   QCNN   │ (Quantum)               │                     │
│  │(12-qubit)│                         │                     │
│  └────┬─────┘                         │                     │
│       │                               │                     │
│  ┌────▼─────┐                         │                     │
│  │ ViT Stack│◄────────────┬───────────┘                     │
│  │ + MLA    │             │                                 │
│  └────┬─────┘        ┌────▼─────────┐                       │
│       │              │ VLM Decoder  │                       │
│       └──────────────► MLA + mHC    │                       │
│                      │ Sparse MoE   │                       │
│                      └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 48GB+ VRAM recommended (tested on NVIDIA RTX A6000)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Hybrid-QVLM.git
cd Hybrid-QVLM

# Create virtual environment
conda create -n hybrid-qvlm python=3.10
conda activate hybrid-qvlm

# Install dependencies
pip install -r requirements.txt

# Install TorchQuantum for quantum simulation
pip install torchquantum --break-system-packages
```

### Requirements
```
torch==2.4.0
torchvision==0.19.0
torchquantum==0.1.8
transformers==4.40.0
datasets==2.18.0
numpy==1.24.3
pillow==10.3.0
tqdm==4.66.2
pycocoevalcap
```

---

## 📂 Project Structure
```
Hybrid-QVLM/
├── LanguageModel.py          # PAQM + Transformer decoder
├── VisionModel.py             # QCNN + ViT encoder
├── QMM.py                     # Main multimodal architecture
├── Preprocessing.py           # Data preprocessing pipeline
├── Config_file.py             # Model configurations
├── setup.py                   # Training script
├── requirements.txt
├── checkpoints/               # Saved model weights
├── data/                      # Dataset directory
└── README.md
```

---


## 🔧 Configuration

Key hyperparameters in `Config_file.py`:
```python
# Quantum Components
quantum_n_qubits = 6          # PAQM qubits
quantum_n_layers = 2          # VQC depth
n_qubits_vision = 12          # QCNN qubits
n_vqc_layers = 2              # QCNN depth

# Architecture
text_embed_dim = 1024
img_embed_dim = 768
num_heads = 8
n_layers = 8

# MoE Settings
use_moe = True
moe_num_experts = 4
moe_top_k = 1

# Training
learning_rate = 1e-4
batch_size = 12
gradient_accumulation_steps = 1
max_grad_norm = 0.5
```

---

## 📈 Reproducing Results

### 1. Download Datasets
```bash
# DiffusionDB-50k (auto-downloaded via Hugging Face)
# VQA v2 (subset)
# Food-101
```

### 2. Train Full Model
```bash
python setup.py \
    --dataset diffusiondb \
    --epochs 5 \
    --lr 1e-4 \
    --batch_size 12 \
    --use_amp
```

### 3. Ablation Study (Classical Baseline)

Modify `LanguageModel.py` and `VisionModel.py`:
- Replace `PositionAwareQuantumMixing` → `Classical_Positional_Embedding`
- Replace `QuantumVisionEmbedding` → `Classical_CNN_Block`
```bash
python setup.py --dataset diffusiondb --epochs 5
```

---

## 🧪 Evaluation Metrics

We report:
- **Accuracy**: Next-token prediction accuracy
- **Perplexity (PPL)**: exp(cross-entropy loss)
- **CIDEr**: Consensus-based image description evaluation
- **BLEU-4**: 4-gram precision for fluency

---

## 📊 Profiling

Component execution times (mean across datasets):

| Component | Quantum (ms) | Classical (ms) |
|-----------|--------------|----------------|
| Vision Embedding | 98.86 (QCNN) | 0.82 (CNN) |
| Positional Encoding | 34.67 (PAQM) | 0.14 (PE) |
| Vision Attention | 1.10 | 1.14 |
| Language Attention (MLA) | 1.62 | 1.54 |
| Sparse MoE | 24.20 | 24.55 |

**Note**: Quantum simulation overhead will be eliminated on native quantum hardware.

---


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **TorchQuantum** for quantum circuit simulation
- **Hugging Face** for datasets and transformers library
- **DeepSeek** for Multi-head Latent Attention (MLA) architecture
- Conducted at **Veermata Jijabai Technological Institute (VJTI), Mumbai**

---

## 📧 Contact

- **Vinit Sharma** - [vdsharma_m24@ce.vjti.ac.in](mailto:vdsharma_m24@ce.vjti.ac.in)
- **Project Link**: [https://github.com/yourusername/Hybrid-QVLM](https://github.com/yourusername/Hybrid-QVLM)


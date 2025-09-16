# Educational Story Analysis

This project analyzes educational stories to predict intervention outcomes using transformer models optimized for Apple Silicon.

[Hugging Face deployment](https://huggingface.co/polkas/educational-story-outcome-predictor)

## Quick Start

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

1. **Run the model comparison:**

```bash
python modelling.py
```

This will train and compare:

- Fine-tuned DistilBERT
- Fine-tuned RoBERTa
- Most Frequent Class Baseline

## Dataset

Uses the [MU-NLPC/Edustories-en](https://huggingface.co/datasets/MU-NLPC/Edustories-en) dataset for dual-sequence classification:

- **Input**: Situation + Solution
- **Output**: Success/Failure prediction

## Features

- ✅ Apple M3 chip optimization with MPS acceleration
- ✅ Memory-efficient training with gradient checkpointing
- ✅ Dual-sequence transformer architecture
- ✅ Comprehensive evaluation metrics
- ✅ Model comparison and benchmarking

## Results

| Model | Accuracy | F1 Score | Training Time | Notes |
|-------|----------|----------|---------------|-------|
| **Fine-tuned DistilBERT** | **74.18%** | **82.18%** | ~5 minutes | 🥇 Best overall performance |
| **Fine-tuned RoBERTa** | **73.91%** | **80.08%** | ~12 minutes | 🥈 Good but slower training |
| **Most Frequent Baseline** | **61.96%** | **76.51%** | Instant | 🥉 Simple baseline |

### Key Findings:
- **DistilBERT wins**: Slightly better accuracy and much faster training
- **Both models significantly outperform baseline** (~12 percentage point improvement)
- **Training efficiency**: DistilBERT trains 2.4x faster than RoBERTa
- **Model deployment recommendation**: Use DistilBERT for production

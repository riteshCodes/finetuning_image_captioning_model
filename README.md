# Efficient Fine-Tuning for Image Captioning

---

## Table of Contents

1. [Overview](#overview)
2. [Concept and Motivation](#concept-and-motivation)
3. [Repository Structure](#repository-structure)
4. [Model Architecture](#model-architecture)
5. [Attention Masking Design](#attention-masking-design)
6. [LoRA Fine-tuning](#lora-fine-tuning)
7. [Dataset](#dataset)
8. [Training Pipeline](#training-pipeline)
9. [Evaluation](#evaluation)
10. [File-by-File Reference](#file-by-file-reference)
11. [How to Run](#how-to-run)
12. [Expected Results](#expected-results)
13. [References](#references)

---

## Overview

This project implements an **image captioning model** by combining a pretrained vision backbone with a pretrained language model, then fine-tuning the combined system using **LoRA (Low-Rank Adaptation)**. It demonstrates two key ideas:

1. **Multimodal Transformers** — Vision and language models can be combined trivially because they both operate on sequences of embeddings.
2. **Parameter-Efficient Fine-Tuning (PEFT)** — LoRA makes it feasible to fine-tune large pretrained models (300M + 135M parameters) even with limited GPU memory and compute.

The architecture closely follows the design principles of [PaliGemma](https://arxiv.org/abs/2407.07726) and [GIT](https://arxiv.org/abs/2205.14100).

---

## Concept and Motivation

### Why Image Captioning?

Image captioning is the task of generating a natural-language description for a given image. It requires:
- **Visual understanding**: recognizing objects, scenes, and their relationships.
- **Language generation**: producing fluent, grammatically correct sentences.
- **Cross-modal alignment**: grounding language tokens in visual concepts.

### Why Not Train From Scratch?

Training a competitive image captioning model from scratch requires massive datasets and compute. Instead, this project:
1. Reuses a strong **vision encoder** pretrained on image understanding (AIMv2).
2. Reuses a strong **language model** pretrained on text generation (SmolLM2).
3. Connects them with a single **linear projection layer** and fine-tunes only a small fraction of parameters via LoRA.

This approach achieves good performance with roughly 2–4 hours of GPU training.

### Why LoRA?

Full fine-tuning of 435M parameters is memory-intensive. LoRA freezes the original weight matrices and injects small trainable low-rank matrices:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$. This reduces the number of trainable parameters by orders of magnitude while retaining most of the fine-tuning quality.

---

## Repository Structure

```
finetuning_image_captioning_model/
│
├── README.md                         ← This file
├── pyproject.toml                    ← Package metadata and dependencies
│
├── notebooks/
│   └── Finetuning.ipynb              ← End-to-end training and evaluation notebook
│
└── src/
    └── captioning/                   ← Main package
        ├── config.py                 ← ExperimentConfig dataclass + ImageTextProcessor
        ├── flickr_dataset.py         ← Flickr30k dataset, dataloader, collation
        ├── model.py                  ← Model implementation (MaskMixin + ImageCaptioningModel)
        ├── trainer.py                ← Training loop (extends BaseTrainer)
        ├── utils.py                  ← BLEU scoring, loss, visualization
        └── core/
            └── base_trainer.py       ← Abstract BaseTrainer + make_summary_writer
```

---

## Model Architecture

```
Image (3×H×W)
     │
     ▼
┌─────────────────────────────┐
│  AIMv2 Vision Encoder       │  (frozen or LoRA, 300M params)
│  ViT-Large / patch14 / 224  │
└────────────┬────────────────┘
             │ last_hidden_state
             │ (B, N_patches, 1024)
             ▼
┌─────────────────────────────┐
│  Linear Projection Layer    │  (always trained, 1024 → 576)
└────────────┬────────────────┘
             │ image_embeds (B, N_patches, 576)
             │
Text tokens ─┤─ text_embeds (B, T, 576)  ← token embedding from decoder
             │
             ▼
┌─────────────────────────────┐
│  Concatenated Embeddings    │  [image_embeds | text_embeds]
│  (B, N_patches + T, 576)    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  SmolLM2-135M Decoder       │  (frozen or LoRA, 135M params)
│  LLaMA-based causal LM      │
└────────────┬────────────────┘
             │ logits[:, N_patches:, :vocab_size]
             ▼
     Caption logits (B, T, V)
```

### Components

| Component | Model | Parameters | Role |
|-----------|-------|-----------|------|
| Vision Encoder | `apple/aimv2-large-patch14-224` | ~300M | Encodes 224×224 images into 256 patch embeddings of dim 1024 |
| Projection | `nn.Linear(1024, 576)` | ~590K | Aligns image embedding dimension to text embedding dimension |
| Language Decoder | `HuggingFaceTB/SmolLM2-135M` | ~135M | Autoregressively generates text conditioned on image + previous tokens |

The text token embeddings are **shared** with the decoder's input embedding layer (`decoder.get_input_embeddings()`), ensuring the projection target space is consistent with the language model's semantic space.

---

## Attention Masking Design

The core technical challenge is defining a correct **cross-modal attention mask** that allows the decoder to properly attend to both image patches and text tokens simultaneously.

### Mask Structure

For a sequence of `I` image tokens followed by `T` text tokens, the attention mask is a $(I+T) \times (I+T)$ matrix:

$$M = \begin{pmatrix} M_{II} & M_{IT} \\ M_{TI} & M_{TT} \end{pmatrix}$$

| Block | Shape | Value | Meaning |
|-------|-------|-------|---------|
| $M_{II}$ | $I \times I$ | 0 (attend) | Image tokens attend to all other image tokens |
| $M_{IT}$ | $I \times T$ | $-\infty$ (ignore) | Image tokens do **not** attend to text (no future leakage) |
| $M_{TI}$ | $T \times I$ | 0 (attend) | Text tokens attend to all image tokens |
| $M_{TT}$ | $T \times T$ | upper-triangular $-\infty$ | Text tokens attend only to preceding text (causal mask) |

For example, with $I=2$ and $T=3$:

$$M = \begin{pmatrix} 0 & 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty & -\infty \\ 0 & 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}$$

This is combined with a **padding mask** (extending the tokenizer's attention mask to cover image tokens, which are never padded) via element-wise addition. The values $0$ and $-\infty$ are used because HuggingFace LLaMA-based models apply the mask additively to the raw attention scores before softmax.

This masking design is from Figure 3 of the [GIT paper](https://arxiv.org/abs/2205.14100) and is adopted in PaliGemma.

---

## LoRA Fine-tuning

### Configuration

LoRA is applied separately to the decoder and (optionally) the image encoder:

**Decoder (SmolLM2) target modules:**
- `q_proj`, `k_proj` — query and key projection in self-attention
- `gate_proj`, `up_proj`, `down_proj` — feed-forward network (SwiGLU)

**Image Encoder (AIMv2) target modules (optional):**
- `qkv` — combined query/key/value projection
- `proj` — output projection

**Default hyperparameters:**
- Rank $r = 8$
- Alpha $\alpha = 16$ (effective scaling = $\alpha / r = 2$)
- Dropout = 0.1
- Bias: not trained (`bias="none"`)

The `build_model` function in the notebook applies LoRA via the [PEFT library](https://github.com/huggingface/peft). If no LoRA modules are specified for a component, that component is either frozen entirely or fine-tuned in full.

## Dataset

**Flickr30k** — loaded from HuggingFace datasets (`nlphuji/flickr30k`).

The dataset is accessed via its `"test"` HuggingFace partition, which contains all images along with an internal `split` column used to separate train and val:

| Split | Source | Size (approx.) |
|-------|--------|----------------|
| Train | internal `train` + `test` splits | ~29,000 images |
| Validation | internal `val` split | ~1,000 images |

Each image has **5 human-written reference captions**. During training, one caption is chosen at random per image per epoch. During BLEU evaluation, all 5 references are used.

The `FlickrDataset` class wraps the HuggingFace dataset, prepending BOS and appending EOS tokens to each caption. The `BatchCollator` uses the `ImageTextProcessor` to:
1. Apply the AIMv2 image processor (resize, normalize) to images.
2. Tokenize captions and pad them to equal length within the batch.

---

## Training Pipeline

### Data Flow

```
FlickrDataset → BatchCollator → DataLoader
    ↓ (images, captions, attn_mask, indices)
Trainer.training_step()
    ↓ teacher-forcing: input = captions[:, :-1], target = captions[:, 1:]
ImageCaptioningModel.forward()
    ↓ logits (B, T-1, vocab_size)
compute_loss()  ← cross-entropy, ignoring padding positions
    ↓ loss.backward()
AdamW optimizer + LR scheduler
```

### Optimizer and Scheduler

- **Optimizer**: AdamW with default betas
- **LR Schedule**: Linear warmup for 1,000 steps → Cosine annealing over the full training run
- **Gradient accumulation**: 4 micro-steps (effective batch size = 4 × 4 = 16)
- **Mixed precision**: `bfloat16` autocast for forward + backward pass

### Loss Function

Cross-entropy loss over valid (non-padding) positions only:

$$\mathcal{L} = -\frac{1}{|V|} \sum_{t \in V} \log p_\theta(w_t \mid \text{image}, w_{<t})$$

where $V$ is the set of non-padding token positions (including the EOS token).

### TensorBoard Logging

The trainer logs to `runs/<exp_name>_<config_name>_<timestamp>/`:
- `train_loss`, `val_loss` per step
- `learning_rate` per step
- `val_bleu` (if `compute_bleu=True`)
- Caption visualizations (images + predicted + reference captions) logged as figures at every validation checkpoint

---

## Evaluation

### BLEU Score

BLEU (Bilingual Evaluation Understudy) is a standard metric for text generation tasks. It measures n-gram overlap between the model's output and the human reference captions. The implementation uses [sacrebleu](https://github.com/mjpost/sacrebleu) for reproducible corpus-level BLEU.

Computing BLEU requires running `model.generate()` on the full validation set (greedy decoding, max 50 tokens), which takes approximately 5 minutes.

### Greedy Decoding

During generation, the model autoregressively predicts one token at a time:
1. Start with the BOS token.
2. Encode the image once (cached in `image_embeds`).
3. At each step, forward-pass the growing token sequence and take the argmax of the last token's logits.
4. Stop when EOS is produced or `max_length` is reached.
5. Decode the token IDs back to a string with the tokenizer.

---

## File-by-File Reference

### `notebooks/Finetuning.ipynb`

End-to-end notebook covering model construction, LoRA application via PEFT, training, BLEU evaluation, and caption visualization.

### `src/captioning/model.py`

- **`MaskMixin`**: Static methods for building the cross-modal attention mask — `image_text_attention_mask`, `padding_mask`, `convert_to_float`, and `combine_masks`.
- **`ImageCaptioningModel`**: Main model class. Wraps the AIMv2 encoder, projection layer, and SmolLM2 decoder. Exposes `forward()` for teacher-forcing and `generate()` for greedy decoding.

### `src/captioning/config.py`

- **`ExperimentConfig`** (`@dataclass`): All configuration in one place — learning rate, batch size, model checkpoints, LoRA hyperparameters, training flags. Automatically initializes the `ImageTextProcessor` on construction.
- **`ImageTextProcessor`** (extends `ProcessorMixin`): A combined processor that routes images to `AutoImageProcessor` and text to `AutoTokenizer`, returning a unified dictionary compatible with HuggingFace's `return_tensors="pt"`.

### `src/captioning/flickr_dataset.py`

- **`FlickrDataset`**: Maps integer indices into the HuggingFace Flickr30k dataset. Each `__getitem__` returns `(PIL image, randomly sampled caption with BOS/EOS, dataset index)`.
- **`BatchCollator`**: Called by `DataLoader`. Processes a list of `(image, caption, index)` tuples into batched tensors via `ImageTextProcessor`.
- **`create_datasets` / `create_dataloaders`**: Factory functions for train/val splits.
- **`get_pil_image` / `get_ref_captions`**: Utility functions to retrieve PIL images and all 5 reference captions for a batch of indices (used during BLEU evaluation and visualization).
- **`sample_random_batch`**: Samples a small random batch from a DataLoader for visualization.

### `src/captioning/trainer.py`

- **`create_optimizer`**: AdamW.
- **`create_scheduler`**: Chained LinearLR warmup + CosineAnnealingLR.
- **`Trainer(BaseTrainer)`**:
  - `training_step`: Teacher-forcing forward pass with bfloat16 autocast.
  - `validation_step`: Identical to training step (loss evaluation only).
  - `run_experiment`: Iterates `self.fit()` generator, optionally computes BLEU, generates and logs caption visualizations to TensorBoard.

### `src/captioning/utils.py`

- **`visualize_captioning_results`**: Creates a multi-column matplotlib figure showing each image with its predicted caption and all 5 reference captions, with text wrapping.
- **`compute_bleu`**: Runs `model.generate()` over the full validation DataLoader and computes corpus BLEU with sacrebleu.
- **`compute_loss`**: Masked cross-entropy. Reshapes logits and targets to 1D, selects only valid (non-padding) positions, and applies `F.cross_entropy`.

### `src/captioning/core/base_trainer.py`

- **`make_summary_writer`**: Creates a `tensorboardX.SummaryWriter` in `runs/<prefix>_<name>_<timestamp>/`.
- **`BaseTrainer`** (abstract): Reusable training harness.
  - `fit()`: A generator that runs up to `max_steps` training steps. Handles DataLoader exhaustion/restart, gradient accumulation (`(step+1) % grad_accum == 0`), LR scheduler stepping, per-step TensorBoard logging, and periodic validation. Yields `(step, train_metrics, val_metrics)` at each validation checkpoint.
  - `train_step()`: Calls abstract `training_step`, scales loss by gradient accumulation, calls `.backward()`, conditionally steps optimizer.
  - `validate()`: No-grad loop over `val_loader`, averages metrics.

---

## How to Run

### Prerequisites

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -e .
```

### Running the Notebook

Open `notebooks/Finetuning.ipynb` in VS Code or Jupyter:

1. Run all cells in order.
2. The training cell will:
   - Download AIMv2 and SmolLM2 weights (~1.5 GB)
   - Download Flickr30k dataset (~5 GB)
   - Train for 30,000 steps (~2–4 hours on a GPU)
   - Log metrics and visualizations to TensorBoard
3. Monitor training: `tensorboard --logdir runs/`

### Running as a Script

```python
from captioning.config import ExperimentConfig
from captioning.trainer import Trainer
from captioning.model import ImageCaptioningModel
from peft import get_peft_model, LoraConfig

config = ExperimentConfig(
    config_name="aimv2_smollm2_lora",
    encoder_hidden_size=1024,
    decoder_hidden_size=576,
    train_encoder=True,
    image_encoder_lora_modules=["qkv", "proj"],
    train_decoder=True,
    decoder_lora_modules=["q_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
)

model = ImageCaptioningModel(config)
trainer = Trainer(model, config)
trainer.run_experiment()
```

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 6 GB | 12+ GB |
| Disk Space | 8 GB | 12 GB |
| RAM | 16 GB | 32 GB |

---

## Expected Results

| Training Steps | Validation Loss | BLEU Score |
|---------------|----------------|-----------|
| 0 (random) | ~10+ | < 1 |
| 10,000 | ~2.5 | ~15–20 |
| 30,000 | ~2.1 | > 30 |

A BLEU score above 30 after 30k steps indicates the model is generating captions that closely match the human references on Flickr30k.

---

## References

- **PaliGemma**: [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726) — inspiration for the architecture (linear projection from vision to language).
- **GIT**: [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) — source of the cross-modal attention mask design (Figure 3).
- **AIMv2**: [Apple's Scalable Vision Encoder](https://huggingface.co/apple/aimv2-large-patch14-224) — the vision backbone used in this project.
- **SmolLM2**: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — the language model decoder.
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — parameter-efficient fine-tuning method.
- **PEFT**: [HuggingFace PEFT library](https://github.com/huggingface/peft) — LoRA implementation used in this project.
- **sacrebleu**: [A Call for Clarity in Reporting BLEU Scores](https://arxiv.org/abs/1804.08771) — standardized BLEU evaluation.

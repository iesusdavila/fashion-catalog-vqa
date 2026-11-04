# Model — Qwen2-VL-2B + LoRA for fashion VQA

How the model was built and evaluated. To use it, see the [main README](../README.md).

| | |
|---|---|
| Base model | [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), vision tower frozen |
| Fine-tuning | LoRA, 18.5M trainable params (0.83% of 2.23B) |
| Data | [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset), 16k training conversations |
| Hardware | 2× Tesla T4 (Kaggle), 8.5 h |
| Result | **52.3% → 90.0%** exact-match vs zero-shot (3,299 held-out questions) |
| Weights | Kaggle model [`iesusdavila/fashion-vlm`](https://www.kaggle.com/models/iesusdavila/fashion-vlm) |
| Notebook | [`vision-language-model-vlm-fine-tuning-for-vqa.ipynb`](vision-language-model-vlm-fine-tuning-for-vqa.ipynb) |

## Why a VLM

The previous version of this project answered questions with an MLP over **two pooled CLIP vectors**: the image embedding and the question embedding, combined as `[i, q, i*q] → MLP`. That is *late fusion*. The question never looks at the image, only at a single 512-d summary of it. Whether *"Is this item yellow?"* can be answered depends on whether "yellow" survived the pooling, which is why attribute questions, and color in particular, failed.

A VLM uses **early fusion** instead:

- The image is split into 14 px patches, which are merged 2×2 into visual tokens (1 token = 28×28 px, capped at 196 tokens). These tokens are inserted into the LLM sequence.
- Every text token attends to every visual token in every layer, so "yellow" can attend directly to the pixels of the garment.
- Answers are generated as text, so VQA and description share one model and one loss.

## Data

The dataset has 44.4k studio product photos with catalog labels. After lowercasing, the answer vocabulary is:

- 5 genders
- 7 master categories
- 45 subcategories
- 142 article types
- 46 colors
- 4 seasons
- 8 usages

**Leakage-free split.** The data is split with `GroupKFold` (5 folds) on a synthetic group `brand | articleType | baseColour | gender`. No group appears in both train and validation, so validation products are unseen combinations rather than near-duplicates of training items.

| Split | Products |
|---|---:|
| Training folds | 35,529 (16,000 sampled for training) |
| Validation fold | 8,883 (400 for validation loss, 300 for generative evaluation) |

### From labels to conversations

Each image becomes **one multi-turn conversation** with up to 10 Q/A turns, so the visual tokens are encoded once and supervised many times. Conversations are re-sampled every time a sample is loaded:

- **Question bank:** 3–5 paraphrases per attribute.
- **Yes/no questions:** negatives are drawn from the real vocabulary, skipping values that share a word with the true one. For example, a *navy blue* item is never asked "Is it *blue*?" with the target "no".
- **Random turn order and turn count:** the model cannot infer one answer from an earlier one (e.g. gender from article type) instead of from the image.
- **Description turn:** built from visual attributes only. Brand is left out because it is usually not visible and would teach the model to hallucinate.
- **Loss on answer tokens only:** the system prompt, questions and visual tokens are masked with `-100`. `<|im_end|>` is supervised so the model learns to stop.

## Training

| | |
|---|---|
| LoRA | r=16, α=32, dropout 0.05, on q/k/v/o/gate/up/down of the 28 LLM layers |
| Optimizer | AdamW, lr 2e-4, cosine schedule, 3% warmup, grad clip 1.0 |
| Effective batch | 32 (4 per GPU × 2 GPUs × 4 accumulation steps) |
| Precision | fp16 base, fp32 LoRA + `GradScaler`, SDPA attention, gradient checkpointing |
| Distributed | DDP, one process per GPU, via `accelerate launch` |
| Time | 8.5 h (time budget); best checkpoint at epoch 6 |
| Peak memory | 9.1 GB per GPU |
| Adapter size | ~74 MB |

<p align="center"><img src="../assets/training_curves.png" width="900"></p>

Validation loss reaches its minimum (0.0481) at epoch 6 and then climbs. That checkpoint is the released adapter.

### Engineering notes (T4 / Kaggle)

- **`nn.DataParallel` does not work with Qwen2-VL.** `pixel_values` concatenates variable-length patch sequences, so scattering along dim 0 misaligns it with `image_grid_thw`. The training loop is written to a script and launched with `accelerate launch`.
- **The T4 has no bf16 and no FlashAttention-2.** The workaround is fp16 base weights, SDPA attention, and LoRA parameters cast to fp32 so `GradScaler` can unscale them.
- **`NCCL_P2P_DISABLE=1`** avoids the known P2P hang between the two Kaggle T4s.
- **Image cache.** Images are pre-resized once to a 448 px long side using JPEG draft decoding. Without it, the 4 vCPUs would spend every epoch decoding 1800×2400 JPEGs.
- **One `Config` dataclass** is serialized to `config.json` and read by the training script, so the notebook and the DDP processes never disagree on a hyperparameter.

## Evaluation

Zero-shot and fine-tuned use **the same weights, prompt and greedy decoding**. For the baseline, the adapter is simply switched off with `disable_adapter()`.

| Question type | n | Zero-shot | Fine-tuned |
|---|---:|---:|---:|
| masterCategory | 300 | 17.7% | **99.7%** |
| yes/no type | 600 | 96.3% | **98.3%** |
| yes/no color | 600 | 88.0% | **95.0%** |
| subCategory | 300 | 23.7% | **94.3%** |
| gender | 300 | 45.0% | **94.0%** |
| usage | 299 | 32.4% | **92.6%** |
| articleType | 300 | 13.0% | **86.3%** |
| season | 300 | 16.7% | **68.3%** |
| baseColour | 300 | 58.0% | **68.0%** |
| **Overall** | **3,299** | **52.3%** | **90.0%** |

The share of answers that are a valid catalog value goes from 69.3% to **99.9%**.

<p align="center"><img src="../assets/accuracy_by_qtype.png" width="620"></p>

- **Taxonomy alignment explains most of the gain on open questions.** Zero-shot answers `"T-shirt"` or `"Men's Clothing"` where the catalog uses `tshirts` / `topwear`.
- **Visual recognition improves too.** Yes/no color goes from 88.0% to 95.0%, mostly by fixing the base model's bias toward "yes": correct "no" answers rise from 81.7% to 96.7%.
- **Remaining color errors are between neighboring shades** that are ambiguous in the labels themselves: `silver → white/grey`, `tan → brown`, `maroon → red`, `navy blue → blue`.
- **Season is a merchandising label** that is often not visible in the image.

<p align="center"><img src="../assets/yellow_probe.png" width="900"><br><sub>"Is this item yellow?" / "What color is this product?" — zero-shot (base) vs fine-tuned (ft).</sub></p>

## Limitations

- **Domain:** trained on studio packshots (one product, light background) from a single retailer. Lifestyle photos, several items in one image, or non-fashion images reduce accuracy.
- **Color and season** (68%) are much less reliable than category, type, gender and usage (86–100%).
- **Not trained** for brand, material, pattern, size or price.
- **Target gender** is the retailer's market-segment label and can reflect stereotyped associations.
- **Language:** only English questions were evaluated.

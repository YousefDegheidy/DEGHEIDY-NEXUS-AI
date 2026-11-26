# QA74 — T4 / Transformers 5.17 compatibility repair

## Evidence from supplied Colab log
- Python 3.13.16; torch 2.14.1+cu130; CUDA 13.0; Tesla T4 14.56 GiB, free 14.46 GiB.
- Unsloth 2026.11.11, Transformers 5.17.0. All 140 training examples passed the actual native tokenizer and assistant masks; longest 314 tokens. Model weights loaded 723/723. No training updates occurred.
- TrainingArguments constructor failed on removed warmup_ratio. Replaced with warmup_steps=0.05; v5.17 supports fractional ratios. Reference: https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer
- Unsloth logged forced float32 for Qwen3.5 on T4. FP16 autocast disabled as precaution; BF16 only on supported GPU. Actual model loader controls weight storage. xformers fallback was not the crash. Subsequent GPU memory, forward/backward, optimizer, and loss behavior remain unverified.

## Changes
- Actual installed TrainingArguments constructed before loading weights, also in preflight. Explicit labels for Trainer evaluation. Manifest records precision flags and Transformers version.
- App, src, 140 examples and acceptance/release tests unchanged. No new training data or base weights.
- Local CPU tests: 208 passed, 13 subtests passed, versus 204 previously. Four new tests use a strict simulated constructor based on official 5.17 API; real Transformers, Unsloth and CUDA are not installed locally. No GPU training or 8/10 claim.

## Run
Open full notebook with T4, Stage 1, Stage 3 (after data review), Stage 4 for two independent evaluated runs, Stage 5 for UI, Stage 6 for release gate. Stage 2 is optional baseline. Stage 3D downloads logs if needed. Preflight now checks real TrainingArguments before model loading. No need for paid API. Do not replace main until >=8/10 twice and no critical failures. If next failure is OOM, inspect logs rather than silently change model or claim training success.

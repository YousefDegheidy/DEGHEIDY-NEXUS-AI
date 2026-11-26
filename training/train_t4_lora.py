"""DEGHEIDY QA74 supervised: experimental text-only 16-bit LoRA on a single Colab T4.

Uses the EXACT Qwen3.5-4B base family already in DEGHEIDY QA74, and keeps
its Gradio/Whisper/news/memory program unchanged. No OpenAI API or gpt-oss.

Important: importing this module NEVER trains or downloads model weights.
Requires an actual NVIDIA GPU for main(). A completed run is *not* quality approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-4B"
DATASET = "training/data/qa74_supervised.jsonl"
OUTPUT = "artifacts/qa74_t4_lora"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DEGHEIDY T4 16-bit LoRA (experimental / no API)")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--train-jsonl", default=DATASET)
    p.add_argument("--output-dir", default=OUTPUT)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=7e-5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reviewed", action="store_true", help="ONLY after independently reviewing ALL training rows")
    p.add_argument("--preflight-only", action="store_true", help="Check packages, GPU, and tokenize ALL rows before loading weights")
    p.add_argument("--revision", default="", help="Optionally pin model commit for repeatable comparisons")
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_plan(model: str, gpu_gib: float, max_length: int, rank: int,
                  epochs: float, gradient_accumulation: int) -> None:
    if model != BASE_MODEL:
        raise ValueError("Use exact original Qwen3.5-4B model to keep adapter compatible with QA 7.4 inference")
    if gpu_gib < 14.0:
        raise ValueError("This T4 recipe needs >=14 GiB VRAM; choose T4 in Colab")
    if not (128 <= max_length <= 1024):
        raise ValueError("T4 requires max-length between 128 and 1024; try 512 if OOM")
    if rank not in (4, 8, 16):
        raise ValueError("Choose LoRA rank 4, 8, or 16")
    if not (0.1 <= epochs <= 3.0):
        raise ValueError("Refusing excessive epochs on small synthetic dataset")
    if not (1 <= gradient_accumulation <= 64):
        raise ValueError("Invalid accumulation")


def audit_training_data(path: Path) -> list[list[dict[str, str]]]:
    from training.validate_dataset import load_conversations
    from training.audit import audit_file
    report = audit_file(path)
    if not report["approved_for_training"]:
        raise ValueError("Dataset audit FAILED: " + "; ".join(report["issues"][:6]))
    records = load_conversations(path)
    from acceptance_suite import CASES
    from training.audit import audit_rows
    overlap, _ = audit_rows(records, [q for _,q in CASES])
    if overlap:
        raise ValueError('Acceptance leakage or invalid training data: '+ '; '.join(overlap[:6]))
    if len(records) < 50:
        raise ValueError("Need >=50 examples for this experimental run")
    return records


ROLE_START = re.compile(r"<\|im_start\|>(system|user|assistant)\n")
ROLE_END = "<|im_end|>"


def assistant_spans(rendered: str, messages: list[dict]) -> list[tuple[int, int]]:
    """Validate the native Qwen template and identify answer/EOS character spans.

    The pinned Qwen3.5 template has no Jinja generation blocks. We do not
    rewrite it or rely on TRL private helpers. The empty thinking prefill is
    excluded, because inference supplies it when enable_thinking=False.
    """
    if not messages or not any(m["role"] == "assistant" for m in messages):
        raise ValueError("No assistant message")
    for m in messages:
        if any(marker in m["content"] for marker in ("<|im_start|>", ROLE_END, "<think>", "</think>")):
            raise ValueError("Reserved chat marker inside training content")
    starts = list(ROLE_START.finditer(rendered))
    if len(starts) != len(messages):
        raise ValueError("Unexpected Qwen role layout; refusing to train")
    spans = []
    for i, (match, msg) in enumerate(zip(starts, messages)):
        if match.group(1) != msg["role"]:
            raise ValueError("Qwen role order changed")
        end = rendered.find(ROLE_END, match.end())
        boundary = starts[i+1].start() if i+1 < len(starts) else len(rendered)
        if end < match.end() or end + len(ROLE_END) > boundary:
            raise ValueError("Missing Qwen end marker")
        body = rendered[match.end():end]
        expected = msg["content"].strip()
        if msg["role"] == "assistant":
            prefix = "<think>\n\n</think>\n\n"
            if body == expected:
                start = match.end()
            elif body == prefix + expected:
                start = match.end() + len(prefix)
            else:
                raise ValueError("Assistant text changed unexpectedly in template")
            if not expected:
                raise ValueError("Empty assistant answer")
            spans.append((start, end + len(ROLE_END)))
        elif body != expected:
            raise ValueError("User/system text changed unexpectedly in template")
    return spans


def encode_conversation(tokenizer, messages, max_length: int) -> dict:
    """Render with native Qwen format; supervise answer/EOS tokens only."""
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    spans = assistant_spans(rendered, messages)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if not ids or len(ids) != len(offsets):
        raise ValueError("Fast tokenizer returned invalid offsets")
    if len(ids) > max_length:
        raise ValueError(f"Example has {len(ids)} tokens, exceeds {max_length}; increase context or review row")
    labels = [int(t) if end > start and any(start >= a and end <= b for a,b in spans)
              else -100 for t,(start,end) in zip(ids,offsets)]
    for a,b in spans:
        if not any(label != -100 and start >= a and end <= b
                   for label,(start,end) in zip(labels,offsets)):
            raise ValueError("Assistant turn has no supervised tokens")
    return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels}


def assert_assistant_mask(tokenizer) -> None:
    rows = [{"role":"system","content":"SYSTEM_SENTINEL"},
            {"role":"user","content":"PRIVATE_USER_INPUT_DO_NOT_TRAIN"},
            {"role":"assistant","content":"EXPECTED_ASSISTANT_TARGET"}]
    encoded = encode_conversation(tokenizer, rows, 256)
    target = tokenizer.decode([i for i in encoded["labels"] if i != -100])
    if "EXPECTED_ASSISTANT_TARGET" not in target or "PRIVATE_USER_INPUT_DO_NOT_TRAIN" in target or "SYSTEM_SENTINEL" in target:
        raise RuntimeError("Assistant-only mask verification failed")


def collate_assistant_only(batch: list[dict], pad_token_id: int | None):
    import torch
    if pad_token_id is None:
        raise ValueError("Tokenizer must set pad_token_id; use EOS as pad for Qwen")
    max_len = max(len(row["input_ids"]) for row in batch)
    ids, masks, labels = [], [], []
    for row in batch:
        n = max_len - len(row["input_ids"])
        ids.append(list(row["input_ids"]) + [pad_token_id] * n)
        masks.append(list(row["attention_mask"]) + [0] * n)
        labels.append(list(row["labels"]) + [-100] * n)
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)}


def phase(name: str) -> None:
    print(f"\n[QA74 TRAIN] {name}", flush=True)


def build_training_arguments(TrainingArguments, cfg, *, native_bf16: bool):
    """Use v5 warmup_steps and avoid FP16 autocast on Qwen3.5 T4 forced-float32 path."""
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs, learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        fp16=False, bf16=native_bf16, optim="adamw_8bit",
        prediction_loss_only=True, eval_accumulation_steps=1,
        gradient_checkpointing=True,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        warmup_steps=0.05, logging_steps=5, report_to="none",
        remove_unused_columns=False, label_names=["labels"], seed=cfg.seed,
    )


def train(cfg: argparse.Namespace) -> dict:
    if not cfg.reviewed:
        raise RuntimeError("Review all dataset rows before starting; --reviewed is required")
    phase("1/7 Dataset audit")
    train_path = Path(cfg.train_jsonl)
    rows = audit_training_data(train_path)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU here. Run in Google Colab with T4.")
    gpu = torch.cuda.get_device_properties(0)
    gpu_gib = gpu.total_memory / (1024**3)
    validate_plan(cfg.base_model, gpu_gib, cfg.max_length, cfg.lora_rank,
                  cfg.epochs, cfg.gradient_accumulation)
    free,total = torch.cuda.mem_get_info()
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.version.cuda} | GPU {gpu.name} | free {free/1024**3:.2f} GiB / {total/1024**3:.2f} GiB",flush=True)
    if free < 12*1024**3:
        raise RuntimeError("Less than 12 GiB free GPU memory; restart runtime and do not launch Gradio before training")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    phase("2/7 Import training stack")
    # Import Unsloth before Transformers/TRL to enable its memory patches.
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, AutoTokenizer
    from huggingface_hub import HfApi
    for package in ("unsloth","unsloth_zoo","transformers","trl","peft","accelerate","bitsandbytes","datasets"):
        print(f"{package}=={version(package)}", flush=True)
    if int(version("transformers").split('.')[0]) != 5:
        raise RuntimeError("Qwen3.5 requires Transformers v5 for this recipe")
    if not cfg.revision:
        cfg.revision=HfApi().model_info(cfg.base_model).sha
    if not cfg.revision or not re.fullmatch(r"[0-9a-fA-F]{40}", cfg.revision):
        raise RuntimeError("Could not resolve exact model revision")
    revkw = {"revision": cfg.revision}
    phase("3/7 Native tokenizer and assistant-only mask preflight (no model weights yet)")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True, **revkw)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for exact assistant offsets")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert_assistant_mask(tokenizer)
    encoded = []
    for index, row in enumerate(rows, 1):
        try:
            encoded.append(encode_conversation(tokenizer, row, cfg.max_length))
        except Exception as exc:
            raise RuntimeError(f"Training row {index}: {exc}") from exc
    print(f"Encoded {len(encoded)} examples; longest={max(len(x['input_ids']) for x in encoded)} tokens; assistant masks verified", flush=True)
    # Validate actual installed TrainingArguments before expensive model loading.
    native_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
    opts = build_training_arguments(TrainingArguments, cfg, native_bf16=native_bf16)
    print(f"TrainingArguments validated: warmup_steps={opts.warmup_steps}; fp16={opts.fp16}; bf16={opts.bf16}", flush=True)
    if cfg.preflight_only:
        return {"status":"PREFLIGHT_PASSED_NO_TRAINING","rows":len(encoded),"gpu":gpu.name}
    phase("4/7 Download/load pinned model and attach LoRA")
    model, _loaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model, max_seq_length=cfg.max_length,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
        trust_remote_code=False, **revkw,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_rank, lora_alpha=cfg.lora_rank,
        lora_dropout=0, bias="none", random_state=cfg.seed,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", max_seq_length=cfg.max_length,
    )
    # Use the same native pinned tokenizer from preflight, not a patched template.
    model.config.use_cache = False
    phase("5/7 Split data and initialize Trainer")
    raw_rows=[json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    from collections import defaultdict
    groups=defaultdict(list)
    for index,row in enumerate(raw_rows):groups[row.get('domain','other')].append(index)
    training_indices=[];validation_indices=[]
    rng=random.Random(cfg.seed)
    for indices in groups.values():
        rng.shuffle(indices); n=max(1,round(len(indices)*.15)) if len(indices)>1 else 0
        validation_indices.extend(indices[:n]);training_indices.extend(indices[n:])
    if not training_indices or not validation_indices:raise RuntimeError('Empty split')
    split={'train':Dataset.from_list([encoded[i] for i in training_indices]),'test':Dataset.from_list([encoded[i] for i in validation_indices])}
    trainer = Trainer(
        model=model, args=opts, train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=lambda batch: collate_assistant_only(batch, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    phase("6/7 Evaluate initial loss, train, and evaluate final loss")
    before=trainer.evaluate()
    train_result=trainer.train()  # Actual GPU weight updates happen only here.
    after=trainer.evaluate()
    phase("7/7 Save adapter and manifest")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    # Metadata separates completed training from independent quality evaluation.
    manifest = {
        "status": "TRAINED_NOT_QUALITY_CERTIFIED", "base_model": cfg.base_model,
        "base_model_revision": cfg.revision or "NOT_PINNED",
        "method": "Unsloth 16-bit LoRA; assistant-only SFT", "no_openai_api": True,
        "training_rows": len(rows), "train_split_rows":len(training_indices), "validation_split_rows":len(validation_indices),
        "validation_loss_before":before.get("eval_loss"), "validation_loss_after":after.get("eval_loss"),
        "training_metrics":train_result.metrics, "dataset_sha256": sha256_file(train_path),
        "reviewed_by_human": bool(cfg.reviewed),
        "training_dataset_status": "human-reviewed" if cfg.reviewed else "synthetic, automated audit only",
        "lora_rank": cfg.lora_rank, "max_length": cfg.max_length,
        "epochs": cfg.epochs, "learning_rate": cfg.learning_rate,
        "gpu": gpu.name, "gpu_gib": round(gpu_gib, 2),
        "transformers_version": version("transformers"), "warmup_steps": opts.warmup_steps,
        "fp16_autocast": opts.fp16, "bf16_autocast": opts.bf16,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "real_holdout_score": None, "promoted_to_main": False,
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    result = train(args)
    if result.get("status") != "PREFLIGHT_PASSED_NO_TRAINING":
        print("TRAINING_FINISHED; still NOT 8/10 verified", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

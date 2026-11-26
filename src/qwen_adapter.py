"""Explicit adapter validation; never silently switch base model or load missing weights."""
from __future__ import annotations
import json
from pathlib import Path


def validate_adapter_base(adapter: Path, requested_base: str, revision: str = "") -> dict:
    cfg_path = adapter / 'adapter_config.json'
    weights = [adapter / 'adapter_model.safetensors', adapter / 'adapter_model.bin']
    if not cfg_path.is_file() or not any(w.is_file() for w in weights):
        raise FileNotFoundError(f"LoRA adapter was not trained/saved: {adapter}")
    data = json.loads(cfg_path.read_text(encoding='utf-8'))
    training_base = data.get('base_model_name_or_path', '').strip()
    if training_base and training_base != requested_base:
        raise ValueError(f"Different base model! trained={training_base} requested={requested_base}")
    manifest_path = adapter / 'training_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('base_model') != requested_base:
            raise ValueError('Adapter manifest does not match selected Qwen base model')
        trained_revision=manifest.get('base_model_revision','')
        if trained_revision and trained_revision != 'NOT_PINNED' and trained_revision != revision:
            raise ValueError('Set DEGHEIDY_MODEL_REVISION to the exact training commit')
    return data

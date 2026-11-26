from pathlib import Path
from training.train_t4_lora import build_training_arguments, parser

class V517TrainingArguments:
    # Strict simulated constructor, not actual Transformers/GPU.
    def __init__(self, *, output_dir, num_train_epochs, learning_rate,
                 per_device_train_batch_size, per_device_eval_batch_size,
                 gradient_accumulation_steps, fp16, bf16, optim,
                 prediction_loss_only, eval_accumulation_steps,
                 gradient_checkpointing, eval_strategy, save_strategy,
                 save_total_limit, warmup_steps, logging_steps, report_to,
                 remove_unused_columns, label_names, seed):
        self.__dict__.update({k:v for k,v in locals().items() if k != 'self'})

def test_v517_warmup():
    args=build_training_arguments(V517TrainingArguments, parser().parse_args([]), native_bf16=False)
    assert args.warmup_steps == .05 and not hasattr(args, 'warmup_ratio')
    assert args.learning_rate == 7e-5 and args.gradient_accumulation_steps == 8

def test_t4_precision():
    args=build_training_arguments(V517TrainingArguments, parser().parse_args([]), native_bf16=False)
    assert not args.fp16 and not args.bf16
    assert args.gradient_checkpointing and args.per_device_train_batch_size == 1
    assert args.label_names == ['labels']

def test_bf16_hardware():
    args=build_training_arguments(V517TrainingArguments, parser().parse_args([]), native_bf16=True)
    assert args.bf16 and not args.fp16

def test_preflight_before_loading():
    source=(Path(__file__).resolve().parents[1]/'training/train_t4_lora.py').read_text()
    assert source.index('opts = build_training_arguments') < source.index('if cfg.preflight_only:') < source.index('FastLanguageModel.from_pretrained(')
    assert 'warmup_ratio=0.05' not in source

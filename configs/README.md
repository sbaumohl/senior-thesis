# Configuration Guide for Post-Training

This directory contains YAML configuration files for the post-training pipeline.

## Quick Reference

### Available Configs

- **dpo_example.yaml**: Comprehensive DPO configuration with all available options
- **simple_dpo.yaml**: Minimal configuration for quick experiments
- **multi_model_dpo.yaml**: Example for training multiple models sequentially

## Configuration Parameters

### Models

```yaml
models:
  - name: "model-name"              # Friendly name for outputs
    huggingface_id: "org/model"     # Hugging Face model identifier
    trust_remote_code: false        # Allow remote code execution
```

You can specify multiple models to train them sequentially.

### Preference Dataset

```yaml
dataset:
  name: "dataset-name"              # Hugging Face dataset name
  subset: "subset-name"             # Optional dataset subset
  split: "train"                    # Dataset split to use
  prompt_column: "prompt"           # Column with prompts
  chosen_column: "chosen"           # Column with preferred responses
  rejected_column: "rejected"       # Column with rejected responses
  max_samples: 1000                 # Limit samples (null for all)
  validation_split: 0.1             # Validation set proportion
```

**Common datasets:**
- `Anthropic/hh-rlhf`: Helpful and Harmless RLHF data
- `trl-lib/ultrafeedback_binarized`: High-quality preferences
- `lvwerra/stack-exchange-paired`: Programming Q&A

### Post-Training Method

```yaml
post_training:
  method: "dpo"                     # Options: dpo, ipo, kto, orpo
  beta: 0.1                         # Temperature parameter
  label_smoothing: 0.0              # Label smoothing (0-1)
  loss_type: "sigmoid"              # Loss function variant
  use_ref_model: true               # Use separate reference model
  ref_model_id: null                # Reference model (null = use policy model)
```

**Beta parameter guide:**
- 0.05-0.1: Aggressive optimization, faster learning
- 0.1-0.3: Balanced (recommended starting point)
- 0.3-0.5: Conservative, more stable

**Loss types:**
- `sigmoid`: Standard DPO loss (default)
- `hinge`: Hinge loss variant
- `ipo`: Identity Preference Optimization
- `kto_pair`: Kahneman-Tversky Optimization

### LoRA Parameters

```yaml
lora:
  r: 16                             # Rank (lower = fewer parameters)
  lora_alpha: 32                    # Scaling factor (usually 2*r)
  lora_dropout: 0.05                # Dropout probability
  bias: "none"                      # Bias training: none/all/lora_only
  target_modules:                   # Modules to apply LoRA
    - "q_proj"
    - "v_proj"
  task_type: "CAUSAL_LM"            # Task type
  use_rslora: false                 # Use rank-stabilized LoRA
  use_dora: false                   # Use DoRA
```

**Target modules by model family:**
- **GPT-2/GPT-Neo/GPT-J**: `["c_attn"]` or `["q_proj", "v_proj"]`
- **LLaMA/Mistral/Qwen**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Pythia**: `["query_key_value"]`
- **BLOOM**: `["query_key_value"]`
- **Falcon**: `["query_key_value"]`

### Training Hyperparameters

```yaml
training:
  output_dir: "./outputs"           # Output directory
  num_epochs: 1                     # Number of epochs
  per_device_train_batch_size: 2   # Batch size per GPU (train)
  per_device_eval_batch_size: 2    # Batch size per GPU (eval)
  gradient_accumulation_steps: 8   # Gradient accumulation
  learning_rate: 5.0e-7            # Learning rate (lower for post-training!)
  warmup_steps: 100                # Warmup steps
  max_seq_length: 512              # Max total sequence length
  max_prompt_length: 256           # Max prompt length
  max_completion_length: 256       # Max completion length
  logging_steps: 10                # Log every N steps
  save_steps: 500                  # Save checkpoint every N steps
  eval_steps: 500                  # Evaluate every N steps
  save_total_limit: 3              # Max checkpoints to keep
  fp16: false                      # Use mixed precision (FP16)
  bf16: true                       # Use bfloat16 (better for A100)
  gradient_checkpointing: true     # Enable gradient checkpointing
  optim: "adamw_torch"             # Optimizer
  weight_decay: 0.0                # Weight decay (often 0 for DPO)
  max_grad_norm: 1.0               # Gradient clipping
```

**Learning rate guide for post-training:**
- Too high (>1e-5): Model collapse, nonsense outputs
- Recommended: 5e-7 to 1e-6
- Too low (<1e-7): Very slow learning

### GPU Configuration

```yaml
gpu:
  device_ids: [0, 1, 2]            # GPU device IDs to use
  use_ddp: false                   # Use DistributedDataParallel
```

### Fine-tuning Options

```yaml
finetuning:
  resume_from_checkpoint: null     # Path to checkpoint
  load_best_model_at_end: true     # Load best model after training
  metric_for_best_model: "eval_loss"  # Metric to optimize
  greater_is_better: false         # Whether higher is better
  early_stopping_patience: 3       # Early stopping patience
```

### Logging

```yaml
logging:
  report_to: ["tensorboard"]       # Logging backends
  logging_dir: "./logs"            # Log directory
  log_level: "info"                # Log level
```

**Available report_to options:**
- `tensorboard`: TensorBoard logging (recommended)
- `wandb`: Weights & Biases (requires WANDB_API_KEY)
- `mlflow`: MLflow tracking

## Usage Examples

### Train All Models in Config

```bash
uv run src/train.py --config configs/dpo_example.yaml
```

### Train Specific Model

```bash
uv run src/train.py --config configs/dpo_example.yaml --model-index 0
```

### Validate Configuration

```bash
uv run src/train.py --config configs/dpo_example.yaml --dry-run
```

### Custom Output Directory

```bash
uv run src/train.py --config configs/dpo_example.yaml --output-dir ./experiments
```

## Tips for Post-Training

### Memory Management

If you run out of GPU memory:
1. Reduce `per_device_train_batch_size` (e.g., from 4 to 2 or 1)
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_length`, `max_prompt_length`, or `max_completion_length`
4. Enable `gradient_checkpointing: true`
5. Use smaller LoRA rank (e.g., r=8 instead of r=16)
6. Set `use_ref_model: false` (uses policy model as reference)
7. Use `bf16: true` instead of `fp16: false`

### Hyperparameter Tuning

**Start with these defaults:**
- Beta: 0.1
- Learning rate: 5e-7
- LoRA r: 16, alpha: 32
- Batch size: 2-4 per device
- Gradient accumulation: 4-8

**Then adjust:**
- If learning too slowly → increase LR slightly or decrease beta
- If model outputs degrade → decrease LR or increase beta
- If running out of memory → see memory management tips above
- If underfitting → increase LoRA rank or number of epochs

### Effective Batch Size

Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`

Example: 2 × 8 × 1 = 16 effective batch size

Larger effective batch sizes (16-32) generally work better for DPO.

### Model-Specific Tips

**Small models (<1B params):**
- Can use higher learning rates (1e-6)
- May need less regularization (lower beta)

**Medium models (1B-7B params):**
- Standard settings work well
- Consider r=16-32 for LoRA

**Large models (>7B params):**
- Use lower learning rates (5e-7)
- May need larger LoRA rank (r=32-64)
- Definitely use gradient checkpointing

### Dataset Size

- **<1K samples**: May overfit, consider more epochs
- **1K-10K samples**: Good for experimentation
- **10K-100K samples**: Ideal for robust training
- **>100K samples**: Can use subsampling for faster iteration

# Senior Thesis: LLM Post-Training Pipeline

A flexible pipeline for post-training large language models using preference-based methods like DPO (Direct Preference Optimization), IPO, KTO, and ORPO. Supports LoRA for parameter-efficient training and multi-model experimentation.

## Features

- **Multiple Post-Training Methods**: DPO, IPO, KTO, ORPO (PPO coming soon)
- **Multi-Model Training**: Train multiple models sequentially from a single config
- **LoRA Fine-tuning**: Parameter-efficient post-training with configurable LoRA parameters
- **Reference Model Support**: Optional separate reference model for stability
- **Flexible Configuration**: YAML-based configuration for all training parameters
- **Preference Datasets**: Easy integration with Hugging Face preference datasets
- **GPU Management**: Support for single or multi-GPU training
- **Comprehensive Logging**: TensorBoard, Weights & Biases, and MLflow support

## What is Post-Training?

Post-training (also called alignment or RLHF) refines pre-trained language models to better align with human preferences. Unlike pretraining which learns from raw text, post-training uses preference data (chosen vs. rejected responses) to teach models which outputs are preferred.

**Common methods:**
- **DPO**: Direct Preference Optimization - directly optimizes for preferences without RL
- **IPO**: Identity Preference Optimization - variant of DPO with different loss
- **KTO**: Kahneman-Tversky Optimization - uses individual preferences
- **ORPO**: Odds Ratio Preference Optimization - combines SFT and preference learning

## Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync
```

## Quick Start

1. **Configure your post-training** by creating or modifying a YAML config file (see `configs/dpo_example.yaml`):

```yaml
models:
  - name: "pythia-1b-dpo"
    huggingface_id: "EleutherAI/pythia-1b-deduped"

dataset:
  name: "Anthropic/hh-rlhf"
  prompt_column: "prompt"
  chosen_column: "chosen"
  rejected_column: "rejected"

post_training:
  method: "dpo"
  beta: 0.1

lora:
  r: 16
  lora_alpha: 32

training:
  num_epochs: 1
  learning_rate: 5.0e-7
```

2. **Run post-training**:

```bash
# Train all models in config
uv run src/train.py --config configs/dpo_example.yaml

# Train a specific model by index
uv run src/train.py --config configs/dpo_example.yaml --model-index 0

# Validate config without training
uv run src/train.py --config configs/dpo_example.yaml --dry-run
```

## Configuration

### Config Structure

- **models**: List of models to post-train (from Hugging Face)
- **dataset**: Preference dataset configuration (prompt, chosen, rejected columns)
- **post_training**: Method configuration (DPO, IPO, KTO, ORPO)
  - `method`: Which post-training method to use
  - `beta`: Temperature parameter (controls strength of preference optimization)
  - `loss_type`: Loss function variant
  - `use_ref_model`: Whether to use a separate reference model
- **lora**: LoRA parameters for parameter-efficient training
- **training**: Hyperparameters (epochs, batch size, learning rate, etc.)
- **gpu**: GPU configuration (device IDs, DDP settings)
- **finetuning**: Training options (checkpointing, early stopping)
- **logging**: Logging configuration (TensorBoard, W&B, etc.)

### Example Configs

- `configs/dpo_example.yaml`: Comprehensive DPO example with all options
- `configs/simple_dpo.yaml`: Minimal config for quick experiments
- `configs/multi_model_dpo.yaml`: Example training multiple models

## Project Structure

```
senior-thesis/
├── src/
│   ├── train.py          # Main post-training script
│   ├── config.py         # Configuration parser and validators
│   ├── model.py          # Model loading and LoRA setup
│   └── data.py           # Preference dataset loading
├── configs/              # Training configurations
├── utils/                # Utility scripts
└── outputs/              # Training outputs (created automatically)
```

## Usage Examples

### Train with DPO

```bash
uv run src/train.py --config configs/dpo_example.yaml
```

### Train Multiple Models

```bash
uv run src/train.py --config configs/multi_model_dpo.yaml
```

### Train Single Model with Custom Output

```bash
uv run src/train.py --config configs/dpo_example.yaml \
    --model-index 0 \
    --output-dir ./my_experiments
```

### Resume from Checkpoint

```bash
uv run src/train.py --config configs/dpo_example.yaml --resume
```

## Preference Datasets

The pipeline expects datasets with three columns:
- **prompt**: The input prompt
- **chosen**: The preferred/better response
- **rejected**: The rejected/worse response

**Popular datasets:**
- `Anthropic/hh-rlhf`: Helpful and Harmless human feedback data
- `trl-lib/ultrafeedback_binarized`: High-quality preference data
- `lvwerra/stack-exchange-paired`: Programming Q&A preferences

You can specify custom column names in your config if your dataset uses different names.

## Post-Training Methods

### DPO (Direct Preference Optimization)
- **Best for**: General alignment, RLHF without RL complexity
- **Beta**: 0.1-0.5 typical range (lower = more aggressive)
- **Loss**: `sigmoid` (default) or `hinge`

### IPO (Identity Preference Optimization)
- **Best for**: More stable training than DPO
- **Beta**: Similar to DPO
- **Loss**: Set `loss_type: "ipo"`

### KTO (Kahneman-Tversky Optimization)
- **Best for**: Datasets with individual preferences (not pairs)
- **Loss**: Set `loss_type: "kto_pair"`

### ORPO (Odds Ratio Preference Optimization)
- **Best for**: Combined SFT + preference learning
- **Method**: Set `method: "orpo"`

## Tips

1. **Learning Rate**: Use much lower LR than pretraining (5e-7 to 1e-6 typical)

2. **Beta Parameter**:
   - Higher beta (0.3-0.5) = more conservative, closer to reference model
   - Lower beta (0.05-0.1) = more aggressive optimization

3. **Batch Size**: Effective batch size = `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`

4. **Memory Issues**:
   - Reduce batch size and increase gradient accumulation
   - Enable gradient checkpointing
   - Use smaller LoRA rank
   - Disable reference model if needed

5. **LoRA Settings**:
   - Start with r=16, alpha=32
   - Increase rank for more capacity
   - Target attention modules: q_proj, k_proj, v_proj, o_proj

## Development

### Environment Variables

Create a `.env` file for API keys:

```bash
HUGGINGFACE_TOKEN=your_token_here
WANDB_API_KEY=your_key_here
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (strongly recommended)
- Dependencies managed via `uv` (see `pyproject.toml`)

## License

TBD

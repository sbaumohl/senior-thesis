# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM post-training pipeline implementing preference-based optimization methods (DPO, IPO, KTO, ORPO) for aligning language models with human preferences. The pipeline uses LoRA for parameter-efficient training and supports multi-model experimentation through YAML-based configuration.

## Environment and Dependencies

**Package Manager**: `uv` (replaces pip/poetry)
**Python Version**: 3.12+

```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>

# Add dev dependency
uv add --dev <package-name>
```

**Environment Variables**: Create a `.env` file with:
```bash
HUGGINGFACE_TOKEN=your_token_here
WANDB_API_KEY=your_key_here  # Optional, for W&B logging
```

## Running Training

The main entry point is `src/train.py`:

```bash
# Train all models in config
uv run src/train.py --config configs/dpo_example.yaml

# Train specific model by index (0-based)
uv run src/train.py --config configs/dpo_example.yaml --model-index 0

# Validate config without training
uv run src/train.py --config configs/dpo_example.yaml --dry-run

# Override output directory
uv run src/train.py --config configs/dpo_example.yaml --output-dir ./my_experiments

# Resume from checkpoint
uv run src/train.py --config configs/dpo_example.yaml --resume

# Alternative: Direct python invocation (requires activated venv)
python -m src.train --config configs/dpo_example.yaml
```

## Architecture

### Core Components

The pipeline follows a modular architecture with four main components:

1. **Configuration (`src/config.py`)**: Dataclass-based YAML parser with validation
   - `TrainingPipelineConfig`: Top-level config orchestrating all components
   - Nested configs: `ModelConfig`, `DatasetConfig`, `LoRAConfig`, `PostTrainingConfig`, `TrainingConfig`, etc.
   - Validation happens via `config.validate()` before training starts

2. **Model Loading (`src/model.py`)**: Handles model initialization and LoRA setup
   - `load_model_and_tokenizer()`: Loads base model and applies LoRA adapters
   - `load_reference_model()`: Loads frozen reference model for DPO/IPO
   - `setup_device()`: Configures GPU device mapping
   - LoRA is applied using PEFT library with configurable target modules per architecture

3. **Data Pipeline (`src/data.py`)**: Preference dataset loading and formatting
   - `load_preference_dataset()`: Loads from Hugging Face with train/val split
   - `format_dataset_for_training()`: Renames columns to TRL-expected format (prompt/chosen/rejected)
   - Expects three columns: prompt, chosen (preferred response), rejected (worse response)

4. **Training Orchestration (`src/train.py`)**: Main execution logic
   - `train_single_model()`: Router to appropriate training method
   - `train_dpo_single_model()`: DPO/IPO/KTO/ORPO training via TRL's DPOTrainer
   - Supports sequential multi-model training from single config
   - Uses TRL (Transformer Reinforcement Learning) library's DPOTrainer with different loss types

### Training Flow

1. Parse command-line args → Load YAML config → Validate config
2. Set random seed for reproducibility
3. For each model in config:
   - Setup GPU devices
   - Load policy model + tokenizer with LoRA adapters
   - Optionally load separate frozen reference model
   - Load and split preference dataset (train/eval)
   - Format dataset columns for TRL compatibility
   - Initialize DPOConfig with hyperparameters
   - Create DPOTrainer instance
   - Train with `trainer.train()`
   - Save final model + tokenizer
   - Run final evaluation if eval set exists

### Configuration System

All training parameters are defined in YAML configs (see `configs/`). The configuration hierarchy:

```
TrainingPipelineConfig
├── models: List[ModelConfig]           # Can train multiple models sequentially
├── dataset: DatasetConfig              # Preference dataset (prompt/chosen/rejected)
├── post_training: PostTrainingConfig   # Method (dpo/ipo/kto/orpo), beta, loss_type
├── lora: LoRAConfig                    # LoRA rank, alpha, target modules
├── training: TrainingConfig            # Epochs, batch size, LR, sequence lengths
├── gpu: GPUConfig                      # Device IDs, DDP settings
├── finetuning: FineTuningConfig        # Checkpointing, early stopping
└── logging: LoggingConfig              # TensorBoard, W&B, etc.
```

### Post-Training Methods

All methods use TRL's `DPOTrainer` with different `loss_type` settings:

- **DPO** (`loss_type: "sigmoid"`): Direct preference optimization without RL
- **IPO** (`loss_type: "ipo"`): Identity preference optimization variant
- **KTO** (`loss_type: "kto_pair"`): Kahneman-Tversky optimization
- **ORPO** (`method: "orpo"`): Odds ratio preference optimization combining SFT + preferences

The `beta` parameter controls optimization strength (higher = more conservative, closer to reference model).

### LoRA Target Modules by Architecture

Different model families require different target modules in the LoRA config:

- **GPT-2/GPT-Neo/GPT-J**: `["c_attn"]` or `["q_proj", "v_proj"]`
- **LLaMA/Mistral/Qwen**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Pythia**: `["query_key_value"]`
- **BLOOM/Falcon**: `["query_key_value"]`

When working with a new model architecture, check the model's attention layer names to set target modules correctly.

## Configuration Files

- `configs/dpo_example.yaml`: Comprehensive DPO example with all options and comments
- `configs/simple_dpo.yaml`: Minimal config for quick experiments
- `configs/multi_model_dpo.yaml`: Example training multiple models sequentially
- `configs/README.md`: Detailed configuration parameter documentation

## Output Structure

Training outputs are organized as:

```
{output_dir}/
└── {model_name}/
    ├── checkpoint-{step}/    # Intermediate checkpoints
    ├── final/                # Final trained model
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── tokenizer files
    └── ...
```

Logs are saved to `{logging_dir}/{model_name}/`.

## Key Hyperparameters

**Learning Rate**: Much lower than pretraining (5e-7 to 1e-6 typical). Too high causes model collapse.

**Beta Parameter** (post_training.beta):
- 0.05-0.1: Aggressive optimization
- 0.1-0.3: Balanced (recommended start)
- 0.3-0.5: Conservative, more stable

**LoRA Settings**:
- Start with `r=16, alpha=32`
- Increase rank for more capacity
- Typical effective params: <1% of model size

**Effective Batch Size** = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`
- Target 16-32 for DPO

**Memory Management**: If OOM, reduce batch size, increase gradient accumulation, enable gradient checkpointing, reduce LoRA rank, or disable separate reference model (`use_ref_model: false`).

## Preference Datasets

Datasets must have three columns (configurable names):
- `prompt`: Input prompt
- `chosen`: Preferred/better response
- `rejected`: Rejected/worse response

Popular datasets:
- `Anthropic/hh-rlhf`: Helpful and Harmless RLHF
- `trl-lib/ultrafeedback_binarized`: High-quality preferences
- `HuggingFaceH4/orca_dpo_pairs`: Orca-style preferences
- `lvwerra/stack-exchange-paired`: Programming Q&A

## Development Notes

- The pipeline is designed for experimentation with different post-training methods and models
- Multi-model training continues to next model on failure (unless training single model)
- All logging goes through Python's `logging` module with timestamps
- TRL's DPOTrainer handles tokenization internally; datasets are passed as raw text
- Reference model is optional but recommended for stability (uses more memory)
- Gradient checkpointing trades compute for memory (slower but enables larger batch sizes)

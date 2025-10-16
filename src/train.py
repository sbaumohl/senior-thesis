"""Main post-training script with support for DPO, IPO, KTO, and other methods."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import set_seed
from trl import DPOConfig, DPOTrainer

from config import TrainingPipelineConfig
from data import format_dataset_for_training, load_preference_dataset
from model import load_model_and_tokenizer, load_reference_model, setup_device

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-train language models using DPO, IPO, KTO, and other methods"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        default=None,
        help="Train only a specific model by index (0-based). If not specified, trains all models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and validate without training",
    )

    return parser.parse_args()


def train_dpo_single_model(
    config: TrainingPipelineConfig,
    model_index: int,
    override_output_dir: str = None,
    resume: bool = False,
) -> None:
    """Train a single model using DPO (Direct Preference Optimization).

    Args:
        config: Training pipeline configuration
        model_index: Index of the model to train
        override_output_dir: Optional override for output directory
        resume: Whether to resume from checkpoint
    """
    model_config = config.models[model_index]
    logger.info(f"\n{'='*80}")
    logger.info(
        f"Training model {model_index + 1}/{len(config.models)}: {model_config.name}"
    )
    logger.info(f"Method: {config.post_training.method.upper()}")
    logger.info(f"{'='*80}\n")

    # Setup device
    device = setup_device(config.gpu.device_ids, config.gpu.use_ddp)
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logger.info("Loading policy model...")
    model, tokenizer = load_model_and_tokenizer(
        model_config=model_config,
        lora_config=config.lora,
        use_fp16=config.training.fp16,
        use_bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        is_trainable=True,
    )

    # Load reference model if specified
    ref_model = None
    if config.post_training.use_ref_model:
        ref_model_id = config.post_training.ref_model_id or model_config.huggingface_id
        logger.info(f"Loading reference model: {ref_model_id}")
        ref_model = load_reference_model(
            ref_model_id=ref_model_id,
            use_fp16=config.training.fp16,
            use_bf16=config.training.bf16,
            trust_remote_code=model_config.trust_remote_code,
        )

    # Load and prepare dataset
    print(config.dataset.name)
    train_dataset, eval_dataset = load_preference_dataset(
        dataset_config=config.dataset,
        max_samples=config.dataset.max_samples,
        validation_split=config.dataset.validation_split,
    )

    # Format datasets for TRL
    train_dataset = format_dataset_for_training(train_dataset, config.dataset)
    if eval_dataset is not None:
        eval_dataset = format_dataset_for_training(eval_dataset, config.dataset)

    # Determine output directory
    if override_output_dir:
        output_dir = os.path.join(override_output_dir, model_config.name)
    else:
        output_dir = os.path.join(config.training.output_dir, model_config.name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup DPO training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps if eval_dataset is not None else None,
        save_total_limit=config.training.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        load_best_model_at_end=(
            config.finetuning.load_best_model_at_end
            if eval_dataset is not None
            else False
        ),
        metric_for_best_model=(
            config.finetuning.metric_for_best_model
            if eval_dataset is not None
            else None
        ),
        greater_is_better=config.finetuning.greater_is_better,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        optim=config.training.optim,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        gradient_checkpointing=config.training.gradient_checkpointing,
        logging_dir=os.path.join(config.logging.logging_dir, model_config.name),
        report_to=config.logging.report_to,
        seed=config.seed,
        remove_unused_columns=False,
        # DPO-specific parameters
        beta=config.post_training.beta,
        label_smoothing=config.post_training.label_smoothing,
        loss_type=config.post_training.loss_type,
        max_prompt_length=config.training.max_prompt_length,
        max_length=config.training.max_seq_length,
    )

    # Initialize DPO trainer
    logger.info("Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Resume from checkpoint if specified
    checkpoint = None
    if resume and config.finetuning.resume_from_checkpoint:
        checkpoint = config.finetuning.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint}")

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    logger.info("Training complete! Saving model...")
    final_output_dir = os.path.join(output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # Evaluate if eval dataset exists
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        metrics = trainer.evaluate()
        logger.info(f"Final metrics: {metrics}")

    logger.info(f"Model {model_config.name} training complete!\n")


def train_single_model(
    config: TrainingPipelineConfig,
    model_index: int,
    override_output_dir: str = "",
    resume: bool = False,
) -> None:
    """Train a single model using the configured post-training method.

    Args:
        config: Training pipeline configuration
        model_index: Index of the model to train
        override_output_dir: Optional override for output directory
        resume: Whether to resume from checkpoint
    """
    method = config.post_training.method.lower()

    if method in ["dpo", "ipo", "kto", "orpo"]:
        # These methods all use the DPOTrainer with different loss types
        train_dpo_single_model(config, model_index, override_output_dir, resume)
    elif method == "ppo":
        logger.error("PPO training not yet implemented. Use DPO, IPO, KTO, or ORPO.")
        raise NotImplementedError("PPO training support coming soon")
    else:
        logger.error(f"Unknown post-training method: {method}")
        raise ValueError(
            f"Unsupported method '{method}'. " f"Supported methods: dpo, ipo, kto, orpo"
        )


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    try:
        config = TrainingPipelineConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Validate configuration
    logger.info("Validating configuration...")
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)

    # Print configuration
    logger.info(f"\n{config}\n")

    # Dry run mode - just validate config
    if args.dry_run:
        logger.info("Dry run complete. Configuration is valid.")
        return

    # Set seed for reproducibility
    set_seed(config.seed)
    logger.info(f"Set random seed to {config.seed}")

    # Determine which models to train
    if args.model_index is not None:
        if args.model_index < 0 or args.model_index >= len(config.models):
            logger.error(
                f"Invalid model index {args.model_index}. "
                f"Must be between 0 and {len(config.models) - 1}"
            )
            sys.exit(1)
        model_indices = [args.model_index]
        logger.info(f"Training only model at index {args.model_index}")
    else:
        model_indices = range(len(config.models))
        logger.info(f"Training all {len(config.models)} models")

    # Train each model
    for idx in model_indices:
        try:
            train_single_model(
                config=config,
                model_index=idx,
                override_output_dir=args.output_dir,
                resume=args.resume,
            )
        except Exception as e:
            logger.error(
                f"Failed to train model {config.models[idx].name}: {e}", exc_info=True
            )
            if len(model_indices) == 1:
                # If training only one model, exit with error
                sys.exit(1)
            else:
                # If training multiple models, continue with next model
                logger.warning(f"Continuing with next model...")
                continue

    logger.info("\n" + "=" * 80)
    logger.info("All training jobs complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

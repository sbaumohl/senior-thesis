import logging
import os
from pathlib import Path

import wandb
from trl import DPOConfig, DPOTrainer

from config import TrainingPipelineConfig
from data import format_dataset_for_training, load_preference_dataset
from model import load_model_and_tokenizer, load_reference_model, setup_device

logger = logging.getLogger(__name__)


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

    # Initialize WandB run for this model
    wandb_run = None
    if config.wandb and "wandb" in config.logging.report_to:
        run_name = f"{model_config.name}_{config.post_training.method}"
        logger.info(f"Initializing WandB run: {run_name}")
        wandb_run = wandb.init(
            project=config.wandb.project_name,
            name=run_name,
            config={
                "model_name": model_config.name,
                "model_id": model_config.huggingface_id,
                "method": config.post_training.method,
                "beta": config.post_training.beta,
                "loss_type": config.post_training.loss_type,
                "learning_rate": config.training.learning_rate,
                "num_epochs": config.training.num_epochs,
                "batch_size": config.training.per_device_train_batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "dataset": config.dataset.name,
                "seed": config.seed,
            },
            reinit=True,  # Allow multiple runs in same process
        )

    try:
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
            ref_model_id = (
                config.post_training.ref_model_id or model_config.huggingface_id
            )
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

    finally:
        # Always finish WandB run, even if there's an exception
        if wandb_run is not None:
            logger.info("Finishing WandB run...")
            wandb.finish()

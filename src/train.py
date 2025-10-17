"""Main post-training script with support for DPO, IPO, KTO, and other methods."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from transformers import set_seed

from config import TrainingPipelineConfig
from dpo import train_dpo_single_model

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
    elif method == "compo":
        logger.error("ComPO fine-tuning not yet implemented...")
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

    # log in to wandb, if config passes
    wandb.login()

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

"""Dataset loading and preprocessing utilities for preference data."""

import logging
from typing import Optional, Tuple

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from config import DatasetConfig

logger = logging.getLogger(__name__)


def load_preference_dataset(
    dataset_config: DatasetConfig,
    max_samples: Optional[int] = None,
    validation_split: float = 0.1,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and prepare preference dataset for post-training (DPO, etc.).

    Preference datasets contain prompt, chosen, and rejected completions.

    Args:
        dataset_config: Dataset configuration
        max_samples: Maximum number of samples to use
        validation_split: Proportion for validation split

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading preference dataset: {dataset_config.name}")

    # Load dataset from Hugging Face
    if dataset_config.subset:
        dataset = load_dataset(
            dataset_config.name,
            dataset_config.subset,
            split=dataset_config.split,
        )
    else:
        dataset = load_dataset(
            dataset_config.name,
            split=dataset_config.split,
        )

    # Verify required columns exist
    required_columns = [
        dataset_config.prompt_column,
        dataset_config.chosen_column,
        dataset_config.rejected_column,
    ]
    missing_columns = [
        col for col in required_columns if col not in dataset.column_names
    ]
    if missing_columns:
        raise ValueError(
            f"Dataset missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )

    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(
        f"Using prompt='{dataset_config.prompt_column}', "
        f"chosen='{dataset_config.chosen_column}', "
        f"rejected='{dataset_config.rejected_column}'"
    )

    # Limit samples if specified
    max_samples = max_samples or dataset_config.max_samples
    if max_samples is not None:
        logger.info(f"Limiting dataset to {max_samples} samples")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Split into train and validation
    validation_split = validation_split or dataset_config.validation_split
    if validation_split > 0:
        logger.info(f"Splitting dataset with validation ratio: {validation_split}")
        split_dataset = dataset.train_test_split(
            test_size=validation_split,
            seed=42,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def format_dataset_for_training(
    dataset: Dataset,
    dataset_config: DatasetConfig,
) -> Dataset:
    """Format preference dataset columns to standard format for TRL trainers.

    TRL trainers expect specific column names. This function renames columns
    to match TRL's expected format if needed.

    Args:
        dataset: Raw preference dataset
        dataset_config: Dataset configuration with column names

    Returns:
        Dataset with standardized column names
    """
    # TRL trainers expect these column names by default
    standard_columns = {"prompt", "chosen", "rejected"}

    # Create mapping if columns have different names
    column_mapping = {}
    if dataset_config.prompt_column != "prompt":
        column_mapping[dataset_config.prompt_column] = "prompt"
    if dataset_config.chosen_column != "chosen":
        column_mapping[dataset_config.chosen_column] = "chosen"
    if dataset_config.rejected_column != "rejected":
        column_mapping[dataset_config.rejected_column] = "rejected"

    # Rename columns if needed
    if column_mapping:
        logger.info(f"Renaming columns: {column_mapping}")
        dataset = dataset.rename_columns(column_mapping)

    # Keep only the required columns (and any additional columns that might be useful)
    columns_to_keep = ["prompt", "chosen", "rejected"]
    columns_to_remove = [
        col for col in dataset.column_names if col not in columns_to_keep
    ]

    if columns_to_remove:
        logger.info(f"Removing unused columns: {columns_to_remove}")
        dataset = dataset.remove_columns(columns_to_remove)

    return dataset

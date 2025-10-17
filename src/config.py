"""Configuration parser and validator for training pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class WandBConfig:
    project_name: str

@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    huggingface_id: str
    trust_remote_code: bool = False


@dataclass
class DatasetConfig:
    """Dataset configuration for preference/reward datasets."""

    name: str
    split: str = "train"
    subset: Optional[str] = None
    # Preference dataset columns
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    # Optional additional columns
    max_samples: Optional[int] = None
    validation_split: float = 0.1


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) parameters."""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False  # Use rank-stabilized LoRA
    use_dora: bool = False  # Use DoRA (Weight-Decomposed Low-Rank Adaptation)


@dataclass
class PostTrainingConfig:
    """Post-training method configuration (DPO, PPO, etc.)."""

    method: str = "dpo"  # Options: dpo, ipo, kto, orpo, ppo
    # DPO-specific parameters
    beta: float = 0.1  # Temperature parameter for DPO
    label_smoothing: float = 0.0  # Label smoothing for DPO
    loss_type: str = "sigmoid"  # Options: sigmoid, hinge, ipo, kto_pair
    # Reference model settings
    use_ref_model: bool = True  # Whether to use a separate reference model
    ref_model_id: Optional[str] = None  # If None, uses the base model
    # PPO-specific (if used)
    ppo_epochs: int = 4
    kl_penalty: str = "kl"  # Options: kl, abs, mse, full


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    output_dir: str = "./outputs"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7  # Typically lower for post-training
    warmup_steps: int = 100
    max_seq_length: int = 512
    max_prompt_length: int = 256  # For preference learning
    max_completion_length: int = 256  # For preference learning
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class GPUConfig:
    """GPU configuration."""

    device_ids: List[int] = field(default_factory=lambda: [0])
    use_ddp: bool = False


@dataclass
class FineTuningConfig:
    """Fine-tuning specific parameters."""

    resume_from_checkpoint: Optional[str] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""

    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    logging_dir: str = "./logs"
    log_level: str = "info"


@dataclass
class TrainingPipelineConfig:
    """Complete post-training pipeline configuration."""

    models: List[ModelConfig]
    dataset: DatasetConfig
    post_training: PostTrainingConfig
    lora: LoRAConfig
    training: TrainingConfig
    gpu: GPUConfig
    finetuning: FineTuningConfig
    logging: LoggingConfig
    wandb: WandBConfig | None
    seed: int = 42

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingPipelineConfig":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            TrainingPipelineConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        if not config_dict:
            raise ValueError("Empty configuration file")

        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingPipelineConfig":
        """Parse configuration dictionary into dataclass instances."""
        # Parse models
        models = []
        for model_dict in config_dict.get("models", []):
            models.append(ModelConfig(**model_dict))

        if not models:
            raise ValueError("At least one model must be specified")

        # Parse dataset
        dataset = DatasetConfig(**config_dict.get("dataset", {}))

        # Parse post-training config
        post_training = PostTrainingConfig(**config_dict.get("post_training", {}))

        # Parse LoRA config
        lora = LoRAConfig(**config_dict.get("lora", {}))

        # Parse training config
        training = TrainingConfig(**config_dict.get("training", {}))

        # Parse GPU config
        gpu = GPUConfig(**config_dict.get("gpu", {}))

        # Parse fine-tuning config
        finetuning = FineTuningConfig(**config_dict.get("finetuning", {}))

        # Parse logging config
        logging = LoggingConfig(**config_dict.get("logging", {}))

        # Get seed
        seed = config_dict.get("seed", 42)

        # Parse WandB config
        wandb = None
        if "wandb" in config_dict and config_dict["wandb"] is not None:
            wandb = WandBConfig(**config_dict["wandb"])

        return cls(
            models=models,
            dataset=dataset,
            post_training=post_training,
            lora=lora,
            training=training,
            gpu=gpu,
            finetuning=finetuning,
            logging=logging,
            seed=seed,
            wandb=wandb,
        )

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate LoRA parameters
        if self.lora.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.lora.r}")

        if self.lora.lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.lora.lora_alpha}")

        if not 0 <= self.lora.lora_dropout < 1:
            raise ValueError(
                f"LoRA dropout must be in [0, 1), got {self.lora.lora_dropout}"
            )

        # Validate training parameters
        if self.training.num_epochs <= 0:
            raise ValueError(
                f"Number of epochs must be positive, got {self.training.num_epochs}"
            )

        if self.training.learning_rate <= 0:
            raise ValueError(
                f"Learning rate must be positive, got {self.training.learning_rate}"
            )

        if self.training.per_device_train_batch_size <= 0:
            raise ValueError(
                f"Batch size must be positive, got {self.training.per_device_train_batch_size}"
            )

        # Validate dataset parameters
        if self.dataset.validation_split < 0 or self.dataset.validation_split >= 1:
            raise ValueError(
                f"Validation split must be in [0, 1), got {self.dataset.validation_split}"
            )

        # Validate GPU config
        if not self.gpu.device_ids:
            raise ValueError("At least one GPU device ID must be specified")

        for device_id in self.gpu.device_ids:
            if device_id < 0:
                raise ValueError(f"GPU device ID must be non-negative, got {device_id}")

    def __str__(self) -> str:
        """Return a formatted string representation of the configuration."""
        lines = ["Post-Training Pipeline Configuration:"]
        lines.append(f"  Method: {self.post_training.method.upper()}")
        lines.append(f"  Models: {', '.join([m.name for m in self.models])}")
        lines.append(f"  Dataset: {self.dataset.name}")
        lines.append(f"  LoRA rank: {self.lora.r}, alpha: {self.lora.lora_alpha}")
        lines.append(
            f"  Beta: {self.post_training.beta}, Loss: {self.post_training.loss_type}"
        )
        lines.append(
            f"  Epochs: {self.training.num_epochs}, LR: {self.training.learning_rate}"
        )
        lines.append(f"  Batch size: {self.training.per_device_train_batch_size}")
        lines.append(f"  GPUs: {self.gpu.device_ids}")
        lines.append(f"  Output dir: {self.training.output_dir}")
        if self.wandb:
            lines.append(f"  WandB Project Name: {self.wandb.project_name}")
        return "\n".join(lines)

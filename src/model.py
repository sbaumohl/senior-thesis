"""Model loading and LoRA configuration utilities for post-training."""

import logging
import os
from typing import Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer)

from config import LoRAConfig, ModelConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    use_fp16: bool = False,
    use_bf16: bool = False,
    gradient_checkpointing: bool = True,
    is_trainable: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a model and tokenizer from Hugging Face with LoRA configuration.

    Args:
        model_config: Model configuration
        lora_config: LoRA configuration
        use_fp16: Whether to use FP16 precision
        use_bf16: Whether to use BF16 precision
        gradient_checkpointing: Whether to use gradient checkpointing
        is_trainable: Whether this model will be trained (apply LoRA)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_config.huggingface_id}")

    # Determine dtype
    if use_bf16:
        torch_dtype = torch.bfloat16
    elif use_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.huggingface_id,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Set chat template if not present (needed for conversational datasets)
    if tokenizer.chat_template is None:
        # Use a simple default chat template compatible with most models
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'user' %}"
            "{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '\n\n' }}{% endif %}"
            "{% endif %}"
            "{% endfor %}"
        )
        logger.info("Set default chat template for tokenizer")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.huggingface_id,
        trust_remote_code=model_config.trust_remote_code,
        dtype=torch_dtype,
        device_map="auto",
    )

    # Only apply LoRA and gradient checkpointing if this is a trainable model
    if is_trainable:
        # Enable gradient checkpointing if specified
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
            target_modules=lora_config.target_modules,
            use_rslora=lora_config.use_rslora,
            use_dora=lora_config.use_dora,
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
    else:
        # For reference models, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Loaded reference model (frozen)")

    return model, tokenizer


def load_reference_model(
    ref_model_id: str,
    use_fp16: bool = False,
    use_bf16: bool = False,
    trust_remote_code: bool = False,
) -> PreTrainedModel:
    """Load a reference model for post-training methods like DPO.

    Args:
        ref_model_id: Hugging Face model ID for reference model
        use_fp16: Whether to use FP16 precision
        use_bf16: Whether to use BF16 precision
        trust_remote_code: Whether to trust remote code

    Returns:
        Reference model (frozen)
    """
    logger.info(f"Loading reference model: {ref_model_id}")

    # Determine dtype
    if use_bf16:
        torch_dtype = torch.bfloat16
    elif use_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_id,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
        device_map="auto",
    )

    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False

    logger.info("Reference model loaded and frozen")

    return ref_model


def setup_device(device_ids: list[int], use_ddp: bool = False) -> torch.device:
    """Setup device configuration for training.

    Args:
        device_ids: List of GPU device IDs
        use_ddp: Whether to use Distributed Data Parallel

    Returns:
        Primary torch device
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return torch.device("cpu")

    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

    if use_ddp:
        logger.info(f"Using DDP with devices: {device_ids}")
    else:
        logger.info(f"Using device(s): {device_ids}")

    return torch.device(f"cuda:{device_ids[0]}")


def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    model_name: str,
) -> None:
    """Save fine-tuned model and tokenizer.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        output_dir: Output directory
        model_name: Name of the model
    """
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    logger.info(f"Saving model to {save_path}")

    # Save LoRA adapter
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    logger.info(f"Model and tokenizer saved successfully")

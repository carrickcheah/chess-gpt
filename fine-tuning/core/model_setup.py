"""
Model preparation with LoRA adapters for chess fine-tuning.
"""

import logging
from typing import Any, List, Optional, Tuple

# Import Unsloth FIRST before any other ML imports
# Don't do availability check here - just import when needed

from config.config import TrainingJobConfig

logger = logging.getLogger(__name__)


def prepare_model(
    config: TrainingJobConfig
) -> Tuple[Any, Any]:
    """
    Load base model and add LoRA adapters for fine-tuning.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Preparing model: {config.model_name}")

    # Load base model with smart caching
    model, tokenizer = load_pretrained_model(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )

    # Add LoRA adapters
    model = add_lora_adapters(
        model=model,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_bias=config.lora_bias,
        lora_target_modules=config.lora_target_modules,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        seed=config.seed,
        use_rslora=config.use_rslora,
    )

    return model, tokenizer


def load_pretrained_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> Tuple[Any, Any]:
    """
    Load pretrained language model with quantization options.

    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        load_in_4bit: Enable 4-bit quantization
        load_in_8bit: Enable 8-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    # Use standard transformers since Unsloth has dependency conflicts
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    logger.info(f"Loading pretrained model with transformers: {model_name}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    logger.info(f"  Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'None'}")

    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
            bnb_4bit_use_double_quant=True if load_in_4bit else None,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if not (load_in_4bit or load_in_8bit) else "auto",
        device_map="auto",
        trust_remote_code=True,  # Needed for some models
        max_position_embeddings=max_seq_length,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=max_seq_length,
    )

    logger.info("Model loaded successfully with transformers")

    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Model dtype: {next(model.parameters()).dtype}")

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    return model, tokenizer


def add_lora_adapters(
    model: Any,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    lora_target_modules: List[str],
    use_gradient_checkpointing: str,
    seed: int,
    use_rslora: bool,
) -> Any:
    """
    Add LoRA adapters to model for efficient fine-tuning.

    Args:
        model: Base model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout for LoRA layers
        lora_bias: Bias configuration
        lora_target_modules: Modules to apply LoRA to
        use_gradient_checkpointing: Gradient checkpointing strategy
        seed: Random seed
        use_rslora: Use rank-stabilized LoRA

    Returns:
        Model with LoRA adapters
    """
    # Use PEFT directly since Unsloth has conflicts
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    logger.info("Configuring LoRA adapters with PEFT:")
    logger.info(f"  Rank (r): {lora_r}")
    logger.info(f"  Alpha: {lora_alpha}")
    logger.info(f"  Alpha/r ratio: {lora_alpha/lora_r:.2f}")
    logger.info(f"  Dropout: {lora_dropout}")
    logger.info(f"  Target modules: {lora_target_modules}")
    logger.info(f"  Gradient checkpointing: {use_gradient_checkpointing}")
    logger.info(f"  Rank-stabilized: {use_rslora}")

    # Prepare model for k-bit training if quantized
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=use_rslora,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing and use_gradient_checkpointing != "false":
        model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

    # Calculate and log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    logger.info("LoRA adapters added successfully:")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable: {trainable_percent:.2f}%")

    return model


def _add_lora_with_peft(
    model: Any,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    lora_target_modules: List[str],
) -> Any:
    """
    Fallback method to add LoRA using PEFT library.

    Args:
        model: Base model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout probability
        lora_bias: Bias configuration
        lora_target_modules: Target modules for LoRA

    Returns:
        Model with LoRA adapters
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        logger.info("Using PEFT library for LoRA configuration")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        # Log configuration
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"PEFT LoRA applied: {trainable_params:,} trainable parameters")

        return model

    except Exception as e:
        logger.error(f"Failed to apply LoRA with PEFT: {e}")
        raise
"""
Training orchestration with SFTTrainer for chess fine-tuning.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from config.config import TrainingJobConfig

logger = logging.getLogger(__name__)


def prepare_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    config: TrainingJobConfig,
    checkpoint_path: Path,
) -> Any:
    """
    Prepare SFTTrainer for supervised fine-tuning.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Training configuration
        checkpoint_path: Path for saving checkpoints

    Returns:
        Configured SFTTrainer instance
    """
    try:
        from trl import SFTTrainer
    except ImportError as e:
        logger.error("TRL not installed. Please install with: uv add trl")
        raise ImportError("TRL is required for training") from e

    logger.info("Preparing SFTTrainer...")

    # Get training arguments
    training_args = get_training_arguments(config, checkpoint_path)

    # Initialize trainer
    # Note: In TRL 0.22.2, when using PEFT models, the tokenizer is embedded in the model
    # So we need to set it as an attribute
    try:
        # Attach tokenizer to the model for newer TRL versions
        if hasattr(model, "add_adapter"):  # Check if it's a PEFT model
            model.tokenizer = tokenizer

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if not config.skip_eval else None,
            args=training_args,
        )

        # Log training information
        _log_training_info(trainer, train_dataset, eval_dataset, model, config)

        return trainer

    except Exception as e:
        logger.error(f"Failed to initialize SFTTrainer: {e}")
        raise


def get_training_arguments(
    config: TrainingJobConfig,
    output_path: Path
) -> Any:
    """
    Create training arguments for SFTTrainer.

    Args:
        config: Training configuration
        output_path: Directory for outputs

    Returns:
        TrainingArguments instance
    """
    try:
        import torch
        from transformers import TrainingArguments
    except ImportError as e:
        logger.error("Transformers not installed. Please install with: uv add transformers")
        raise ImportError("Transformers is required for training") from e

    logger.info("Creating training arguments...")

    # Determine precision settings
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    use_fp16 = not use_bf16 and torch.cuda.is_available()

    if use_bf16:
        logger.info("Using BF16 precision (preferred)")
    elif use_fp16:
        logger.info("Using FP16 precision (BF16 not available)")
    else:
        logger.info("Using FP32 precision (no GPU acceleration)")

    try:
        args = TrainingArguments(
            # Core training configuration
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            warmup_ratio=config.warmup_ratio,

            # Evaluation and checkpointing
            eval_steps=config.eval_steps if not config.skip_eval else None,
            save_steps=config.save_steps,
            eval_strategy="no" if config.skip_eval else "steps",
            save_strategy="steps",
            do_eval=not config.skip_eval,

            # Optimization settings
            fp16=use_fp16,
            bf16=use_bf16,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,

            # Logging configuration
            logging_steps=config.logging_steps,
            logging_first_step=True,
            output_dir=str(output_path),
            report_to="wandb" if config.wandb_enabled else "none",

            # Other settings
            seed=config.seed,
            load_best_model_at_end=not config.skip_eval,
            metric_for_best_model="eval_loss" if not config.skip_eval else None,
            greater_is_better=False,
            save_total_limit=3,  # Keep only 3 best checkpoints
            push_to_hub=False,  # Will handle separately if needed
            remove_unused_columns=True,
            dataloader_pin_memory=True,
            gradient_checkpointing=config.use_gradient_checkpointing != "false",
        )

        # Log effective batch size
        effective_batch_size = (
            config.batch_size *
            config.gradient_accumulation_steps *
            torch.cuda.device_count()
            if torch.cuda.is_available() else config.batch_size
        )
        logger.info(f"Effective batch size: {effective_batch_size}")

        return args

    except Exception as e:
        logger.error(f"Failed to create training arguments: {e}")
        raise


def _log_training_info(
    trainer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    model: Any,
    config: TrainingJobConfig,
) -> None:
    """
    Log comprehensive training information.

    Args:
        trainer: SFTTrainer instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model: Model being trained
        config: Training configuration
    """
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 50)

    # Dataset info
    logger.info("Dataset:")
    logger.info(f"  Training samples: {len(train_dataset):,}")
    if eval_dataset:
        logger.info(f"  Evaluation samples: {len(eval_dataset):,}")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # Training info
    logger.info("Training:")
    logger.info(f"  Max steps: {config.max_steps:,}")
    logger.info(f"  Learning rate: {config.learning_rate:.2e}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Save frequency: every {config.save_steps} steps")

    # Experiment info
    logger.info("Experiment:")
    logger.info(f"  Name: {config.wandb_experiment_name}")
    logger.info(f"  Tracking: {'Weights & Biases' if config.wandb_enabled else 'Disabled'}")

    # Estimated training time
    steps_per_epoch = len(train_dataset) // config.batch_size
    estimated_epochs = config.max_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    logger.info("Estimates:")
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"  Total epochs: {estimated_epochs:.1f}")

    logger.info("=" * 50)
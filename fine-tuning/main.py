"""
Main entry point for chess fine-tuning on Modal.
Cache update: 2025-09-19-02:47
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import modal
import wandb
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from evaluation.checkpoints import check_for_existing_checkpoint, get_or_create_checkpoint_path
from config.config import TrainingJobConfig
from core.data_processing import prepare_datasets
from core.infra import (
    estimate_training_cost,
    get_docker_image,
    get_gpu_config,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from core.model_setup import prepare_model
from core.trainer import prepare_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize configuration
config = TrainingJobConfig()

# Get HF token from local .env
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("HF_TOKEN not found in .env file")
    sys.exit(1)
logger.info(f"HuggingFace token loaded from .env")

# Get Modal app and infrastructure
modal_app = get_modal_app()
docker_image = get_docker_image(hf_token=HF_TOKEN)

# Modal volumes for caching
pretrained_models_volume = get_volume(config.modal_volume_pretrained_models)
datasets_volume = get_volume(config.modal_volume_datasets)
model_checkpoints_volume = get_volume(config.modal_volume_model_checkpoints)


@modal_app.function(
    image=docker_image,
    gpu=get_gpu_config(),
    volumes={
        "/pretrained_models": pretrained_models_volume,
        "/datasets": datasets_volume,
        "/model_checkpoints": model_checkpoints_volume,
    },
    secrets=get_secrets(),  # Use Modal secrets for API access
    timeout=config.modal_timeout_hours * 60 * 60,
    retries=get_retries(),
    max_inputs=1,  # Fresh container on retry
)
def finetune(config: TrainingJobConfig) -> str:
    """
    Run fine-tuning job on Modal infrastructure.

    Args:
        config: Training configuration

    Returns:
        Experiment name
    """
    logger.info("=" * 60)
    logger.info("CHESS FINE-TUNING SYSTEM")
    logger.info("=" * 60)

    # Log configuration
    config.log_configuration()

    # Initialize Weights & Biases if enabled
    if config.wandb_enabled:
        logger.info(f"Initializing Weights & Biases: {config.wandb_experiment_name}")
        try:
            wandb.init(
                project=config.wandb_project_name,
                name=config.wandb_experiment_name,
                config=config.model_dump(),
                resume="allow",
                entity=None,  # Use default entity from wandb login
            )
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            logger.info("Continuing without W&B tracking")
            config.wandb_enabled = False

    # Prepare model and tokenizer
    logger.info(f"Loading model: {config.model_name}")
    try:
        model, tokenizer = prepare_model(config)
    except Exception as e:
        logger.error(f"Model preparation failed: {e}")
        raise

    # Load and process datasets
    logger.info(f"Loading dataset: {config.dataset_name}")
    try:
        train_dataset, eval_dataset = prepare_datasets(
            config, datasets_volume, tokenizer
        )
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise

    # Set up checkpoint directory
    checkpoint_path = get_or_create_checkpoint_path(config.wandb_experiment_name)
    logger.info(f"Checkpoint directory: {checkpoint_path}")

    # Check for existing checkpoints
    resume_from = check_for_existing_checkpoint(checkpoint_path)
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
    else:
        logger.info("Starting training from scratch")

    # Prepare trainer
    logger.info("Initializing trainer...")
    try:
        trainer = prepare_trainer(
            model, tokenizer, train_dataset, eval_dataset, config, checkpoint_path
        )
    except Exception as e:
        logger.error(f"Trainer initialization failed: {e}")
        raise

    # Start training
    logger.info("Starting training...")
    try:
        if resume_from:
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        _save_emergency_checkpoint(trainer, checkpoint_path)
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        _save_emergency_checkpoint(trainer, checkpoint_path)
        raise

    # Save final model
    logger.info("Saving final model...")
    final_model_path = checkpoint_path / "final_model"
    try:
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    # Cleanup
    if config.wandb_enabled:
        wandb.finish()

    logger.info("=" * 60)
    logger.info(f"Training completed: {config.wandb_experiment_name}")
    logger.info("=" * 60)

    return config.wandb_experiment_name


def _save_emergency_checkpoint(trainer: Any, checkpoint_path: Path) -> None:
    """
    Save emergency checkpoint on failure.

    Args:
        trainer: Trainer instance
        checkpoint_path: Path to save checkpoint
    """
    try:
        emergency_path = checkpoint_path / "emergency_checkpoint"
        logger.info(f"Saving emergency checkpoint to: {emergency_path}")
        trainer.save_model(emergency_path)
        logger.info("Emergency checkpoint saved successfully")
    except Exception as e:
        logger.error(f"Failed to save emergency checkpoint: {e}")


@modal_app.local_entrypoint()
def main(
    model_name: str = "unsloth/LFM2-350M",
    learning_rate: float = 2e-4,
    max_steps: int = 10000,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.05,
    experiment_name: Optional[str] = None,
    invalidate_dataset_cache: bool = False,
    estimate_cost: bool = True,
) -> None:
    """
    Local entry point for Modal deployment.

    Args:
        model_name: HuggingFace model identifier
        learning_rate: Peak learning rate
        max_steps: Maximum training steps
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_ratio: Warmup ratio
        experiment_name: Optional experiment name
        invalidate_dataset_cache: Force dataset regeneration
        estimate_cost: Show cost estimate before starting
    """
    logger.info("Preparing chess fine-tuning job...")

    # Create configuration
    config = TrainingJobConfig(
        model_name=model_name,
        learning_rate=learning_rate,
        max_steps=max_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        wandb_experiment_name=experiment_name,
        invalidate_dataset_cache=invalidate_dataset_cache,
    )

    # Show configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Dataset: {config.dataset_name}")
    logger.info(f"  LoRA: rank={config.lora_r}, alpha={config.lora_alpha}")
    logger.info(f"  Training: lr={config.learning_rate:.2e}, steps={config.max_steps}")
    logger.info(f"  Batch: size={config.batch_size}, grad_accum={config.gradient_accumulation_steps}")

    # Estimate costs
    if estimate_cost:
        cost_estimate = estimate_training_cost(config)
        logger.info(f"Estimated cost: ${cost_estimate['total_cost_estimate']}")
        # Skip interactive confirmation for Modal deployment

    # Launch training on Modal
    logger.info(f"Launching training job: {config.wandb_experiment_name}")
    try:
        result = finetune.remote(config)
        logger.info(f"Training completed successfully: {result}")
    except Exception as e:
        logger.error(f"Training job failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
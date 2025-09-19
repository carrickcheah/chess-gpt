"""
Modal infrastructure setup for cloud GPU training.
"""

import logging
from typing import Any, Dict, List, Optional

import modal
import modal.exception

from config.config import TrainingJobConfig

logger = logging.getLogger(__name__)


def get_modal_app() -> modal.App:
    """
    Get or create Modal application instance.

    Returns:
        Modal App instance
    """
    # Create config instance when needed
    try:
        config = TrainingJobConfig()
        app_name = config.modal_app_name
    except Exception:
        app_name = "finetune-chess-llm"  # Default fallback

    logger.info(f"Initializing Modal app: {app_name}")
    return modal.App(app_name)


def get_docker_image(hf_token: Optional[str] = None) -> modal.Image:
    """
    Create Docker image with all required dependencies.

    Args:
        hf_token: Optional HuggingFace token to include

    Returns:
        Modal Image with ML dependencies installed
    """
    logger.info("Building Docker image with dependencies...")

    docker_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            # Core ML libraries
            "torch==2.5.1",
            "transformers==4.55.2",
            "datasets>=3.0.0",
            "accelerate>=0.27.0",
            "peft>=0.12.0",
            "trl>=0.8.0",
            "bitsandbytes>=0.41.0",  # For quantization
            "hf-transfer>=0.1.5",
            "huggingface_hub>=0.34.0",

            # Monitoring and utilities
            "wandb==0.19.1",
            "pydantic-settings==2.7.0",
            "tqdm==4.67.1",
            "jinja2==3.1.5",

            # Chess specific
            "chess==1.11.0",
            "python-chess==1.999",
        )
        .env(
            {
                "HF_HOME": "/model_cache",
                "TRANSFORMERS_CACHE": "/model_cache",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "TOKENIZERS_PARALLELISM": "false",
                **({"HF_TOKEN": hf_token} if hf_token else {}),
            }
        )
        .add_local_dir(".", remote_path="/root")
    )


    return docker_image


def get_docker_image_for_evaluation() -> modal.Image:
    """
    Create lighter Docker image for evaluation tasks.

    Returns:
        Modal Image for evaluation
    """
    logger.info("Building evaluation Docker image...")

    docker_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "datasets>=3.0.0",
            "hf-transfer>=0.1.5",
            "huggingface_hub>=0.34.0",
            "peft>=0.12.0",
            "transformers==4.55.2",
            "torch==2.5.1",
            "wandb==0.19.1",
            "pydantic-settings==2.7.0",
            "chess==1.11.0",
            "python-chess==1.999",
        )
        .env({"HF_HOME": "/model_cache"})
        .add_local_dir(".", remote_path="/root")
    )

    return docker_image


def get_volume(volume_name: str) -> modal.Volume:
    """
    Get or create a Modal volume for persistent storage.

    Args:
        volume_name: Name of the volume

    Returns:
        Modal Volume instance
    """
    logger.info(f"Accessing Modal volume: {volume_name}")

    try:
        volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        logger.info(f"Volume '{volume_name}' ready")
        return volume
    except Exception as e:
        logger.error(f"Failed to access volume '{volume_name}': {e}")
        raise


def get_secrets() -> List[modal.Secret]:
    """
    Get Modal secrets for API access.

    Returns:
        List of Modal Secret instances
    """
    secrets = []

    # Load HuggingFace secret
    try:
        hf_secret = modal.Secret.from_name("huggingface-secret")
        secrets.append(hf_secret)
        logger.info("HuggingFace secret loaded")
    except Exception as e:
        logger.info(f"HuggingFace secret not found: {e} - using HF_TOKEN from environment")

    # Load Weights & Biases secret (now available)
    try:
        wandb_secret = modal.Secret.from_name("wandb-secret")
        secrets.append(wandb_secret)
        logger.info("Weights & Biases secret loaded")
    except Exception as e:
        logger.info(f"W&B secret not found: {e} - W&B tracking disabled")

    if not secrets:
        logger.warning(
            "No secrets found. Please configure secrets in Modal dashboard:\n"
            "  - wandb-secret: {'WANDB_API_KEY': 'your-key'}\n"
            "  - huggingface-secret: {'HF_TOKEN': 'your-token'}"
        )

    return secrets


def get_retries() -> modal.Retries:
    """
    Get retry configuration for fault tolerance.

    Returns:
        Modal Retries configuration
    """
    try:
        config = TrainingJobConfig()
        max_retries = config.modal_max_retries
    except Exception:
        max_retries = 3  # Default fallback

    retries = modal.Retries(
        max_retries=max_retries,
        backoff_coefficient=2.0,
        initial_delay=10.0,  # 10 seconds initial delay
        max_delay=60.0,      # Max 60 seconds between retries (Modal limit)
    )

    logger.info(f"Retry policy: max {max_retries} attempts")
    return retries


def get_gpu_config() -> str:
    """
    Get GPU configuration string for Modal.

    Returns:
        GPU type string
    """
    try:
        config = TrainingJobConfig()
        gpu_type = config.modal_gpu_type
    except Exception:
        gpu_type = "L40S"  # Default fallback

    # Validate GPU availability
    valid_gpus = ["T4", "L4", "A10G", "L40S", "A100", "H100"]
    if gpu_type not in valid_gpus:
        logger.warning(f"Unknown GPU type: {gpu_type}, defaulting to L40S")
        gpu_type = "L40S"

    # Log GPU specifications
    gpu_specs = {
        "T4": "16GB VRAM, good for small models",
        "L4": "24GB VRAM, efficient inference",
        "A10G": "24GB VRAM, good price/performance",
        "L40S": "48GB VRAM, excellent for training",
        "A100": "40-80GB VRAM, premium performance",
        "H100": "80GB VRAM, top tier performance",
    }

    logger.info(f"GPU configuration: {gpu_type} ({gpu_specs.get(gpu_type, 'Unknown')})")
    return gpu_type


def estimate_training_cost(config: TrainingJobConfig) -> Dict[str, float]:
    """
    Estimate training cost based on configuration.

    Args:
        config: Training configuration

    Returns:
        Dictionary with cost estimates
    """
    # Approximate GPU costs per hour (USD)
    gpu_costs = {
        "T4": 0.59,
        "L4": 1.25,
        "A10G": 1.50,
        "L40S": 2.95,
        "A100": 3.85,
        "H100": 8.50,
    }

    gpu_cost_per_hour = gpu_costs.get(config.modal_gpu_type, 3.00)

    # Estimate training time
    # Rough estimate: 100 steps per minute for small models
    estimated_minutes = config.max_steps / 100
    estimated_hours = estimated_minutes / 60

    # Calculate costs
    training_cost = estimated_hours * gpu_cost_per_hour
    total_cost = training_cost * 1.2  # Add 20% buffer for retries/overhead

    estimates = {
        "gpu_type": config.modal_gpu_type,
        "gpu_cost_per_hour": gpu_cost_per_hour,
        "estimated_hours": round(estimated_hours, 2),
        "training_cost": round(training_cost, 2),
        "total_cost_estimate": round(total_cost, 2),
        "max_cost": round(config.modal_timeout_hours * gpu_cost_per_hour, 2),
    }

    logger.info("Training cost estimate:")
    logger.info(f"  GPU: {estimates['gpu_type']} (${estimates['gpu_cost_per_hour']}/hour)")
    logger.info(f"  Estimated time: {estimates['estimated_hours']} hours")
    logger.info(f"  Estimated cost: ${estimates['total_cost_estimate']}")
    logger.info(f"  Maximum cost: ${estimates['max_cost']} (timeout limit)")

    return estimates
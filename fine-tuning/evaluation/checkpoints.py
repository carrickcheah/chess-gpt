"""
Checkpoint management for training resumption and fault tolerance.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_or_create_checkpoint_path(experiment_name: str) -> Path:
    """
    Get or create checkpoint directory for an experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Path to checkpoint directory
    """
    checkpoint_base = Path("/model_checkpoints")
    checkpoint_path = checkpoint_base / experiment_name

    try:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory ready: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to create checkpoint directory: {e}")
        # Fallback to local directory
        checkpoint_path = Path("checkpoints") / experiment_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using local checkpoint directory: {checkpoint_path}")

    return checkpoint_path


def check_for_existing_checkpoint(checkpoint_path: Path) -> Optional[str]:
    """
    Check for existing checkpoints to resume training.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Path to latest checkpoint or None if no checkpoints exist
    """
    if not checkpoint_path.exists():
        return None

    # Look for checkpoint directories (format: checkpoint-XXXX)
    checkpoint_dirs = sorted(
        [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]) if len(x.name.split("-")) > 1 else 0
    )

    if not checkpoint_dirs:
        logger.info("No existing checkpoints found")
        return None

    latest_checkpoint = checkpoint_dirs[-1]
    step_num = latest_checkpoint.name.split("-")[1]
    logger.info(f"Found {len(checkpoint_dirs)} checkpoints")
    logger.info(f"Latest checkpoint: {latest_checkpoint.name} (step {step_num})")

    # Verify checkpoint is valid
    required_files = ["config.json", "adapter_model.safetensors"]
    for file_name in required_files:
        if not (latest_checkpoint / file_name).exists():
            logger.warning(f"Checkpoint missing required file: {file_name}")
            logger.info("Will start training from scratch")
            return None

    return str(latest_checkpoint)


def cleanup_old_checkpoints(
    checkpoint_path: Path,
    keep_last_n: int = 3,
    keep_best: bool = True
) -> None:
    """
    Clean up old checkpoints to save space.

    Args:
        checkpoint_path: Path to checkpoint directory
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    """
    if not checkpoint_path.exists():
        return

    checkpoint_dirs = sorted(
        [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]) if len(x.name.split("-")) > 1 else 0
    )

    if len(checkpoint_dirs) <= keep_last_n:
        return

    # Keep the best checkpoint if it exists
    best_checkpoint = checkpoint_path / "best_checkpoint"
    checkpoints_to_keep = set()

    if keep_best and best_checkpoint.exists():
        checkpoints_to_keep.add(best_checkpoint)

    # Keep the last N checkpoints
    checkpoints_to_keep.update(checkpoint_dirs[-keep_last_n:])

    # Remove old checkpoints
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir not in checkpoints_to_keep:
            try:
                import shutil
                shutil.rmtree(checkpoint_dir)
                logger.info(f"Removed old checkpoint: {checkpoint_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_dir.name}: {e}")


def save_training_state(
    checkpoint_path: Path,
    step: int,
    metrics: dict,
    config: dict
) -> None:
    """
    Save additional training state information.

    Args:
        checkpoint_path: Path to checkpoint directory
        step: Current training step
        metrics: Training metrics
        config: Training configuration
    """
    import json

    state_file = checkpoint_path / "training_state.json"

    state = {
        "step": step,
        "metrics": metrics,
        "config": config,
        "timestamp": str(Path.ctime(checkpoint_path)),
    }

    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug(f"Saved training state at step {step}")
    except Exception as e:
        logger.warning(f"Failed to save training state: {e}")


def load_training_state(checkpoint_path: Path) -> Optional[dict]:
    """
    Load training state information.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Training state dictionary or None
    """
    import json

    state_file = checkpoint_path / "training_state.json"

    if not state_file.exists():
        return None

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
        logger.info(f"Loaded training state from step {state.get('step', 'unknown')}")
        return state
    except Exception as e:
        logger.warning(f"Failed to load training state: {e}")
        return None
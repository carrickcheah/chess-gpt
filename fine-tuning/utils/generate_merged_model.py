#!/usr/bin/env python3
"""
Download LoRA adapters from Modal volume and merge with base model.
"""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_model_checkpoint_from_modal_volume(
    modal_volume_name: str,
    remote_path_to_checkpoint: str,
    local_path: Optional[str] = None,
) -> str:
    """
    Download LoRA adapters from Modal volume.

    Args:
        modal_volume_name: Name of Modal volume
        remote_path_to_checkpoint: Remote path in volume
        local_path: Optional local path for download

    Returns:
        Local path to downloaded checkpoint
    """
    import subprocess

    logger.info(f"Downloading checkpoint from Modal volume: {modal_volume_name}")
    logger.info(f"Remote path: {remote_path_to_checkpoint}")

    # Determine local path
    if local_path is None:
        local_path = f"./{remote_path_to_checkpoint.split('/')[-1]}"

    try:
        # Download using Modal CLI
        cmd = f"modal volume get {modal_volume_name} {remote_path_to_checkpoint} --force"
        logger.info(f"Running: {cmd}")

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            raise RuntimeError(f"Modal download failed: {result.stderr}")

        logger.info(f"Downloaded to: {local_path}")
        return local_path

    except Exception as e:
        logger.error(f"Error downloading checkpoint: {e}")
        raise


def merge_lora_adapter_to_base_model(
    adapter_path: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_repo_name: Optional[str] = None,
) -> None:
    """
    Merge LoRA adapter with base model.

    Args:
        adapter_path: Path to LoRA adapter checkpoint
        output_dir: Directory to save merged model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_name: HuggingFace repository name
    """
    logger.info("Starting LoRA adapter merge process...")
    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        logger.error(f"Adapter path not found: {adapter_path}")
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    # Step 1: Load adapter configuration
    logger.info("Loading adapter configuration...")
    try:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = peft_config.base_model_name_or_path
        logger.info(f"Base model: {base_model_name}")
    except Exception as e:
        logger.error(f"Failed to load adapter config: {e}")
        raise

    # Step 2: Load base model
    logger.info("Loading base model...")
    try:
        # Determine dtype
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        logger.info(f"Base model loaded with dtype: {dtype}")

    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise

    # Step 3: Load tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # Step 4: Load and merge LoRA adapter
    logger.info("Loading LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("Adapter loaded successfully")

        # Get adapter statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")

    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise

    # Step 5: Merge adapter with base model
    logger.info("Merging adapter with base model...")
    try:
        merged_model = model.merge_and_unload()
        logger.info("Model merged successfully")

    except Exception as e:
        logger.error(f"Failed to merge model: {e}")
        raise

    # Step 6: Save merged model locally
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to: {output_path}")
    try:
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,  # Use safetensors format
        )
        tokenizer.save_pretrained(output_path)
        logger.info("Model saved successfully")

        # Calculate model size
        model_size_gb = sum(
            os.path.getsize(output_path / f) for f in output_path.glob("*")
        ) / (1024 ** 3)
        logger.info(f"Model size: {model_size_gb:.2f} GB")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

    # Step 7: Optionally push to HuggingFace Hub
    if push_to_hub and hub_repo_name:
        logger.info(f"Pushing model to HuggingFace Hub: {hub_repo_name}")
        try:
            merged_model.push_to_hub(hub_repo_name, safe_serialization=True)
            tokenizer.push_to_hub(hub_repo_name)
            logger.info("Model pushed to hub successfully")

        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            logger.info("Model saved locally but not pushed to hub")


def run(
    modal_volume_name: str = "model_checkpoints",
    remote_checkpoint_dir: str = None,
    local_merged_models_dir: str = "models/merged",
    push_to_hub: bool = False,
    hub_repo_name: Optional[str] = None,
    cleanup: bool = True,
) -> None:
    """
    Main function to download and merge LoRA adapters.

    Args:
        modal_volume_name: Name of Modal volume
        remote_checkpoint_dir: Path to checkpoint in volume
        local_merged_models_dir: Local directory for merged model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_name: HuggingFace repository name
        cleanup: Whether to cleanup downloaded adapters
    """
    if not remote_checkpoint_dir:
        logger.error("Remote checkpoint directory required")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("LORA ADAPTER MERGE TOOL")
    logger.info("=" * 60)

    # Download adapters from Modal
    try:
        local_adapter_path = download_model_checkpoint_from_modal_volume(
            modal_volume_name=modal_volume_name,
            remote_path_to_checkpoint=remote_checkpoint_dir,
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    # Merge adapters with base model
    output_dir = Path(local_merged_models_dir) / Path(remote_checkpoint_dir).name
    try:
        merge_lora_adapter_to_base_model(
            adapter_path=local_adapter_path,
            output_dir=str(output_dir),
            push_to_hub=push_to_hub,
            hub_repo_name=hub_repo_name,
        )
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        sys.exit(1)

    # Cleanup downloaded adapters
    if cleanup and os.path.exists(local_adapter_path):
        logger.info("Cleaning up downloaded adapters...")
        try:
            shutil.rmtree(local_adapter_path)
            logger.info("Cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info(f"Merged model saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import fire
    fire.Fire(run)
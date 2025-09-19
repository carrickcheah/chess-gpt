#!/usr/bin/env python
"""
Download trained model from Modal volumes to local machine.
Uses the simple modal volume get command like the original chess-game project.
"""

import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Modal profile
os.environ["MODAL_PROFILE"] = "carrick113"


def download_model_from_modal():
    """Download the trained model from Modal volumes using modal CLI."""

    # Your trained model info
    EXPERIMENT_NAME = "LFM2-350M-r16-a32-20250919-025808"
    VOLUME_NAME = "model_checkpoints"

    # Remote path in Modal volume
    remote_path = f"{EXPERIMENT_NAME}"

    # Local output directory
    local_dir = Path("outputs") / EXPERIMENT_NAME
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model: {EXPERIMENT_NAME}")
    logger.info(f"From volume: {VOLUME_NAME}")
    logger.info(f"To local directory: {local_dir}")

    # Use modal volume get command (simple and reliable)
    cmd = f"modal volume get {VOLUME_NAME} {remote_path} outputs/ --force"

    logger.info(f"Running command: {cmd}")

    try:
        # Execute the download
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env={**os.environ, "MODAL_PROFILE": "carrick113"}
        )

        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            return None

        logger.info("Model downloaded successfully!")
        logger.info(f"Final model at: {local_dir}/final_model/")

        # List downloaded files
        if local_dir.exists():
            logger.info("\nDownloaded files:")
            for item in local_dir.rglob("*"):
                if item.is_file():
                    size = item.stat().st_size / (1024 * 1024)  # MB
                    rel_path = item.relative_to(local_dir)
                    logger.info(f"  - {rel_path} ({size:.2f} MB)")

        return local_dir

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None


if __name__ == "__main__":
    download_model_from_modal()
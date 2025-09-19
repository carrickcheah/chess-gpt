#!/usr/bin/env python3
"""
Push trained model to HuggingFace Hub with proper model card.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import HfApi, create_repo, upload_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_model_card(
    model_name: str,
    base_model: str,
    dataset: str,
    training_details: Dict[str, any],
) -> str:
    """
    Create model card for HuggingFace Hub.

    Args:
        model_name: Name of the fine-tuned model
        base_model: Base model used
        dataset: Dataset used for training
        training_details: Training configuration

    Returns:
        Model card markdown content
    """
    card = f"""---
language:
- en
license: apache-2.0
base_model: {base_model}
tags:
- chess
- game-playing
- magnus-carlsen
- lora
- fine-tuned
datasets:
- {dataset}
---

# {model_name}

## Model Description

This is a chess-playing language model fine-tuned to play in the style of Magnus Carlsen,
the world chess champion. The model has been trained on thousands of Magnus Carlsen's games
to learn his playing patterns and strategies.

### Model Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: {training_details.get('num_samples', 'N/A')} chess positions from Magnus Carlsen games
- **Task**: Chess move prediction in UCI notation

## Training Details

### Configuration

- **LoRA Rank**: {training_details.get('lora_r', 16)}
- **LoRA Alpha**: {training_details.get('lora_alpha', 32)}
- **Learning Rate**: {training_details.get('learning_rate', '2e-4')}
- **Batch Size**: {training_details.get('batch_size', 16)}
- **Training Steps**: {training_details.get('max_steps', 10000)}
- **Optimizer**: {training_details.get('optimizer', 'adamw_8bit')}

### Infrastructure

- **Hardware**: {training_details.get('gpu_type', 'L40S')} GPU
- **Training Framework**: Unsloth + TRL
- **Training Time**: {training_details.get('training_hours', 'N/A')} hours

## Usage

### Installation

```bash
pip install transformers torch chess
```

### Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess

# Load model
model_name = "{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare chess position
board = chess.Board()
prompt = f\"\"\"
You are Magnus Carlsen.
Game state: {{board.fen()}}
Valid moves: {{[m.uci() for m in board.legal_moves][:20]}}
Your move:
\"\"\"

# Generate move
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
move = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Predicted move: {{move}}")
```

## Evaluation Results

- **Win Rate vs Random**: {training_details.get('win_rate', 'N/A')}%
- **Legal Move Rate**: {training_details.get('legal_move_rate', 'N/A')}%
- **Average Game Length**: {training_details.get('avg_game_length', 'N/A')} moves

## Limitations

- The model may occasionally suggest illegal moves
- Performance degrades in highly tactical positions
- Endgame play may not match Magnus Carlsen's actual strength
- The model works best with standard chess positions

## Training Data

The model was trained on:
- Magnus Carlsen tournament games
- Filtered to positions where Magnus made the move
- Approximately {training_details.get('num_games', 'N/A')} games processed

## Citation

If you use this model, please cite:

```
@misc{{{model_name.replace('-', '_')}_2024,
  title={{{model_name}: Chess-Playing LLM Fine-tuned on Magnus Carlsen Games}},
  author={{Chess-GPT Team}},
  year={{2024}},
  url={{https://huggingface.co/{model_name}}}
}}
```

## License

This model is licensed under Apache 2.0. The chess games used for training are
publicly available tournament records.

## Acknowledgments

- Magnus Carlsen for the inspirational games
- Unsloth AI for training optimizations
- Modal for cloud infrastructure
"""

    return card


def validate_model_files(model_path: Path) -> bool:
    """
    Validate that all required model files exist.

    Args:
        model_path: Path to model directory

    Returns:
        True if all files present
    """
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]

    # Check for model weights (various formats)
    weight_files = list(model_path.glob("*.safetensors")) + \
                   list(model_path.glob("*.bin")) + \
                   list(model_path.glob("*.pt"))

    if not weight_files:
        logger.error("No model weight files found")
        return False

    missing_files = []
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    logger.info("All required model files present")
    return True


def get_training_details(model_path: Path) -> Dict[str, any]:
    """
    Extract training details from model config.

    Args:
        model_path: Path to model directory

    Returns:
        Training details dictionary
    """
    details = {}

    # Try to load training config
    config_path = model_path / "training_args.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                training_args = json.load(f)
                details.update(training_args)
        except Exception as e:
            logger.warning(f"Could not load training args: {e}")

    # Try to load model config for details
    model_config_path = model_path / "config.json"
    if model_config_path.exists():
        try:
            with open(model_config_path) as f:
                model_config = json.load(f)
                details["base_model"] = model_config.get("_name_or_path", "unknown")
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")

    return details


def push_model_to_hf(
    model_path: str,
    repo_name: str,
    private: bool = False,
    create_model_card_flag: bool = True,
    dataset_name: str = "MagnusInstruct/chess-positions",
) -> None:
    """
    Push model to HuggingFace Hub.

    Args:
        model_path: Local path to model
        repo_name: HuggingFace repository name
        private: Whether to create private repository
        create_model_card_flag: Whether to create model card
        dataset_name: Dataset used for training
    """
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("HUGGINGFACE HUB UPLOAD")
    logger.info("=" * 60)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Repository: {repo_name}")
    logger.info(f"Private: {private}")

    # Validate model files
    if not validate_model_files(model_path):
        logger.error("Model validation failed")
        sys.exit(1)

    # Initialize HuggingFace API
    try:
        api = HfApi()
        logger.info("HuggingFace API initialized")
    except Exception as e:
        logger.error(f"Failed to initialize HF API: {e}")
        logger.info("Make sure you're logged in: huggingface-cli login")
        sys.exit(1)

    # Create repository
    try:
        logger.info(f"Creating repository: {repo_name}")
        create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        logger.info("Repository created/verified")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        sys.exit(1)

    # Create model card if requested
    if create_model_card_flag:
        logger.info("Creating model card...")
        training_details = get_training_details(model_path)

        # Extract base model from config
        base_model = training_details.get("base_model", "unknown")

        model_card = create_model_card(
            model_name=repo_name,
            base_model=base_model,
            dataset=dataset_name,
            training_details=training_details,
        )

        # Save model card
        model_card_path = model_path / "README.md"
        with open(model_card_path, "w") as f:
            f.write(model_card)
        logger.info("Model card created")

    # Upload model files
    logger.info("Uploading model files...")
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            repo_type="model",
            commit_message="Upload fine-tuned chess model",
        )
        logger.info("Upload complete!")

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("SUCCESS")
    logger.info(f"Model available at: https://huggingface.co/{repo_name}")
    logger.info("=" * 60)


def main(
    model_path: str,
    repo_name: str,
    private: bool = False,
    create_model_card: bool = True,
    dataset_name: str = "MagnusInstruct/chess-positions",
) -> None:
    """
    Main entry point for pushing model to HuggingFace.

    Args:
        model_path: Local path to model
        repo_name: HuggingFace repository name
        private: Whether to create private repository
        create_model_card: Whether to create model card
        dataset_name: Dataset used for training
    """
    push_model_to_hf(
        model_path=model_path,
        repo_name=repo_name,
        private=private,
        create_model_card_flag=create_model_card,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
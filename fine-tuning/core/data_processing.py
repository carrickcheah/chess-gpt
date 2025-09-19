"""
Dataset preparation and preprocessing for chess fine-tuning.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import datasets
import modal
from tqdm import tqdm

from config.config import TrainingJobConfig
from core.prompt import get_prompt

logger = logging.getLogger(__name__)


def prepare_datasets(
    config: TrainingJobConfig,
    datasets_volume: modal.Volume,
    tokenizer: Any,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Prepare training and evaluation datasets with caching.

    Args:
        config: Training configuration
        datasets_volume: Modal volume for caching
        tokenizer: Model tokenizer for chat template application

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Get cache path for this dataset configuration
    dataset_cache_path = _get_path_to_cached_datasets(
        dataset_name=config.dataset_name,
        train_split_ratio=config.train_split_ratio,
        seed=config.seed,
    )

    # Try loading from cache first
    if dataset_cache_path.exists() and not config.invalidate_dataset_cache:
        logger.info(f"Loading cached datasets from {dataset_cache_path}")
        try:
            train_dataset = datasets.load_from_disk(dataset_cache_path / "train")
            eval_dataset = datasets.load_from_disk(dataset_cache_path / "eval")
            logger.info(f"Successfully loaded cached datasets")
            logger.info(f"  Training samples: {len(train_dataset):,}")
            logger.info(f"  Evaluation samples: {len(eval_dataset):,}")
            return train_dataset, eval_dataset
        except Exception as e:
            logger.warning(f"Failed to load cached datasets: {e}")
            logger.info("Will regenerate datasets")

    # Load and process dataset
    logger.info(f"Downloading dataset: {config.dataset_name}")
    try:
        dataset = datasets.load_dataset(config.dataset_name, split="train")
        logger.info(f"Dataset loaded: {len(dataset):,} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset {config.dataset_name}: {e}")
        raise

    # Sample dataset if configured
    if config.dataset_samples is not None:
        original_size = len(dataset)
        dataset = dataset.select(range(min(config.dataset_samples, len(dataset))))
        logger.info(
            f"Sampled {len(dataset):,} from {original_size:,} examples "
            f"({100 * len(dataset) / original_size:.1f}%)"
        )

    # Convert to conversation format
    logger.info("Converting to conversation format...")
    dataset = dataset.map(
        convert_to_conversation_format,
        desc="Converting to conversations",
        num_proc=config.preprocessing_workers,
    )

    # Log sample conversations before processing
    _log_sample_conversations(dataset, num_samples=3, stage="before")

    # Apply chat templates
    logger.info("Applying chat templates...")
    dataset = dataset.map(
        lambda examples: apply_chat_template(examples, tokenizer),
        batched=True,
        num_proc=config.preprocessing_workers,
        remove_columns=dataset.column_names,
        desc="Applying chat templates",
    )

    # Log sample conversations after processing
    _log_sample_conversations(dataset, num_samples=3, stage="after")

    # Split into train and eval
    logger.info(f"Splitting dataset (train={config.train_split_ratio:.1%})")
    dataset = dataset.train_test_split(
        test_size=1.0 - config.train_split_ratio,
        seed=config.seed
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Cache processed datasets
    logger.info(f"Caching processed datasets to {dataset_cache_path}")
    try:
        dataset_cache_path.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(dataset_cache_path / "train")
        eval_dataset.save_to_disk(dataset_cache_path / "eval")
        datasets_volume.commit()
        logger.info("Successfully cached datasets")
    except Exception as e:
        logger.warning(f"Failed to cache datasets: {e}")

    # Log final statistics
    logger.info("Dataset preparation complete:")
    logger.info(f"  Training samples: {len(train_dataset):,}")
    logger.info(f"  Evaluation samples: {len(eval_dataset):,}")
    logger.info(f"  Average text length: {_get_avg_length(train_dataset):.1f} tokens")

    return train_dataset, eval_dataset


def convert_to_conversation_format(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a chess position example to conversation format.

    Args:
        example: Dataset example with game state and move

    Returns:
        Example with conversation format added
    """
    try:
        # Generate prompt from game state
        prompt = get_prompt(
            game_state=example.get("game_state", ""),
            last_5_moves_uci=example.get("last_5_moves_uci", []),
            valid_moves=example.get("valid_moves", []),
        )

        # Create conversation
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example.get("next_move", "")},
        ]

        return {"conversations": conversation}

    except Exception as e:
        logger.error(f"Error converting example to conversation: {e}")
        logger.debug(f"Problematic example: {example}")
        # Return empty conversation as fallback
        return {"conversations": []}


def apply_chat_template(
    examples: Dict[str, list],
    tokenizer: Any
) -> Dict[str, list]:
    """
    Apply chat template to conversations.

    Args:
        examples: Batch of examples with conversations
        tokenizer: Tokenizer with chat template

    Returns:
        Examples with text field added
    """
    texts = []

    for conversation in examples["conversations"]:
        try:
            # Skip empty conversations
            if not conversation:
                texts.append("")
                continue

            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(formatted_text)

        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            texts.append("")

    return {"text": texts}


def _get_path_to_cached_datasets(
    dataset_name: str,
    train_split_ratio: float,
    seed: int,
) -> Path:
    """
    Get path to cached dataset in Modal volume.

    Args:
        dataset_name: Name of the dataset
        train_split_ratio: Training split ratio
        seed: Random seed

    Returns:
        Path to cached dataset directory
    """
    safe_name = dataset_name.replace("/", "--").replace("\\", "--")
    cache_key = f"{safe_name}_train{train_split_ratio:.2f}_seed{seed}"
    return Path("/datasets") / cache_key


def _log_sample_conversations(
    dataset: datasets.Dataset,
    num_samples: int = 3,
    stage: str = ""
) -> None:
    """
    Log sample conversations for debugging.

    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to log
        stage: Description of processing stage
    """
    stage_desc = f" ({stage})" if stage else ""
    logger.debug(f"Sample conversations{stage_desc}:")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if "conversations" in sample:
            logger.debug(f"Sample {i}: {sample['conversations'][:200]}...")
        elif "text" in sample:
            logger.debug(f"Sample {i}: {sample['text'][:200]}...")
        else:
            logger.debug(f"Sample {i}: {list(sample.keys())}")


def _get_avg_length(dataset: datasets.Dataset) -> float:
    """
    Calculate average text length in dataset.

    Args:
        dataset: Dataset with text field

    Returns:
        Average length in characters
    """
    if "text" not in dataset.column_names:
        return 0.0

    total_length = 0
    num_samples = min(100, len(dataset))  # Sample first 100 for efficiency

    for i in range(num_samples):
        text = dataset[i].get("text", "")
        total_length += len(text)

    return total_length / num_samples if num_samples > 0 else 0.0
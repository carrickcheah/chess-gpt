"""
Configuration management for chess fine-tuning with comprehensive validation.
"""

import logging
from datetime import datetime
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class TrainingJobConfig(BaseSettings):
    """Production-ready configuration for chess LLM fine-tuning."""

    # Model configuration
    model_name: str = Field(
        default="LiquidAI/LFM2-350M",
        description="Base model checkpoint name from HuggingFace"
    )
    max_seq_length: int = Field(
        default=2048,
        ge=512,
        le=32768,
        description="Maximum sequence length for model input"
    )
    load_in_4bit: bool = Field(
        default=False,
        description="Use 4-bit quantization for memory efficiency"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Use 8-bit quantization (fallback if 4-bit unavailable)"
    )

    # Dataset configuration
    dataset_name: str = Field(
        default="carrick113/chess-positions",
        description="HuggingFace dataset name"
    )
    dataset_samples: Optional[int] = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Number of samples to use (None for all)"
    )
    dataset_input_column: str = Field(
        default="input",
        description="Column name for input data"
    )
    dataset_output_column: str = Field(
        default="next_move",
        description="Column name for target output"
    )
    train_split_ratio: float = Field(
        default=0.9,
        ge=0.5,
        le=0.95,
        description="Ratio of data for training vs evaluation"
    )
    preprocessing_workers: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of workers for data preprocessing"
    )
    dataset_conversations_field: str = Field(
        default="conversations",
        description="Field name for conversation format"
    )
    dataset_text_field: str = Field(
        default="text",
        description="Field name for processed text"
    )
    invalidate_dataset_cache: bool = Field(
        default=False,
        description="Force regeneration of cached datasets"
    )

    # LoRA-specific hyperparameters
    lora_r: int = Field(
        default=16,
        ge=4,
        le=256,
        description="LoRA rank (higher = more capacity)"
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        le=512,
        description="LoRA scaling parameter"
    )
    lora_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Dropout for LoRA layers"
    )
    lora_bias: str = Field(
        default="none",
        pattern="^(none|all|lora_only)$",
        description="Which bias terms to train"
    )
    use_rslora: bool = Field(
        default=False,
        description="Use rank-stabilized LoRA"
    )
    lora_target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Model modules to apply LoRA to"
    )

    # General training hyperparameters
    optim: str = Field(
        default="adamw_8bit",
        pattern="^(adamw|adamw_8bit|sgd|adafactor)$",
        description="Optimizer type"
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Training batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of steps to accumulate gradients"
    )
    packing: bool = Field(
        default=False,
        description="Pack multiple sequences for efficiency"
    )
    use_gradient_checkpointing: str = Field(
        default="unsloth",
        pattern="^(unsloth|true|false)$",
        description="Gradient checkpointing strategy"
    )
    learning_rate: float = Field(
        default=2e-4,
        ge=1e-7,
        le=1e-2,
        description="Peak learning rate"
    )
    lr_scheduler_type: str = Field(
        default="cosine",
        pattern="^(linear|cosine|cosine_with_restarts|polynomial|constant)$",
        description="Learning rate scheduler"
    )
    warmup_ratio: float = Field(
        default=0.06,
        ge=0.0,
        le=0.5,
        description="Ratio of training for warmup"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=0.5,
        description="Weight decay for regularization"
    )
    max_steps: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum training steps"
    )
    save_steps: int = Field(
        default=1000,
        ge=50,
        description="Save checkpoint every N steps"
    )
    eval_steps: int = Field(
        default=1000,
        ge=50,
        description="Run evaluation every N steps"
    )
    logging_steps: int = Field(
        default=10,
        ge=1,
        description="Log metrics every N steps"
    )
    eval_sample_callback_enabled: bool = Field(
        default=False,
        description="Enable sample generation during evaluation"
    )

    # Modal configuration
    modal_app_name: str = Field(
        default="finetune-chess-llm",
        description="Modal app name"
    )
    modal_volume_pretrained_models: str = Field(
        default="pretrained_models",
        description="Volume name for model cache"
    )
    modal_volume_datasets: str = Field(
        default="datasets",
        description="Volume name for dataset cache"
    )
    modal_volume_model_checkpoints: str = Field(
        default="model_checkpoints",
        description="Volume name for checkpoints"
    )
    modal_gpu_type: str = Field(
        default="L40S",
        pattern="^(T4|L4|A10G|L40S|A100|H100)$",
        description="GPU type for Modal"
    )
    modal_timeout_hours: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Maximum runtime in hours"
    )
    modal_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )

    # Experiment configuration
    seed: int = Field(
        default=105,
        ge=0,
        description="Random seed for reproducibility"
    )
    wandb_project_name: str = Field(
        default="chess-finetuning",
        description="Weights & Biases project name"
    )
    wandb_experiment_name: Optional[str] = Field(
        default=None,
        description="Experiment name (auto-generated if None)"
    )
    wandb_enabled: bool = Field(
        default=True,
        description="Enable Weights & Biases logging"
    )
    skip_eval: bool = Field(
        default=False,
        description="Skip evaluation during training"
    )
    output_dir: str = Field(
        default="outputs",
        description="Local output directory"
    )

    class Config:
        env_prefix = "CHESS_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env

    @field_validator("lora_alpha")
    def validate_lora_alpha(cls, v: int, info) -> int:
        """Ensure LoRA alpha is reasonable relative to rank."""
        if "lora_r" in info.data:
            lora_r = info.data["lora_r"]
            ratio = v / lora_r
            if ratio < 0.5 or ratio > 4:
                logger.warning(
                    f"Unusual LoRA alpha/rank ratio: {ratio:.2f}. "
                    f"Typical range is 0.5-4. Current: alpha={v}, rank={lora_r}"
                )
        return v

    @field_validator("batch_size")
    def validate_batch_size(cls, v: int, info) -> int:
        """Validate effective batch size."""
        if "gradient_accumulation_steps" in info.data:
            grad_accum = info.data["gradient_accumulation_steps"]
            effective_batch = v * grad_accum
            if effective_batch > 256:
                logger.warning(
                    f"Large effective batch size: {effective_batch}. "
                    f"This may cause memory issues or training instability."
                )
        return v

    @field_validator("save_steps")
    def validate_save_steps(cls, v: int, info) -> int:
        """Ensure save frequency is reasonable."""
        if "max_steps" in info.data:
            max_steps = info.data["max_steps"]
            if v > max_steps / 5:
                logger.warning(
                    f"Infrequent checkpoint saving: only {max_steps // v} saves "
                    f"in {max_steps} steps. Consider reducing save_steps."
                )
        return v

    @field_validator("learning_rate")
    def validate_learning_rate(cls, v: float) -> float:
        """Warn about extreme learning rates."""
        if v > 5e-4:
            logger.warning(
                f"High learning rate {v:.2e} may cause instability. "
                f"Typical LoRA fine-tuning uses 1e-5 to 5e-4."
            )
        elif v < 1e-6:
            logger.warning(
                f"Very low learning rate {v:.2e} may result in slow convergence."
            )
        return v

    @model_validator(mode="after")
    def set_experiment_name(self) -> "TrainingJobConfig":
        """Generate experiment name if not provided."""
        if self.wandb_experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.wandb_experiment_name = (
                f"{model_short}-r{self.lora_r}-a{self.lora_alpha}-{timestamp}"
            )
            logger.info(f"Generated experiment name: {self.wandb_experiment_name}")
        return self

    @model_validator(mode="after")
    def validate_quantization(self) -> "TrainingJobConfig":
        """Ensure only one quantization method is active."""
        if self.load_in_4bit and self.load_in_8bit:
            logger.warning(
                "Both 4-bit and 8-bit quantization enabled. Using 4-bit only."
            )
            self.load_in_8bit = False
        return self

    @model_validator(mode="after")
    def validate_memory_settings(self) -> "TrainingJobConfig":
        """Check memory-related settings for compatibility."""
        if self.modal_gpu_type == "T4" and self.batch_size > 8:
            logger.warning(
                f"Batch size {self.batch_size} may be too large for T4 GPU. "
                f"Consider reducing or enabling gradient checkpointing."
            )

        if not self.load_in_4bit and not self.load_in_8bit:
            if self.modal_gpu_type in ["T4", "L4"] and "350M" not in self.model_name:
                logger.warning(
                    "Full precision training on small GPU may cause OOM. "
                    "Consider enabling 4-bit or 8-bit quantization."
                )
        return self

    def log_configuration(self) -> None:
        """Log important configuration parameters."""
        logger.info("Training Configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Dataset: {self.dataset_name} ({self.dataset_samples} samples)")
        logger.info(f"  LoRA: rank={self.lora_r}, alpha={self.lora_alpha}")
        logger.info(f"  Training: lr={self.learning_rate:.2e}, steps={self.max_steps}")
        logger.info(f"  Batch: size={self.batch_size}, grad_accum={self.gradient_accumulation_steps}")
        logger.info(f"  GPU: {self.modal_gpu_type}, timeout={self.modal_timeout_hours}h")
        logger.info(f"  Experiment: {self.wandb_experiment_name}")
# Chess Fine-Tuning System Makefile

.PHONY: help install setup clean format lint test train run evaluate deploy download-magnus-games

# Variables
PYTHON := uv run python
MODAL := MODAL_PROFILE=carrick113 uv run python -m modal
PROJECT_NAME := chess-gpt
BASE_MODEL := unsloth/LFM2-350M
EXPERIMENT_NAME := $(shell date +%Y%m%d-%H%M%S)-chess-gpt
CHECKPOINT := LFM2-350M-r16-20250902-232247/checkpoint-5000

help: ## Show this help message
	@echo "Chess Fine-Tuning System"
	@echo "========================"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing dependencies..."
	uv add torch transformers datasets
	uv add unsloth peft trl
	uv add modal wandb huggingface-hub
	uv add chess python-chess
	uv add fire tqdm jinja2
	uv add pydantic pydantic-settings python-dotenv
	uv add --dev pytest pytest-asyncio pytest-cov
	uv add --dev ruff pyright mypy
	@echo "Dependencies installed!"

setup: ## Initial setup
	@echo "Setting up project..."
	@cp -n .env.example .env 2>/dev/null || true
	@mkdir -p data/raw data/processed
	@mkdir -p models/checkpoints models/merged merged-models
	@mkdir -p outputs logs
	@echo "Project setup complete!"
	@echo "Please edit .env with your API keys"

clean: ## Clean generated files
	@echo "Cleaning up..."
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf .pytest_cache .coverage htmlcov
	@rm -rf outputs/checkpoint-*
	@find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

# Existing commands from current Makefile

download-magnus-games: ## Download Magnus Carlsen games
	@echo "Downloading Magnus Carlsen games..."
	@mkdir -p data/raw
	curl -L -o data/raw/Carlsen.zip https://www.pgnmentor.com/players/Carlsen.zip
	cd data/raw && unzip -o Carlsen.zip
	rm -f data/raw/Carlsen.zip
	@echo "Download complete!"

instruction-dataset: ## Generate instruction dataset
	@echo "Generating instruction dataset..."
	$(PYTHON) utils/generate_instruction_dataset.py \
		--hugging_face_dataset_name carrick113/chess-positions
	@echo "Dataset generated!"

# Training commands (fixed module paths)

fine-tune: ## Launch fine-tuning on Modal
	@echo "Launching fine-tuning on Modal..."
	$(MODAL) run -m main \
		--model-name $(BASE_MODEL) \
		--invalidate-dataset-cache
	@echo "Training launched!"

train: fine-tune ## Alias for fine-tune

run: train ## Quick alias for running Modal training (what you were looking for!)

evaluate: ## Evaluate model
	@echo "Evaluating model..."
	$(MODAL) run -m evaluation.evals \
		--model-checkpoint $(CHECKPOINT)
	@echo "Evaluation complete!"

merge-model: ## Merge LoRA adapters with base model
	@echo "Merging model..."
	$(PYTHON) utils/generate_merged_model.py \
		--modal-volume-name model_checkpoints \
		--remote-checkpoint-dir $(CHECKPOINT) \
		--local-merged-models-dir ./merged-models
	@echo "Model merged!"

# Bundle commands (if leap-bundle is available)

bundle-model: merge-model ## Bundle model with leap
	@echo "Creating model bundle..."
	@mkdir -p merged-models && cd merged-models && \
	uv run leap-bundle create $(CHECKPOINT) 2>/dev/null || echo "leap-bundle not installed"

list-bundles: ## List model bundles
	@uv run leap-bundle list 2>/dev/null || echo "leap-bundle not installed"

BUNDLE_ID = 6
download-model-bundle: ## Download model bundle
	@mkdir -p model-bundles && cd model-bundles && \
	uv run leap-bundle download $(BUNDLE_ID) 2>/dev/null || echo "leap-bundle not installed"

# Code quality commands

lint: ## Lint code
	@echo "Linting code..."
	$(PYTHON) -m ruff check core/ evaluation/ utils/
	@echo "Linting complete!"

lint-fix: ## Fix linting issues
	@echo "Fixing linting issues..."
	$(PYTHON) -m ruff check --fix core/ evaluation/ utils/
	@echo "Fixes applied!"

format: ## Format code
	@echo "Formatting code..."
	$(PYTHON) -m ruff format core/ evaluation/ utils/
	@echo "Code formatted!"

test: ## Run tests
	@echo "Running tests..."
	@$(PYTHON) -m pytest tests/ -v 2>/dev/null || echo "No tests found"
	@echo "Tests complete!"

# Data Processing

process-data: ## Process PGN files to instruction dataset
	@echo "Processing chess games..."
	@$(PYTHON) utils/generate_instruction_dataset.py \
		--raw_data_dir data \
		--processed_data_dir data/processed \
		--filter_player Carlsen \
		--max_games_per_file 1000
	@echo "Data processing complete!"

upload-dataset: ## Upload dataset to HuggingFace
	@echo "Uploading dataset to HuggingFace..."
	@$(PYTHON) utils/generate_instruction_dataset.py \
		--raw_data_dir data \
		--processed_data_dir data/processed \
		--hugging_face_dataset_name $(HF_USERNAME)/chess-instruct \
		--filter_player Carlsen
	@echo "Dataset uploaded!"

# Training variants

train-local: ## Test training locally (small dataset)
	@echo "Starting local training test..."
	@$(PYTHON) -m main \
		--model_name $(BASE_MODEL) \
		--max_steps 100 \
		--experiment_name local-test \
		--estimate_cost false
	@echo "Local training complete!"

train-resume: ## Resume training from checkpoint
	@echo "Resuming training..."
	@$(MODAL) run main \
		--experiment_name $(EXPERIMENT_NAME) \
		--resume_from_checkpoint true
	@echo "Training resumed!"

# Model Management

push-model: ## Push model to HuggingFace Hub
	@echo "Pushing model to HuggingFace..."
	@$(PYTHON) utils/push_model_to_hf.py \
		--model_path merged-models/$(CHECKPOINT) \
		--repo_name $(HF_USERNAME)/chess-gpt-magnus \
		--create_model_card true
	@echo "Model pushed to HuggingFace!"

deploy: merge-model push-model ## Full deployment pipeline
	@echo "Deployment complete!"

# Modal Management

modal-setup: ## Setup Modal
	@echo "Setting up Modal..."
	@$(MODAL) token new
	@$(MODAL) setup
	@echo "Modal setup complete!"

modal-logs: ## Show Modal logs
	@$(MODAL) logs -f

modal-volumes: ## List Modal volumes
	@$(MODAL) volume list

modal-jobs: ## List Modal jobs
	@$(MODAL) app list

# Monitoring

wandb-login: ## Login to Weights & Biases
	@echo "Logging into Weights & Biases..."
	@$(PYTHON) -m wandb login
	@echo "W&B login complete!"

monitor: ## Open monitoring dashboard
	@echo "Opening monitoring dashboard..."
	@open https://wandb.ai/$(WANDB_ENTITY)/chess-finetuning 2>/dev/null || echo "Set WANDB_ENTITY in .env"
	@open https://modal.com/apps 2>/dev/null || echo "Modal dashboard: https://modal.com/apps"

# Utilities

check-env: ## Check environment variables
	@echo "Checking environment..."
	@$(PYTHON) -c "from config.config import TrainingJobConfig; c = TrainingJobConfig(); c.log_configuration()" 2>/dev/null || echo "Environment not configured"
	@echo "Environment check complete!"

estimate-cost: ## Estimate training cost
	@echo "Estimating training cost..."
	@$(PYTHON) -c "from core.infra import estimate_training_cost; from config.config import TrainingJobConfig; estimate_training_cost(TrainingJobConfig())" 2>/dev/null || echo "Configure environment first"
	@echo "Cost estimation complete!"

# Quick Commands

quick-start: setup download-magnus-games process-data ## Quick setup and data prep
	@echo "Quick start complete! Ready to train."

quick-train: process-data fine-tune ## Process data and train
	@echo "Quick training pipeline complete!"

quick-eval: evaluate ## Quick evaluation
	@echo "Quick evaluation complete!"

quick-deploy: merge-model push-model ## Merge and deploy model
	@echo "Quick deployment complete!"

all: install setup download-magnus-games process-data fine-tune evaluate deploy ## Full pipeline
	@echo "Full pipeline complete!"

.DEFAULT_GOAL := help
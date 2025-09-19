"""
Model evaluation system for chess fine-tuning.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from config.config import TrainingJobConfig
from evaluation.game import ChessGame, evaluate_players
from core.infra import get_docker_image_for_evaluation, get_modal_app, get_retries, get_volume
from evaluation.players import LLMPlayer, Player, RandomPlayer

logger = logging.getLogger(__name__)

# Initialize Modal infrastructure
config = TrainingJobConfig()
modal_app = get_modal_app()
docker_image = get_docker_image_for_evaluation()
model_checkpoints_volume = get_volume(config.modal_volume_model_checkpoints)


@modal_app.function(
    image=docker_image,
    gpu=config.modal_gpu_type,
    volumes={
        "/model_checkpoints": model_checkpoints_volume,
    },
    timeout=config.modal_timeout_hours * 60 * 60,
    retries=get_retries(),
    max_inputs=1,
)
def evaluate(
    model_checkpoint_path: str,
    num_games: int = 10,
    baseline: str = "random",
) -> Dict[str, any]:
    """
    Evaluate model on Modal infrastructure.

    Args:
        model_checkpoint_path: Path to model checkpoint
        num_games: Number of games to play
        baseline: Baseline player type ("random" or another model path)

    Returns:
        Evaluation results
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {model_checkpoint_path}")
    logger.info(f"Games: {num_games}")
    logger.info(f"Baseline: {baseline}")

    # Load model checkpoint
    checkpoint_path = Path("/model_checkpoints") / model_checkpoint_path
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize AI player
    logger.info("Initializing AI player...")
    try:
        ai_player = LLMPlayer(
            model_checkpoint_path=checkpoint_path,
            name="ChessGPT",
            temperature=0.7,
            top_p=0.9,
        )
    except Exception as e:
        logger.error(f"Failed to initialize AI player: {e}")
        raise

    # Run sanity checks
    logger.info("Running sanity checks...")
    sanity_check_results = sanity_check(ai_player)
    logger.info(f"Sanity check passed: {sanity_check_results['passed']}/{sanity_check_results['total']}")

    # Initialize baseline player
    if baseline == "random":
        baseline_player = RandomPlayer(name="RandomBaseline")
    else:
        # Load another model as baseline
        baseline_path = Path("/model_checkpoints") / baseline
        baseline_player = LLMPlayer(
            model_checkpoint_path=baseline_path,
            name="BaselineModel"
        )

    logger.info(f"Baseline player: {baseline_player.name}")

    # Evaluate players
    results = evaluate_players(
        player1=ai_player,
        player2=baseline_player,
        num_games=num_games,
        alternate_colors=True,
    )

    # Add sanity check results
    results["sanity_checks"] = sanity_check_results

    # Log summary
    ai_stats = results[ai_player.name]
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model win rate: {ai_stats.get('win_rate', 0):.1%}")
    logger.info(f"Model draw rate: {ai_stats.get('draw_rate', 0):.1%}")
    logger.info(f"Model loss rate: {ai_stats.get('loss_rate', 0):.1%}")
    logger.info(f"Sanity checks: {sanity_check_results['passed']}/{sanity_check_results['total']}")

    return results


def sanity_check(player: Player) -> Dict[str, any]:
    """
    Run sanity checks on player.

    Args:
        player: Player to test

    Returns:
        Sanity check results
    """
    import chess

    test_positions = [
        {
            "name": "Opening position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "expected_moves": ["e2e4", "d2d4", "g1f3", "b1c3"],  # Common openings
        },
        {
            "name": "Checkmate in 1",
            "fen": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
            "expected_moves": ["d1d8"],  # Back rank mate
        },
        {
            "name": "Capture queen",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
            "expected_moves": ["e5d4"],  # Capture pawn
        },
    ]

    results = {
        "passed": 0,
        "total": len(test_positions),
        "tests": [],
    }

    logger.info(f"Running {len(test_positions)} sanity checks for {player.name}")

    for test in test_positions:
        board = chess.Board(test["fen"])
        try:
            move = player.get_move(board)
            passed = move in test["expected_moves"]

            results["tests"].append({
                "name": test["name"],
                "move": move,
                "expected": test["expected_moves"],
                "passed": passed,
            })

            if passed:
                results["passed"] += 1
                logger.debug(f"✓ {test['name']}: {move}")
            else:
                logger.debug(f"✗ {test['name']}: {move} (expected one of {test['expected_moves']})")

        except Exception as e:
            logger.error(f"Error in sanity check '{test['name']}': {e}")
            results["tests"].append({
                "name": test["name"],
                "error": str(e),
                "passed": False,
            })

    logger.info(f"Sanity check results: {results['passed']}/{results['total']} passed")
    return results


@modal_app.local_entrypoint()
def main(
    model_checkpoint: str,
    num_games: int = 10,
    baseline: str = "random",
) -> None:
    """
    Local entry point for evaluation.

    Args:
        model_checkpoint: Model checkpoint path
        num_games: Number of evaluation games
        baseline: Baseline player type
    """
    logger.info(f"Starting evaluation of {model_checkpoint}")
    logger.info(f"  Games: {num_games}")
    logger.info(f"  Baseline: {baseline}")

    try:
        results = evaluate.remote(
            model_checkpoint_path=model_checkpoint,
            num_games=num_games,
            baseline=baseline,
        )

        # Display results
        logger.info("\nEvaluation complete!")
        model_name = list(results.keys())[0]
        if model_name in results:
            stats = results[model_name]
            logger.info(f"Model performance:")
            logger.info(f"  Win rate: {stats.get('win_rate', 0):.1%}")
            logger.info(f"  Draw rate: {stats.get('draw_rate', 0):.1%}")
            logger.info(f"  Loss rate: {stats.get('loss_rate', 0):.1%}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    import fire
    fire.Fire(main)
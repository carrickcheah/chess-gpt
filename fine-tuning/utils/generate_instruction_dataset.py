#!/usr/bin/env python3
"""
Generate instruction dataset from PGN chess games for fine-tuning.
"""

import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import chess.pgn
from datasets import Dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.prompt import get_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def count_games(pgn_file_path: str) -> int:
    """
    Count total number of games in PGN file.

    Args:
        pgn_file_path: Path to PGN file

    Returns:
        Number of games
    """
    game_count = 0
    try:
        with open(pgn_file_path) as pgn_file:
            while chess.pgn.read_game(pgn_file) is not None:
                game_count += 1
    except Exception as e:
        logger.error(f"Error counting games: {e}")
        return 0

    return game_count


def extract_game_data(
    pgn_file_path: str,
    max_games: Optional[int] = None,
    filter_player: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse PGN file and extract move sequences with game states.

    Args:
        pgn_file_path: Path to PGN file
        max_games: Maximum number of games to process
        filter_player: Only include moves by this player

    Returns:
        List of training examples
    """
    extracted_data = []
    game_id = 0

    total_games = count_games(pgn_file_path)
    logger.info(f"Total games in PGN: {total_games}")

    if max_games:
        total_games = min(total_games, max_games)
        logger.info(f"Processing first {total_games} games")

    with open(pgn_file_path) as pgn_file:
        with tqdm(total=total_games, desc="Processing games") as pbar:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    if max_games and game_id >= max_games:
                        break

                    board = chess.Board()
                    moves_uci = []
                    white_player = game.headers.get("White", "Unknown")
                    black_player = game.headers.get("Black", "Unknown")

                    # Extract each move and corresponding game state
                    for move_number, move in enumerate(game.mainline_moves()):
                        # Get current state before applying move
                        game_state = board.fen()
                        valid_moves = [str(legal_move) for legal_move in board.legal_moves]

                        # Determine which player is making the move
                        current_player = white_player if move_number % 2 == 0 else black_player

                        # Filter by player if specified
                        if filter_player and filter_player not in current_player:
                            board.push(move)
                            moves_uci.append(str(move))
                            continue

                        # Create data point for this position
                        data_point = {
                            "moves_uci": moves_uci.copy(),
                            "last_5_moves_uci": moves_uci[-5:] if moves_uci else [],
                            "game_state": game_state,
                            "valid_moves": valid_moves[:20],  # Limit for efficiency
                            "move_number": move_number,
                            "game_id": f"game_{game_id}",
                            "next_move": str(move),
                            "player_to_move": current_player,
                        }

                        # Add game metadata
                        data_point["metadata"] = {
                            "event": game.headers.get("Event", "Unknown"),
                            "date": game.headers.get("Date", "Unknown"),
                            "result": game.headers.get("Result", "*"),
                            "white": white_player,
                            "black": black_player,
                        }

                        extracted_data.append(data_point)

                        # Apply the move and add to move list
                        board.push(move)
                        moves_uci.append(str(move))

                    game_id += 1
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error processing game {game_id}: {e}")
                    game_id += 1
                    pbar.update(1)
                    continue

    logger.info(f"Extracted {len(extracted_data)} positions from {game_id} games")
    return extracted_data


def validate_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate and generate statistics for dataset.

    Args:
        data: List of training examples

    Returns:
        Validation statistics
    """
    stats = {
        "total_examples": len(data),
        "unique_positions": len(set(d["game_state"] for d in data)),
        "unique_games": len(set(d["game_id"] for d in data)),
        "players": {},
        "move_distribution": {},
        "avg_moves_per_game": 0,
    }

    # Count player occurrences
    player_counts = {}
    for example in data:
        player = example.get("player_to_move", "Unknown")
        player_counts[player] = player_counts.get(player, 0) + 1

    stats["players"] = dict(sorted(
        player_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10])

    # Calculate average moves per game
    game_moves = {}
    for example in data:
        game_id = example["game_id"]
        game_moves[game_id] = max(
            game_moves.get(game_id, 0),
            example["move_number"] + 1
        )

    if game_moves:
        stats["avg_moves_per_game"] = sum(game_moves.values()) / len(game_moves)

    # Log statistics
    logger.info("Dataset statistics:")
    logger.info(f"  Total examples: {stats['total_examples']:,}")
    logger.info(f"  Unique positions: {stats['unique_positions']:,}")
    logger.info(f"  Unique games: {stats['unique_games']:,}")
    logger.info(f"  Avg moves/game: {stats['avg_moves_per_game']:.1f}")
    logger.info(f"  Top players:")
    for player, count in list(stats["players"].items())[:5]:
        logger.info(f"    {player}: {count:,} moves")

    return stats


def save_dataset(
    data: List[Dict[str, Any]],
    output_file: str,
    format: str = "json"
) -> None:
    """
    Save extracted data to file.

    Args:
        data: Training examples
        output_file: Output file path
        format: Output format ("json" or "parquet")
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} examples to {output_path}")

        elif format == "parquet":
            dataset = Dataset.from_list(data)
            dataset.save_to_disk(str(output_path))
            logger.info(f"Saved dataset to {output_path}")

        else:
            logger.error(f"Unknown format: {format}")

    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise


def generate_instruction_dataset(
    raw_data_dir: Optional[str] = None,
    processed_data_dir: Optional[str] = None,
    hugging_face_dataset_name: Optional[str] = None,
    max_games_per_file: Optional[int] = None,
    filter_player: str = "Carlsen",
) -> None:
    """
    Process PGN files and create instruction dataset.

    Args:
        raw_data_dir: Directory containing PGN files
        processed_data_dir: Output directory for processed data
        hugging_face_dataset_name: HuggingFace dataset repository
        max_games_per_file: Maximum games to process per file
        filter_player: Only include moves by this player
    """
    # Set default paths
    if raw_data_dir is None:
        script_path = Path(__file__).resolve()
        raw_data_dir = script_path.parent.parent / "data"

    if processed_data_dir is None:
        script_path = Path(__file__).resolve()
        processed_data_dir = script_path.parent.parent / "data" / "processed"

    # Ensure directories exist
    raw_data_dir = Path(raw_data_dir)
    processed_data_dir = Path(processed_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Find all PGN files
    pgn_files = list(raw_data_dir.glob("*.pgn"))
    logger.info(f"Found {len(pgn_files)} PGN files: {[f.name for f in pgn_files]}")

    if not pgn_files:
        logger.error(f"No PGN files found in {raw_data_dir}")
        return

    # Process each PGN file
    all_data = []
    for pgn_path in tqdm(pgn_files, desc="Processing PGN files"):
        logger.info(f"\nProcessing {pgn_path.name}...")

        # Extract game data
        data = extract_game_data(
            str(pgn_path),
            max_games=max_games_per_file,
            filter_player=filter_player
        )

        if data:
            # Save individual file data
            output_path = processed_data_dir / f"{pgn_path.stem}.json"
            save_dataset(data, str(output_path), format="json")
            all_data.extend(data)

    # Validate complete dataset
    if all_data:
        logger.info(f"\nTotal data points: {len(all_data)}")
        stats = validate_dataset(all_data)

        # Save combined dataset
        combined_path = processed_data_dir / "combined_dataset.json"
        save_dataset(all_data, str(combined_path), format="json")

        # Create HuggingFace dataset
        if hugging_face_dataset_name:
            logger.info(f"Creating HuggingFace dataset: {hugging_face_dataset_name}")
            try:
                dataset = Dataset.from_list(all_data)

                # Add instruction format
                def add_instruction_format(example):
                    prompt = get_prompt(
                        game_state=example["game_state"],
                        last_5_moves_uci=example["last_5_moves_uci"],
                        valid_moves=example["valid_moves"],
                    )
                    example["input"] = prompt
                    example["output"] = example["next_move"]
                    return example

                dataset = dataset.map(add_instruction_format)

                # Push to hub
                dataset.push_to_hub(hugging_face_dataset_name)
                logger.info(f"Dataset pushed to hub: {hugging_face_dataset_name}")

            except Exception as e:
                logger.error(f"Failed to push to HuggingFace: {e}")

    else:
        logger.warning("No data extracted")


if __name__ == "__main__":
    import fire
    fire.Fire(generate_instruction_dataset)
"""
Chess game simulation for model evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn

from evaluation.players import Player

logger = logging.getLogger(__name__)


class ChessGame:
    """Chess game simulator for player evaluation."""

    def __init__(
        self,
        white_player: Player,
        black_player: Player,
        max_moves: int = 200,
        time_limit: Optional[float] = None,
    ):
        """
        Initialize chess game.

        Args:
            white_player: Player controlling white pieces
            black_player: Player controlling black pieces
            max_moves: Maximum moves before declaring draw
            time_limit: Optional time limit per move in seconds
        """
        self.white_player = white_player
        self.black_player = black_player
        self.max_moves = max_moves
        self.time_limit = time_limit
        self.board = chess.Board()
        self.move_history: List[str] = []
        self.move_times: List[float] = []

        logger.info(f"Game initialized: {white_player.name} (White) vs {black_player.name} (Black)")

    def play(self) -> Dict[str, any]:
        """
        Play a complete game.

        Returns:
            Dictionary with game result and statistics
        """
        import time

        logger.info("Starting game...")
        move_count = 0
        illegal_moves = {"white": 0, "black": 0}

        while not self.board.is_game_over() and move_count < self.max_moves:
            current_player = self.white_player if self.board.turn else self.black_player
            color = "White" if self.board.turn else "Black"

            # Get move from player
            start_time = time.time()
            try:
                move_uci = current_player.get_move(self.board)
                elapsed_time = time.time() - start_time

                # Validate and apply move
                if move_uci:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        self.move_history.append(move_uci)
                        self.move_times.append(elapsed_time)
                        move_count += 1

                        if move_count % 10 == 0:
                            logger.debug(f"Move {move_count}: {color} played {move_uci}")
                    else:
                        logger.warning(f"Illegal move by {color}: {move_uci}")
                        illegal_moves[color.lower()] += 1
                        # Force random legal move
                        self._make_random_move()
                        move_count += 1
                else:
                    logger.warning(f"No move returned by {color}")
                    self._make_random_move()
                    move_count += 1

            except Exception as e:
                logger.error(f"Error getting move from {color}: {e}")
                self._make_random_move()
                move_count += 1

        # Determine result
        result = self._determine_result()
        statistics = self._calculate_statistics(result, illegal_moves)

        logger.info(f"Game ended: {result['outcome']}")
        logger.info(f"Winner: {result.get('winner', 'None')}")
        logger.info(f"Total moves: {move_count}")

        return statistics

    def _determine_result(self) -> Dict[str, str]:
        """
        Determine game result.

        Returns:
            Dictionary with outcome and winner
        """
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            return {"outcome": "checkmate", "winner": winner}
        elif self.board.is_stalemate():
            return {"outcome": "stalemate", "winner": None}
        elif self.board.is_insufficient_material():
            return {"outcome": "insufficient_material", "winner": None}
        elif self.board.is_fifty_moves():
            return {"outcome": "fifty_move_rule", "winner": None}
        elif self.board.is_repetition():
            return {"outcome": "repetition", "winner": None}
        elif len(self.move_history) >= self.max_moves:
            return {"outcome": "max_moves", "winner": None}
        else:
            return {"outcome": "unknown", "winner": None}

    def _calculate_statistics(
        self,
        result: Dict[str, str],
        illegal_moves: Dict[str, int]
    ) -> Dict[str, any]:
        """
        Calculate game statistics.

        Args:
            result: Game result
            illegal_moves: Count of illegal moves by each player

        Returns:
            Comprehensive statistics dictionary
        """
        stats = {
            "outcome": result["outcome"],
            "winner": result.get("winner"),
            "white_player": self.white_player.name,
            "black_player": self.black_player.name,
            "total_moves": len(self.move_history),
            "move_history": self.move_history,
            "illegal_moves": illegal_moves,
        }

        if self.move_times:
            stats["avg_move_time"] = sum(self.move_times) / len(self.move_times)
            stats["max_move_time"] = max(self.move_times)
            stats["min_move_time"] = min(self.move_times)

        # Add position-specific metrics
        stats["final_position"] = self.board.fen()
        stats["material_balance"] = self._calculate_material_balance()

        return stats

    def _calculate_material_balance(self) -> int:
        """
        Calculate material balance (positive favors white).

        Returns:
            Material balance value
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        balance = 0
        for square, piece in self.board.piece_map().items():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                balance += value
            else:
                balance -= value

        return balance

    def _make_random_move(self) -> None:
        """Make a random legal move."""
        import random

        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            self.board.push(move)
            self.move_history.append(move.uci())
            self.move_times.append(0.0)

    def to_pgn(self) -> str:
        """
        Convert game to PGN format.

        Returns:
            PGN string representation
        """
        game = chess.pgn.Game()
        game.headers["White"] = self.white_player.name
        game.headers["Black"] = self.black_player.name

        node = game
        board = chess.Board()

        for move_uci in self.move_history:
            try:
                move = chess.Move.from_uci(move_uci)
                node = node.add_variation(move)
                board.push(move)
            except:
                break

        return str(game)


def evaluate_players(
    player1: Player,
    player2: Player,
    num_games: int = 10,
    alternate_colors: bool = True,
) -> Dict[str, any]:
    """
    Evaluate two players against each other.

    Args:
        player1: First player
        player2: Second player
        num_games: Number of games to play
        alternate_colors: Whether to alternate colors

    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating {player1.name} vs {player2.name} ({num_games} games)")

    results = {
        player1.name: {"wins": 0, "draws": 0, "losses": 0},
        player2.name: {"wins": 0, "draws": 0, "losses": 0},
        "games": [],
    }

    for game_num in range(num_games):
        # Determine colors
        if alternate_colors and game_num % 2 == 1:
            white_player, black_player = player2, player1
        else:
            white_player, black_player = player1, player2

        logger.info(f"Game {game_num + 1}/{num_games}: {white_player.name} vs {black_player.name}")

        # Play game
        game = ChessGame(white_player, black_player)
        result = game.play()
        results["games"].append(result)

        # Update scores
        if result["winner"] == "White":
            results[white_player.name]["wins"] += 1
            results[black_player.name]["losses"] += 1
        elif result["winner"] == "Black":
            results[black_player.name]["wins"] += 1
            results[white_player.name]["losses"] += 1
        else:
            results[player1.name]["draws"] += 1
            results[player2.name]["draws"] += 1

    # Calculate win rates
    for player_name in [player1.name, player2.name]:
        stats = results[player_name]
        total = stats["wins"] + stats["draws"] + stats["losses"]
        if total > 0:
            stats["win_rate"] = stats["wins"] / total
            stats["draw_rate"] = stats["draws"] / total
            stats["loss_rate"] = stats["losses"] / total

    logger.info("Evaluation complete:")
    logger.info(f"  {player1.name}: {results[player1.name]['wins']}W/{results[player1.name]['draws']}D/{results[player1.name]['losses']}L")
    logger.info(f"  {player2.name}: {results[player2.name]['wins']}W/{results[player2.name]['draws']}D/{results[player2.name]['losses']}L")

    return results
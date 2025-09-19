"""
Chess player interfaces for evaluation.
"""

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import chess

logger = logging.getLogger(__name__)


class Player(ABC):
    """Abstract base class for chess players."""

    def __init__(self, name: str = "Player"):
        self.name = name

    @abstractmethod
    def get_move(self, board: chess.Board) -> str:
        """
        Get next move for the given board position.

        Args:
            board: Current chess board state

        Returns:
            Move in UCI notation
        """
        pass

    def __str__(self) -> str:
        return self.name


class RandomPlayer(Player):
    """Player that makes random legal moves."""

    def __init__(self, name: str = "RandomPlayer"):
        super().__init__(name)
        logger.info(f"Initialized {self.name}")

    def get_move(self, board: chess.Board) -> str:
        """
        Select a random legal move.

        Args:
            board: Current chess board state

        Returns:
            Random legal move in UCI notation
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            logger.warning("No legal moves available")
            return ""

        move = random.choice(legal_moves)
        return move.uci()


class LLMPlayer(Player):
    """Player powered by a fine-tuned language model."""

    def __init__(
        self,
        model_checkpoint_path: Optional[Path] = None,
        name: str = "LLMPlayer",
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize LLM player.

        Args:
            model_checkpoint_path: Path to model checkpoint
            name: Player name
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        super().__init__(name)
        self.model_checkpoint_path = model_checkpoint_path
        self.temperature = temperature
        self.top_p = top_p
        self.model = None
        self.tokenizer = None

        if model_checkpoint_path:
            self._load_model()

    def _load_model(self) -> None:
        """Load model and tokenizer from checkpoint."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"Loading model from {self.model_checkpoint_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_checkpoint_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            self.model.eval()
            logger.info(f"Model loaded successfully for {self.name}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_move(self, board: chess.Board) -> str:
        """
        Get move from language model.

        Args:
            board: Current chess board state

        Returns:
            Predicted move in UCI notation
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded")
            return self._fallback_move(board)

        # Prepare input
        prompt = self._create_prompt(board)

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            # Generate move
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode output
            generated = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Extract move from generated text
            move = self._extract_move(generated, board)

            if move:
                logger.debug(f"{self.name} selected move: {move}")
                return move
            else:
                logger.warning(f"Invalid move generated: {generated}")
                return self._fallback_move(board)

        except Exception as e:
            logger.error(f"Error generating move: {e}")
            return self._fallback_move(board)

    def _create_prompt(self, board: chess.Board) -> str:
        """
        Create prompt for the model.

        Args:
            board: Current chess board state

        Returns:
            Formatted prompt
        """
        from core.prompt import get_prompt

        # Get recent moves
        move_stack = board.move_stack
        last_5_moves = [m.uci() for m in move_stack[-5:]] if move_stack else []

        # Get legal moves
        legal_moves = [m.uci() for m in board.legal_moves]

        # Create prompt
        prompt = get_prompt(
            game_state=board.fen(),
            last_5_moves_uci=last_5_moves,
            valid_moves=legal_moves[:20],  # Limit for token efficiency
        )

        return prompt

    def _extract_move(self, generated: str, board: chess.Board) -> Optional[str]:
        """
        Extract valid move from generated text.

        Args:
            generated: Generated text from model
            board: Current board for validation

        Returns:
            Valid move in UCI notation or None
        """
        import re

        # Try to find UCI notation (e.g., e2e4, f8c8)
        uci_pattern = r'[a-h][1-8][a-h][1-8][qrbn]?'
        matches = re.findall(uci_pattern, generated.lower())

        for match in matches:
            try:
                move = chess.Move.from_uci(match)
                if move in board.legal_moves:
                    return match
            except:
                continue

        # Try algebraic notation conversion
        algebraic_pattern = r'[NBRQK]?[a-h]?[1-8]?[x]?[a-h][1-8][+#]?'
        matches = re.findall(algebraic_pattern, generated)

        for match in matches:
            try:
                move = board.parse_san(match)
                if move:
                    return move.uci()
            except:
                continue

        return None

    def _fallback_move(self, board: chess.Board) -> str:
        """
        Fallback to random move if model fails.

        Args:
            board: Current chess board state

        Returns:
            Random legal move
        """
        logger.warning(f"{self.name} using fallback random move")
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves).uci()
        return ""


class HumanPlayer(Player):
    """Interactive human player."""

    def __init__(self, name: str = "Human"):
        super().__init__(name)

    def get_move(self, board: chess.Board) -> str:
        """
        Get move from human input.

        Args:
            board: Current chess board state

        Returns:
            Human-selected move in UCI notation
        """
        print(f"\n{board}")
        print(f"\n{self.name}'s turn. Legal moves:")

        legal_moves = list(board.legal_moves)
        for i, move in enumerate(legal_moves[:20]):
            print(f"  {move.uci()}", end="  ")
            if (i + 1) % 5 == 0:
                print()

        if len(legal_moves) > 20:
            print(f"  ... and {len(legal_moves) - 20} more")

        while True:
            try:
                move_input = input("\nEnter move (UCI format, e.g., e2e4): ").strip()
                move = chess.Move.from_uci(move_input)

                if move in board.legal_moves:
                    return move.uci()
                else:
                    print(f"Invalid move: {move_input}. Please try again.")

            except KeyboardInterrupt:
                print("\nGame interrupted")
                raise
            except:
                print(f"Invalid format. Please use UCI notation (e.g., e2e4)")
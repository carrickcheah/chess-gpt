//
//  ChessGame.swift
//  ChessGPT
//
//  Chess game logic separated from UI components
//

import Foundation
import ChessKit
import Observation
import os.log

@Observable
class ChessGame {
    // Game state
    var game: Game
    var moves: [Move] = []
    var gameState: GameState = .playing
    var currentPlayer: PlayerType = .human

    // Game history and analysis
    var moveHistory: [String] = []
    var capturedPieces: [PieceType] = []

    private let logger = Logger(subsystem: "com.chessgpt.game", category: "chess-engine")

    // MARK: - Initialization

    init() {
        self.game = Game()
        logger.info("New chess game initialized")
    }

    // MARK: - Game Control

    func makeMove(_ move: Move) throws {
        logger.info("Attempting move: \(move.description)")

        guard gameState == .playing else {
            logger.warning("Attempted move while game not in playing state: \(gameState)")
            throw ChessGameError.gameNotActive
        }

        guard game.legalMoves.contains(move) else {
            logger.warning("Illegal move attempted: \(move.description)")
            throw ChessGameError.illegalMove(move.description)
        }

        // Record move details before making it
        let moveNotation = move.description
        let capturedPiece = game.position.piece(at: move.targetSquare)

        // Make the move
        game.make(move: move)
        moves.append(move)
        moveHistory.append(moveNotation)

        // Record captured piece
        if let captured = capturedPiece {
            capturedPieces.append(captured.kind)
            logger.debug("Piece captured: \(captured.kind)")
        }

        // Update game state
        updateGameState()
        switchPlayer()

        logger.info("Move completed: \(moveNotation)")
    }

    func makeMoveFromUCI(_ uciMove: String) throws {
        logger.info("Converting UCI move: \(uciMove)")

        // Find the move that matches the UCI notation
        guard let move = game.legalMoves.first(where: { $0.description == uciMove }) else {
            logger.error("UCI move not found in legal moves: \(uciMove)")
            throw ChessGameError.invalidUCIMove(uciMove)
        }

        try makeMove(move)
    }

    func restartGame() {
        logger.info("Restarting chess game")
        game = Game()
        moves.removeAll()
        moveHistory.removeAll()
        capturedPieces.removeAll()
        gameState = .playing
        currentPlayer = .human
    }

    func undoLastMove() -> Bool {
        guard !moves.isEmpty else {
            logger.warning("Cannot undo: no moves to undo")
            return false
        }

        // Note: ChessKit doesn't have built-in undo, so we reconstruct the game
        let lastMove = moves.removeLast()
        moveHistory.removeLast()

        logger.info("Undoing move: \(lastMove.description)")

        // Reconstruct game from remaining moves
        game = Game()
        for move in moves {
            game.make(move: move)
        }

        updateGameState()
        switchPlayer()
        return true
    }

    // MARK: - Game State Management

    private func updateGameState() {
        if game.isCheckmate {
            gameState = currentPlayer == .human ? .aiWins : .humanWins
            logger.info("Game ended: checkmate, winner: \(gameState)")
        } else if game.isStalemate {
            gameState = .draw
            logger.info("Game ended: stalemate")
        } else if game.isCheck {
            logger.info("Check detected")
        }
    }

    private func switchPlayer() {
        currentPlayer = currentPlayer == .human ? .ai : .human
        logger.debug("Switched to player: \(currentPlayer)")
    }

    // MARK: - Game Information

    var isHumanTurn: Bool {
        currentPlayer == .human && gameState == .playing
    }

    var isAITurn: Bool {
        currentPlayer == .ai && gameState == .playing
    }

    var isGameActive: Bool {
        gameState == .playing
    }

    var currentFEN: String {
        FenSerialization.default.serialize(position: game.position)
    }

    var legalMoves: [Move] {
        game.legalMoves
    }

    var legalMovesUCI: [String] {
        game.legalMoves.map(\.description)
    }

    var lastFiveMoves: [String] {
        Array(moveHistory.suffix(5))
    }

    var moveCount: Int {
        moves.count
    }

    var isCheck: Bool {
        game.isCheck
    }

    var isCheckmate: Bool {
        game.isCheckmate
    }

    var isStalemate: Bool {
        game.isStalemate
    }

    // MARK: - Statistics

    func getGameStatistics() -> GameStatistics {
        return GameStatistics(
            totalMoves: moveCount,
            capturedPieces: capturedPieces.count,
            gameState: gameState,
            isCheck: isCheck,
            currentFEN: currentFEN
        )
    }
}

// MARK: - Supporting Types

enum PlayerType: String {
    case human = "Human"
    case ai = "AI"
}

enum GameState: String {
    case playing = "Playing"
    case humanWins = "Human Wins"
    case aiWins = "AI Wins"
    case draw = "Draw"
}

struct GameStatistics {
    let totalMoves: Int
    let capturedPieces: Int
    let gameState: GameState
    let isCheck: Bool
    let currentFEN: String
}

// MARK: - Error Types

enum ChessGameError: LocalizedError {
    case illegalMove(String)
    case invalidUCIMove(String)
    case gameNotActive
    case noMovesToUndo

    var errorDescription: String? {
        switch self {
        case .illegalMove(let move):
            return "Illegal move: \(move)"
        case .invalidUCIMove(let uci):
            return "Invalid UCI move: \(uci)"
        case .gameNotActive:
            return "Game is not currently active"
        case .noMovesToUndo:
            return "No moves available to undo"
        }
    }
}
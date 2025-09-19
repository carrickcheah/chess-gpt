//
//  GameManager.swift
//  ChessGPT
//
//  Manages game state, moves, and game flow
//

import Foundation
import ChessKit
import Observation

@Observable
final class GameManager {

    // MARK: - Public Properties

    private(set) var currentGame: Game
    private(set) var gameState: GameState = .waitingForPlayer
    private(set) var moveHistory: [GameMove] = []
    private(set) var gameResult: GameResult?

    // MARK: - Configuration

    private let startingPosition = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    // MARK: - Initialization

    init() {
        self.currentGame = Game(fen: startingPosition)
    }

    // MARK: - Public Methods

    /// Start a new game
    func startNewGame() {
        currentGame = Game(fen: startingPosition)
        gameState = .waitingForPlayer
        moveHistory.removeAll()
        gameResult = nil

        print("🎮 New game started")
    }

    /// Make a player move
    func makePlayerMove(_ move: Move) throws -> Bool {
        guard gameState == .waitingForPlayer else {
            throw GameError.invalidGameState
        }

        guard currentGame.legalMoves.contains(move) else {
            throw GameError.illegalMove(move.description)
        }

        try executeMove(move, player: .human)
        checkGameStatus()

        if gameState == .inProgress {
            gameState = .waitingForAI
        }

        return true
    }

    /// Make an AI move
    func makeAIMove(_ move: Move) throws -> Bool {
        guard gameState == .waitingForAI else {
            throw GameError.invalidGameState
        }

        guard currentGame.legalMoves.contains(move) else {
            throw GameError.illegalMove(move.description)
        }

        try executeMove(move, player: .ai)
        checkGameStatus()

        if gameState == .inProgress {
            gameState = .waitingForPlayer
        }

        return true
    }

    /// Get current FEN string
    var currentFEN: String {
        return FenSerialization.default.serialize(position: currentGame.position)
    }

    /// Get legal moves for current position
    var legalMoves: [Move] {
        return Array(currentGame.legalMoves)
    }

    /// Check if it's the player's turn
    var isPlayerTurn: Bool {
        return gameState == .waitingForPlayer
    }

    /// Check if it's the AI's turn
    var isAITurn: Bool {
        return gameState == .waitingForAI
    }

    /// Get game statistics
    var gameStats: GameStatistics {
        return GameStatistics(
            totalMoves: moveHistory.count,
            humanMoves: moveHistory.filter { $0.player == .human }.count,
            aiMoves: moveHistory.filter { $0.player == .ai }.count,
            gameResult: gameResult
        )
    }

    // MARK: - Private Methods

    private func executeMove(_ move: Move, player: Player) throws {
        let moveNotation = move.description
        let capturedPiece = currentGame.position.piece(at: move.to)

        // Record the move before executing
        let gameMove = GameMove(
            move: move,
            notation: moveNotation,
            player: player,
            timestamp: Date(),
            capturedPiece: capturedPiece,
            isCheck: false, // Will be updated after move
            isCheckmate: false
        )

        // Execute the move
        currentGame.make(move: move)

        // Update move with check/checkmate info
        var updatedMove = gameMove
        updatedMove.isCheck = currentGame.position.isCheck
        updatedMove.isCheckmate = currentGame.isCheckmate

        moveHistory.append(updatedMove)

        print("📝 Move executed: \(player.rawValue) played \(moveNotation)")
    }

    private func checkGameStatus() {
        if currentGame.isCheckmate {
            let winner: Player = currentGame.position.sideToMove == .white ? .ai : .human
            gameResult = .checkmate(winner: winner)
            gameState = .gameOver
            print("🏆 Game Over: Checkmate! Winner: \(winner.rawValue)")

        } else if currentGame.isStalemate {
            gameResult = .stalemate
            gameState = .gameOver
            print("🤝 Game Over: Stalemate")

        } else if currentGame.isDrawByRepetition {
            gameResult = .draw(.repetition)
            gameState = .gameOver
            print("🤝 Game Over: Draw by repetition")

        } else if currentGame.isDrawByInsufficientMaterial {
            gameResult = .draw(.insufficientMaterial)
            gameState = .gameOver
            print("🤝 Game Over: Draw by insufficient material")

        } else {
            gameState = .inProgress
        }
    }
}

// MARK: - Supporting Types

enum GameState {
    case waitingForPlayer
    case waitingForAI
    case inProgress
    case gameOver
}

enum Player: String, CaseIterable {
    case human = "Human"
    case ai = "ChessGPT"
}

enum GameResult: Equatable {
    case checkmate(winner: Player)
    case stalemate
    case draw(DrawReason)
}

enum DrawReason {
    case repetition
    case insufficientMaterial
    case fiftyMoveRule
    case agreement
}

struct GameMove {
    let move: Move
    let notation: String
    let player: Player
    let timestamp: Date
    let capturedPiece: Piece?
    var isCheck: Bool
    var isCheckmate: Bool
}

struct GameStatistics {
    let totalMoves: Int
    let humanMoves: Int
    let aiMoves: Int
    let gameResult: GameResult?

    var averageThinkingTime: TimeInterval {
        // This could be calculated if we track move times
        return 0.0
    }
}

// MARK: - Error Types

enum GameError: Error, LocalizedError {
    case illegalMove(String)
    case invalidGameState
    case gameAlreadyFinished

    var errorDescription: String? {
        switch self {
        case .illegalMove(let move):
            return "Illegal move: \(move)"
        case .invalidGameState:
            return "Invalid game state for this operation"
        case .gameAlreadyFinished:
            return "Game has already finished"
        }
    }
}
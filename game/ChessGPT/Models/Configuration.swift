//
//  Configuration.swift
//  ChessGPT
//
//  Configuration management for chess app settings
//

import Foundation
import Observation

@Observable
class AppConfiguration {
    // AI Model Settings
    var modelBundleName = "LFM2-350M-MagnusInstruct"
    var temperature: Float = 0.8
    var maxTokens: Int = 50

    // Game Settings
    var playerName = "Player"
    var difficulty: Difficulty = .intermediate
    var showValidMoves = true
    var animateAIMoves = true

    // UI Settings
    var boardSize: BoardSize = .medium
    var soundEnabled = true
    var hapticFeedback = true

    // Performance Settings
    var enableModelCaching = true
    var maxConversationHistory = 10

    // Chess Prompt Template
    var promptTemplate: String {
        """
        You are the great Magnus Carlsen.
        Your task is to make the best move in the given game state.

        Game state: {{game_state}}
        Last 5 moves: {{last_5_moves_uci}}
        Valid moves: {{valid_moves}}

        Your next move should be in UCI format (e.g., 'e2e4', 'f8c8').
        Make sure your next move is one of the valid moves.
        """
    }

    enum Difficulty: String, CaseIterable {
        case beginner = "Beginner"
        case intermediate = "Intermediate"
        case advanced = "Advanced"
        case expert = "Expert"

        var temperature: Float {
            switch self {
            case .beginner: return 1.2     // More random moves
            case .intermediate: return 0.8  // Balanced
            case .advanced: return 0.5     // More focused
            case .expert: return 0.2       // Very precise
            }
        }
    }

    enum BoardSize: String, CaseIterable {
        case small = "Small"
        case medium = "Medium"
        case large = "Large"

        var size: CGFloat {
            switch self {
            case .small: return 300
            case .medium: return 350
            case .large: return 400
            }
        }
    }
}
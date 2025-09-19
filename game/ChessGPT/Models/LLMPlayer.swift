//
//  LLMPlayer.swift
//  ChessGPT
//
//  Optimized AI player with local inference and memory management
//

import Foundation
import LeapSDK
import ChessKit
import Observation
import os.log

@Observable
class LLMPlayer {
    var isReady: Bool = false
    var isThinking: Bool = false
    var error: String?

    private var modelRunner: ModelRunner?
    private var conversation: Conversation?
    private let logger = Logger(subsystem: "com.chessgpt.ai", category: "llm-player")

    // MARK: - Initialization

    func setupLLM() async {
        logger.info("Starting LLM setup...")

        do {
            guard let modelURL = Bundle.main.url(
                forResource: "LFM2-350M-MagnusInstruct",
                withExtension: "bundle"
            ) else {
                let errorMsg = "Could not find model bundle"
                logger.error("\(errorMsg)")
                error = errorMsg
                isReady = true
                return
            }

            logger.info("Loading model from bundle: \(modelURL.lastPathComponent)")
            modelRunner = try await Leap.load(url: modelURL)

            // FIXED: Create conversation ONCE, not for every move
            guard let runner = modelRunner else {
                throw LLMError.modelNotInitialized
            }

            conversation = Conversation(modelRunner: runner, history: [])
            isReady = true
            logger.info("LLM setup completed successfully")

        } catch {
            let errorMsg = "Failed to load model: \(error.localizedDescription)"
            logger.error("\(errorMsg)")
            self.error = errorMsg
            isReady = true
        }
    }

    // MARK: - Chess Move Generation

    func getNextMove(game: Game, configuration: AppConfiguration) async throws -> String {
        logger.info("Generating next move...")
        isThinking = true
        defer { isThinking = false }

        guard let conversation = conversation else {
            logger.error("Conversation not initialized")
            throw LLMError.modelNotInitialized
        }

        // Prepare game state data
        let gameState = FenSerialization.default.serialize(position: game.position)
        let last5MovesUci = game.movesHistory.suffix(5).map(\.description)
        let validMoves = game.legalMoves.map(\.description)

        logger.debug("Game state: \(gameState)")
        logger.debug("Valid moves count: \(validMoves.count)")

        // Generate prompt using configuration template
        let prompt = generatePrompt(
            gameState: gameState,
            last5MovesUci: last5MovesUci,
            validMoves: validMoves,
            template: configuration.promptTemplate
        )

        let userMessage = ChatMessage(role: .user, content: [.text(prompt)])

        // Use configuration-based temperature
        var options = GenerationOptions()
        options.temperature = configuration.difficulty.temperature
        options.maxTokens = configuration.maxTokens

        logger.info("Generating response with temperature: \(options.temperature)")

        let stream = conversation.generateResponse(
            message: userMessage,
            generationOptions: options
        )

        var assistantResponse = ""

        for try await response in stream {
            switch response {
            case .chunk(let text):
                assistantResponse += text
            case .reasoningChunk(_):
                break
            case .complete(_, _):
                let finalMove = assistantResponse.trimmingCharacters(in: .whitespacesAndNewlines)
                logger.info("Generated move: \(finalMove)")

                // Validate move is in valid moves list
                if !validMoves.contains(finalMove) {
                    logger.warning("Generated invalid move: \(finalMove), valid moves: \(validMoves)")
                    throw LLMError.invalidMove(finalMove)
                }

                return finalMove
            case .functionCall(_):
                break
            }
        }

        let finalMove = assistantResponse.trimmingCharacters(in: .whitespacesAndNewlines)
        logger.info("Generated move: \(finalMove)")
        return finalMove
    }

    // MARK: - Memory Management

    func clearConversationHistory() {
        logger.info("Clearing conversation history for memory optimization")
        guard let runner = modelRunner else { return }
        conversation = Conversation(modelRunner: runner, history: [])
    }

    func getConversationLength() -> Int {
        return conversation?.history.count ?? 0
    }

    // MARK: - Private Methods

    private func generatePrompt(
        gameState: String,
        last5MovesUci: [String],
        validMoves: [String],
        template: String
    ) -> String {
        return template
            .replacingOccurrences(of: "{{game_state}}", with: gameState)
            .replacingOccurrences(of: "{{last_5_moves_uci}}", with: last5MovesUci.joined(separator: ", "))
            .replacingOccurrences(of: "{{valid_moves}}", with: validMoves.joined(separator: ", "))
    }
}

// MARK: - Error Types

enum LLMError: LocalizedError {
    case invalidMove(String)
    case illegalMove(String)
    case noValidMoves
    case modelNotFound
    case modelNotInitialized

    var errorDescription: String? {
        switch self {
        case .invalidMove(let move):
            return "Invalid move generated: \(move)"
        case .illegalMove(let move):
            return "Illegal move attempted: \(move)"
        case .noValidMoves:
            return "No valid moves available"
        case .modelNotFound:
            return "Model bundle not found in app resources"
        case .modelNotInitialized:
            return "Model not properly initialized"
        }
    }
}
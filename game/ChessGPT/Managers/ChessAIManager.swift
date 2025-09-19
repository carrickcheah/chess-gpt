//
//  ChessAIManager.swift
//  ChessGPT
//
//  Advanced AI Manager with improved error handling and performance
//

import Foundation
import LeapSDK
import ChessKit
import Observation

@Observable
final class ChessAIManager {

    // MARK: - State Properties

    private(set) var isReady: Bool = false
    private(set) var isThinking: Bool = false
    private(set) var error: ChessAIError?
    private(set) var lastMoveTime: TimeInterval = 0

    // MARK: - Private Properties

    private var modelRunner: ModelRunner?
    private let modelBundleName = "LFM2-350M-MagnusInstruct"
    private let maxRetryAttempts = 3

    // MARK: - Configuration

    private struct AIConfiguration {
        static let temperature: Float = 0.8
        static let maxTokens: Int = 10
        static let timeoutInterval: TimeInterval = 30.0
    }

    // MARK: - Public Methods

    /// Initialize the AI model
    func initialize() async {
        await MainActor.run {
            isReady = false
            error = nil
        }

        do {
            let modelURL = try getModelURL()
            modelRunner = try await Leap.load(url: modelURL)

            await MainActor.run {
                isReady = true
            }

            print("✅ ChessAI initialized successfully")

        } catch let aiError as ChessAIError {
            await handleError(aiError)
        } catch {
            await handleError(.modelLoadingFailed(error.localizedDescription))
        }
    }

    /// Get the next move for the current game state
    func getNextMove(for game: Game) async throws -> Move {
        guard isReady else {
            throw ChessAIError.modelNotReady
        }

        await MainActor.run {
            isThinking = true
            error = nil
        }

        defer {
            Task { @MainActor in
                isThinking = false
            }
        }

        let startTime = Date()

        do {
            let move = try await generateMoveWithRetry(for: game)

            await MainActor.run {
                lastMoveTime = Date().timeIntervalSince(startTime)
            }

            print("🎯 AI move generated: \(move) in \(lastMoveTime)s")
            return move

        } catch {
            await handleError(.moveGenerationFailed(error.localizedDescription))
            throw error
        }
    }

    // MARK: - Private Methods

    private func getModelURL() throws -> URL {
        guard let modelURL = Bundle.main.url(
            forResource: modelBundleName,
            withExtension: "bundle"
        ) else {
            throw ChessAIError.modelBundleNotFound(modelBundleName)
        }
        return modelURL
    }

    private func generateMoveWithRetry(for game: Game) async throws -> Move {
        var lastError: Error?

        for attempt in 1...maxRetryAttempts {
            do {
                let move = try await generateMove(for game)

                // Validate the move
                guard game.legalMoves.contains(move) else {
                    throw ChessAIError.illegalMove(move.description)
                }

                return move

            } catch {
                lastError = error
                print("⚠️ AI move attempt \(attempt) failed: \(error)")

                if attempt < maxRetryAttempts {
                    // Brief delay before retry
                    try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
                }
            }
        }

        throw lastError ?? ChessAIError.moveGenerationFailed("All retry attempts failed")
    }

    private func generateMove(for game: Game) async throws -> Move {
        guard let modelRunner = modelRunner else {
            throw ChessAIError.modelNotReady
        }

        let conversation = Conversation(modelRunner: modelRunner, history: [])
        let prompt = createPrompt(for: game)
        let userMessage = ChatMessage(role: .user, content: [.text(prompt)])

        var options = GenerationOptions()
        options.temperature = AIConfiguration.temperature
        options.maxTokens = AIConfiguration.maxTokens

        let stream = conversation.generateResponse(
            message: userMessage,
            generationOptions: options
        )

        var response = ""

        // Add timeout protection
        let timeoutTask = Task {
            try await Task.sleep(nanoseconds: UInt64(AIConfiguration.timeoutInterval * 1_000_000_000))
            throw ChessAIError.responseTimeout
        }

        let responseTask = Task {
            for try await chunk in stream {
                switch chunk {
                case .chunk(let text):
                    response += text
                case .complete(_, _):
                    return response.trimmingCharacters(in: .whitespacesAndNewlines)
                default:
                    break
                }
            }
            return response.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let result = try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask { try await responseTask.value }
            group.addTask { try await timeoutTask.value }

            let firstResult = try await group.next()!
            group.cancelAll()
            return firstResult
        }

        // Parse the UCI move
        let cleanedResponse = result.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedResponse.isEmpty else {
            throw ChessAIError.emptyResponse
        }

        return Move(string: cleanedResponse)
    }

    private func createPrompt(for game: Game) -> String {
        let gameState = FenSerialization.default.serialize(position: game.position)
        let recentMoves = game.movesHistory.suffix(5).map(\.description)
        let validMoves = game.legalMoves.map(\.description)

        return """
        You are Magnus Carlsen, the world chess champion.
        Analyze the position and choose the best move.

        Current position (FEN): \(gameState)
        Recent moves: \(recentMoves.joined(separator: ", "))
        Valid moves: \(validMoves.joined(separator: ", "))

        Respond with only the best move in UCI format (e.g., "e2e4").
        Choose from the valid moves only.
        """
    }

    @MainActor
    private func handleError(_ error: ChessAIError) {
        self.error = error
        print("❌ ChessAI Error: \(error.localizedDescription)")
    }
}

// MARK: - Error Types

enum ChessAIError: Error, LocalizedError {
    case modelBundleNotFound(String)
    case modelLoadingFailed(String)
    case modelNotReady
    case moveGenerationFailed(String)
    case illegalMove(String)
    case emptyResponse
    case responseTimeout
    case invalidMoveFormat(String)

    var errorDescription: String? {
        switch self {
        case .modelBundleNotFound(let name):
            return "Model bundle '\(name)' not found in app bundle"
        case .modelLoadingFailed(let reason):
            return "Failed to load model: \(reason)"
        case .modelNotReady:
            return "AI model is not ready. Please wait for initialization."
        case .moveGenerationFailed(let reason):
            return "Failed to generate move: \(reason)"
        case .illegalMove(let move):
            return "AI generated illegal move: \(move)"
        case .emptyResponse:
            return "AI returned empty response"
        case .responseTimeout:
            return "AI response timed out"
        case .invalidMoveFormat(let move):
            return "Invalid move format: \(move)"
        }
    }
}

// MARK: - Extensions

extension ChessAIManager {

    /// Get AI performance statistics
    var performanceStats: (isReady: Bool, lastMoveTime: TimeInterval, hasError: Bool) {
        return (isReady: isReady, lastMoveTime: lastMoveTime, hasError: error != nil)
    }

    /// Reset the AI state
    func reset() async {
        await MainActor.run {
            error = nil
            isThinking = false
            lastMoveTime = 0
        }
    }
}
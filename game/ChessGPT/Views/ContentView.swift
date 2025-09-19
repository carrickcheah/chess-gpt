//
//  ContentView.swift
//  ChessGPT
//
//  Main view coordinating game, AI, and configuration
//

import SwiftUI
import ChessboardKit
import ChessKit
import os.log

struct ContentView: View {
    @Environment(LLMPlayer.self) private var llmPlayer
    @Environment(AppConfiguration.self) private var configuration

    @State private var chessGame = ChessGame()
    @State private var showError = false
    @State private var errorMessage = ""

    private let logger = Logger(subsystem: "com.chessgpt.ui", category: "content-view")

    var body: some View {
        VStack {
            if !llmPlayer.isReady {
                LoadingView()
                    .task {
                        await llmPlayer.setupLLM()
                    }
            } else {
                GameView(
                    chessGame: $chessGame,
                    showError: $showError,
                    errorMessage: $errorMessage
                )
            }
        }
        .alert("Error", isPresented: $showError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
        .onAppear {
            logger.info("ContentView appeared")
        }
    }
}

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)

            Text("Loading Magnus AI...")
                .font(.title2)
                .fontWeight(.medium)

            Text("Preparing local chess intelligence")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
    }
}

struct GameView: View {
    @Environment(LLMPlayer.self) private var llmPlayer
    @Environment(AppConfiguration.self) private var configuration

    @Binding var chessGame: ChessGame
    @Binding var showError: Bool
    @Binding var errorMessage: String

    private let logger = Logger(subsystem: "com.chessgpt.ui", category: "game-view")

    var body: some View {
        VStack(spacing: 20) {
            // Game Status
            GameStatusView(chessGame: chessGame)

            // Chess Board
            BoardView(
                chessGame: $chessGame,
                showError: $showError,
                errorMessage: $errorMessage
            )

            // Game Controls
            GameControlsView(chessGame: $chessGame)

            Spacer()
        }
        .padding()
        .task(id: chessGame.isAITurn) {
            if chessGame.isAITurn && chessGame.isGameActive {
                await handleAIMove()
            }
        }
    }

    private func handleAIMove() async {
        logger.info("Handling AI move")

        do {
            let aiMove = try await llmPlayer.getNextMove(
                game: chessGame.game,
                configuration: configuration
            )

            try chessGame.makeMoveFromUCI(aiMove)
            logger.info("AI move completed: \(aiMove)")

        } catch {
            logger.error("AI move failed: \(error.localizedDescription)")
            errorMessage = "AI move failed: \(error.localizedDescription)"
            showError = true
        }
    }
}

struct GameStatusView: View {
    let chessGame: ChessGame

    var body: some View {
        VStack(spacing: 8) {
            Text("Chess vs Magnus AI")
                .font(.title)
                .fontWeight(.bold)

            HStack {
                Text("Turn: \(chessGame.currentPlayer.rawValue)")
                    .font(.headline)

                Spacer()

                Text("Moves: \(chessGame.moveCount)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if chessGame.isCheck {
                Text("Check!")
                    .font(.headline)
                    .foregroundColor(.red)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
            }

            if chessGame.gameState != .playing {
                Text(chessGame.gameState.rawValue)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(12)
            }
        }
        .padding(.horizontal)
    }
}

#Preview {
    ContentView()
        .environment(LLMPlayer())
        .environment(AppConfiguration())
}
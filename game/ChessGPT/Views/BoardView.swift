//
//  BoardView.swift
//  ChessGPT
//
//  Modern chessboard view with improved UX and animations
//

import SwiftUI
import ChessboardKit
import ChessKit

struct BoardView: View {

    // MARK: - Environment

    @Environment(ChessAIManager.self) private var aiManager
    @Environment(GameManager.self) private var gameManager

    // MARK: - State

    @State private var chessboardModel: ChessboardModel
    @State private var showingError = false
    @State private var errorMessage = ""
    @State private var showingGameResult = false
    @State private var isProcessingMove = false
    @State private var boardSize: CGFloat = 350

    // MARK: - Initialization

    init() {
        let initialFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        _chessboardModel = State(initialValue: ChessboardModel(
            fen: initialFEN,
            perspective: .white,
            colorScheme: .light
        ))
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 24) {
            headerSection
            gameStatusSection
            chessboardSection
            controlsSection
        }
        .padding()
        .background(backgroundGradient)
        .alert("Game Result", isPresented: $showingGameResult) {
            Button("New Game") {
                startNewGame()
            }
            Button("OK") { }
        } message: {
            Text(gameResultMessage)
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
        .onChange(of: gameManager.gameResult) { _, newResult in
            if newResult != nil {
                showingGameResult = true
            }
        }
    }

    // MARK: - View Components

    private var headerSection: some View {
        VStack(spacing: 8) {
            Text("ChessGPT")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundStyle(.primary)

            Text("Play against Magnus Carlsen AI")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }

    private var gameStatusSection: some View {
        HStack {
            // Current player indicator
            HStack(spacing: 8) {
                Circle()
                    .fill(gameManager.isPlayerTurn ? .green : .gray)
                    .frame(width: 8, height: 8)
                    .animation(.easeInOut, value: gameManager.isPlayerTurn)

                Text(gameManager.isPlayerTurn ? "Your Turn" : "ChessGPT Thinking...")
                    .font(.subheadline)
                    .fontWeight(.medium)
            }

            Spacer()

            // AI status indicator
            if aiManager.isThinking {
                HStack(spacing: 4) {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Thinking")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else if !aiManager.isReady {
                HStack(spacing: 4) {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Loading AI")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.horizontal)
    }

    private var chessboardSection: some View {
        Chessboard(chessboardModel: chessboardModel)
            .onMove { move, isLegal, from, to, lan, promotionPiece in
                handlePlayerMove(move, isLegal: isLegal, notation: lan)
            }
            .frame(width: boardSize, height: boardSize)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .shadow(color: .black.opacity(0.2), radius: 8, x: 0, y: 4)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(.quaternary, lineWidth: 1)
            )
            .scaleEffect(isProcessingMove ? 0.98 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: isProcessingMove)
    }

    private var controlsSection: some View {
        HStack(spacing: 16) {
            Button(action: startNewGame) {
                Label("New Game", systemImage: "arrow.clockwise")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(
                        LinearGradient(
                            colors: [.blue, .blue.opacity(0.8)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .disabled(isProcessingMove)

            Button(action: undoLastMove) {
                Label("Undo", systemImage: "arrow.uturn.backward")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(
                        LinearGradient(
                            colors: [.orange, .orange.opacity(0.8)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .disabled(gameManager.moveHistory.isEmpty || isProcessingMove)
        }
    }

    private var backgroundGradient: some View {
        LinearGradient(
            colors: [
                Color(.systemBackground),
                Color(.systemGroupedBackground)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }

    // MARK: - Computed Properties

    private var gameResultMessage: String {
        guard let result = gameManager.gameResult else { return "" }

        switch result {
        case .checkmate(let winner):
            return "\(winner.rawValue) wins by checkmate!"
        case .stalemate:
            return "Game ended in stalemate."
        case .draw(let reason):
            switch reason {
            case .repetition:
                return "Game ended in a draw by repetition."
            case .insufficientMaterial:
                return "Game ended in a draw due to insufficient material."
            case .fiftyMoveRule:
                return "Game ended in a draw by the fifty-move rule."
            case .agreement:
                return "Game ended in a draw by agreement."
            }
        }
    }

    // MARK: - Actions

    private func handlePlayerMove(_ move: Move, isLegal: Bool, notation: String) {
        guard !isProcessingMove else { return }
        guard isLegal else {
            showError("Illegal move: \(notation)")
            return
        }

        guard gameManager.isPlayerTurn else {
            showError("It's not your turn!")
            return
        }

        isProcessingMove = true

        Task {
            do {
                // Make player move
                try gameManager.makePlayerMove(move)

                await MainActor.run {
                    updateChessboard()
                }

                // If game is still in progress, get AI move
                if gameManager.isAITurn {
                    let aiMove = try await aiManager.getNextMove(for: gameManager.currentGame)
                    try gameManager.makeAIMove(aiMove)

                    await MainActor.run {
                        updateChessboard()
                    }
                }

            } catch {
                await MainActor.run {
                    showError(error.localizedDescription)
                }
            }

            await MainActor.run {
                isProcessingMove = false
            }
        }
    }

    private func startNewGame() {
        gameManager.startNewGame()
        chessboardModel.setFen(gameManager.currentFEN)
        showingGameResult = false
        showingError = false
    }

    private func undoLastMove() {
        // Implementation for undo functionality
        // This would require more complex logic to undo both player and AI moves
        showError("Undo functionality coming soon!")
    }

    private func updateChessboard() {
        chessboardModel.setFen(gameManager.currentFEN)
    }

    private func showError(_ message: String) {
        errorMessage = message
        showingError = true
        print("⚠️ BoardView Error: \(message)")
    }
}

// MARK: - Preview

#Preview {
    BoardView()
        .environment(ChessAIManager())
        .environment(GameManager())
}
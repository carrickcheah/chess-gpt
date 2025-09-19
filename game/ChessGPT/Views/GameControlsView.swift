//
//  GameControlsView.swift
//  ChessGPT
//
//  Game control UI components
//

import SwiftUI
import os.log

struct GameControlsView: View {
    @Environment(AppConfiguration.self) private var configuration
    @Environment(LLMPlayer.self) private var llmPlayer

    @Binding var chessGame: ChessGame

    @State private var showSettings = false
    @State private var showStatistics = false

    private let logger = Logger(subsystem: "com.chessgpt.ui", category: "game-controls")

    var body: some View {
        VStack(spacing: 16) {
            // Primary Controls
            HStack(spacing: 20) {
                // Restart Game
                Button(action: restartGame) {
                    Label("New Game", systemImage: "arrow.clockwise")
                }
                .buttonStyle(PrimaryButtonStyle())

                // Undo Move (only if human's last move)
                Button(action: undoMove) {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                }
                .buttonStyle(SecondaryButtonStyle())
                .disabled(!canUndoMove)
            }

            // Secondary Controls
            HStack(spacing: 20) {
                // Settings
                Button(action: { showSettings = true }) {
                    Label("Settings", systemImage: "gear")
                }
                .buttonStyle(SecondaryButtonStyle())

                // Statistics
                Button(action: { showStatistics = true }) {
                    Label("Stats", systemImage: "chart.bar")
                }
                .buttonStyle(SecondaryButtonStyle())

                // Memory Management
                if llmPlayer.getConversationLength() > configuration.maxConversationHistory {
                    Button(action: clearAIMemory) {
                        Label("Clear AI Memory", systemImage: "memorychip")
                    }
                    .buttonStyle(TertiaryButtonStyle())
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(configuration: configuration)
        }
        .sheet(isPresented: $showStatistics) {
            StatisticsView(chessGame: chessGame)
        }
    }

    // MARK: - Actions

    private func restartGame() {
        logger.info("Restarting game")
        chessGame.restartGame()
        llmPlayer.clearConversationHistory()
    }

    private func undoMove() {
        logger.info("Undoing move")
        _ = chessGame.undoLastMove()
    }

    private func clearAIMemory() {
        logger.info("Clearing AI conversation memory")
        llmPlayer.clearConversationHistory()
    }

    // MARK: - Computed Properties

    private var canUndoMove: Bool {
        chessGame.moveCount > 0 && chessGame.isHumanTurn
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @ObservedObject var configuration: AppConfiguration

    var body: some View {
        NavigationView {
            Form {
                Section("Game Settings") {
                    Picker("Difficulty", selection: $configuration.difficulty) {
                        ForEach(AppConfiguration.Difficulty.allCases, id: \.self) { difficulty in
                            Text(difficulty.rawValue).tag(difficulty)
                        }
                    }

                    Picker("Board Size", selection: $configuration.boardSize) {
                        ForEach(AppConfiguration.BoardSize.allCases, id: \.self) { size in
                            Text(size.rawValue).tag(size)
                        }
                    }

                    Toggle("Show Valid Moves", isOn: $configuration.showValidMoves)
                    Toggle("Animate AI Moves", isOn: $configuration.animateAIMoves)
                }

                Section("Audio & Haptics") {
                    Toggle("Sound Effects", isOn: $configuration.soundEnabled)
                    Toggle("Haptic Feedback", isOn: $configuration.hapticFeedback)
                }

                Section("Performance") {
                    Toggle("Model Caching", isOn: $configuration.enableModelCaching)

                    Stepper(
                        "Max History: \(configuration.maxConversationHistory)",
                        value: $configuration.maxConversationHistory,
                        in: 5...50,
                        step: 5
                    )
                }

                Section("AI Settings") {
                    HStack {
                        Text("Temperature")
                        Spacer()
                        Text(String(format: "%.1f", configuration.temperature))
                    }

                    Slider(
                        value: $configuration.temperature,
                        in: 0.1...2.0,
                        step: 0.1
                    )

                    Stepper(
                        "Max Tokens: \(configuration.maxTokens)",
                        value: $configuration.maxTokens,
                        in: 20...200,
                        step: 10
                    )
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// MARK: - Statistics View

struct StatisticsView: View {
    let chessGame: ChessGame

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                let stats = chessGame.getGameStatistics()

                StatCard(title: "Total Moves", value: "\(stats.totalMoves)")
                StatCard(title: "Captured Pieces", value: "\(stats.capturedPieces)")
                StatCard(title: "Game Status", value: stats.gameState.rawValue)

                if stats.isCheck {
                    StatCard(title: "Check Status", value: "In Check", color: .red)
                }

                Spacer()

                Text("Current Position (FEN)")
                    .font(.headline)

                Text(stats.currentFEN)
                    .font(.caption)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    .textSelection(.enabled)
            }
            .padding()
            .navigationTitle("Game Statistics")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

struct StatCard: View {
    let title: String
    let value: String
    var color: Color = .primary

    var body: some View {
        HStack {
            Text(title)
                .font(.headline)
            Spacer()
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Button Styles

struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundColor(.white)
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Color.blue)
            .cornerRadius(25)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline)
            .foregroundColor(.blue)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.blue.opacity(0.1))
            .cornerRadius(20)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct TertiaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.caption)
            .foregroundColor(.orange)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.orange.opacity(0.1))
            .cornerRadius(15)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

#Preview {
    @State var chessGame = ChessGame()

    return GameControlsView(chessGame: $chessGame)
        .environment(AppConfiguration())
        .environment(LLMPlayer())
}
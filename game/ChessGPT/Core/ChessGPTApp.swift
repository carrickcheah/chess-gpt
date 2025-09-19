//
//  ChessGPTApp.swift
//  ChessGPT
//
//  Created by Chess GPT Team
//

import SwiftUI
import LeapSDK
import Observation

@main
struct ChessGPTApp: App {
    @State private var aiPlayer = ChessAIManager()
    @State private var gameManager = GameManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(aiPlayer)
                .environment(gameManager)
                .task {
                    await setupApplication()
                }
        }
    }

    // MARK: - Private Methods

    private func setupApplication() async {
        await aiPlayer.initialize()
    }
}
//
//  ChessGPTApp.swift
//  ChessGPT
//
//  Optimized chess app with local AI inference
//

import SwiftUI
import LeapSDK
import Observation
import os.log

@main
struct ChessGPTApp: App {
    @State private var llmPlayer = LLMPlayer()
    @State private var configuration = AppConfiguration()

    private let logger = Logger(subsystem: "com.chessgpt.app", category: "main")

    init() {
        logger.info("ChessGPT app starting with local AI inference")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(llmPlayer)
                .environment(configuration)
                .onAppear {
                    logger.info("Main view appeared")
                }
        }
    }
}
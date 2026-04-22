//
//  main.swift
//  rag​-eval
//
//  Created by Roger Pingleton on 4/21/26.
//  Copyright © 2026 Roger Pingleton. All rights reserved.
//  Command-line tool for running RAG evaluations.
//

import Foundation

// MARK: - Argument parsing

var dbPath: String?
var mode = "rag"  // default mode
var queryParts: [String] = []

var args = CommandLine.arguments.dropFirst() // skip executable name
while let arg = args.first {
    args = args.dropFirst()
    if arg == "--db" {
        guard let path = args.first else {
            fputs("Error: --db requires a path argument\n", stderr)
            exit(1)
        }
        dbPath = String(path)
        args = args.dropFirst()
    } else if arg == "--mode" {
        guard let value = args.first else {
            fputs("Error: --mode requires a value (rag or raw)\n", stderr)
            exit(1)
        }
        let lowered = value.lowercased()
        guard lowered == "rag" || lowered == "raw" else {
            fputs("Error: --mode must be 'rag' or 'raw'\n", stderr)
            exit(1)
        }
        mode = lowered
        args = args.dropFirst()
    } else {
        queryParts.append(arg)
    }
}

guard !queryParts.isEmpty else {
    let name = URL(fileURLWithPath: CommandLine.arguments[0]).lastPathComponent
    fputs("""
    Usage: \(name) [--db <path-to-rag.sqlite>] [--mode rag|raw] <query>

    Options:
      --db <path>     Path to rag.sqlite. If omitted, looks next to the executable.
      --mode rag|raw  Choose 'rag' for RAG-enhanced answers (default) or
                      'raw' for plain foundation model answers.

    Examples:
      \(name) "What is RAG?"
      \(name) --mode raw "What is RAG?"
      \(name) --db ./rag.sqlite --mode rag "How do transformers work?"

    """, stderr)
    exit(1)
}

let query = queryParts.joined(separator: " ")

// MARK: - Locate database

let databaseURL: URL
if let dbPath {
    databaseURL = URL(fileURLWithPath: dbPath)
} else {
    // Look next to the executable (works with Copy Files build phase)
    let execDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    databaseURL = execDir.appendingPathComponent("rag.sqlite")
}

guard FileManager.default.fileExists(atPath: databaseURL.path) else {
    fputs("Error: database not found at \(databaseURL.path)\n", stderr)
    fputs("Use --db <path> to specify the database location.\n", stderr)
    exit(1)
}

// MARK: - Run query

let rag = try await MainActor.run {
    try RAGSystem(databaseURL: databaseURL)
}

if mode == "raw" {
    let result = try await rag.answerRAW(query: query)
    print(result)
} else {
    let answer = try await rag.answer(query: query)
    print(answer.text)
}

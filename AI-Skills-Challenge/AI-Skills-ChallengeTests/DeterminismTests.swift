//
//  DeterminismTests.swift
//  AI-Skills-ChallengeTests
//
//  Created by Roger Pingleton on 4/20/26.
//  Copyright © 2026 Roger Pingleton. All rights reserved.
//

import Testing
import NaturalLanguage
@testable import AI_Skills_Challenge

@Suite("Retrieval determinism")
struct RetrievalDeterminismTests {

    // Shared fixtures — created once per test method instance.
    // Swift Testing creates a fresh struct per test, so this is effectively
    // a fresh setup each time without needing setUp/tearDown.
    let store: VectorStore
    let retriever: BruteForceRetriever
    let embedder: NLEmbedding

    init() throws {
        self.store = try VectorStoreLoader.loadFromBundle(name: "rag")
        self.retriever = BruteForceRetriever(store)
        self.embedder = try #require(
            NLEmbedding.sentenceEmbedding(for: .english),
            "English sentence embedding unavailable on this platform"
        )
    }

    /// Same query, called 20 times, must return byte-identical results.
    @Test("Retrieval is deterministic across repeated calls",
          arguments: [
            "What is RAG?",
            "How is Linear Algebra used in AI?",
            "How important are evals?"
          ])
    func sameQuerySameResults(question: String) throws {
        let queryVec = try embedQuery(question)

        let baseline = retriever.search(query: queryVec, k: 10)
        #expect(!baseline.isEmpty, "retriever returned nothing for: \(question)")

        for iteration in 1...20 {
            let again = retriever.search(query: queryVec, k: 10)

            #expect(
                baseline.map(\.chunkId) == again.map(\.chunkId),
                "chunk IDs differ on iteration \(iteration) for query: \(question)"
            )

            for (a, b) in zip(baseline, again) {
                #expect(
                    a.score == b.score,
                    "score drifted for chunk \(a.chunkId) on iteration \(iteration): \(a.score) vs \(b.score)"
                )
            }
        }
    }

    /// Scores should be in strictly non-increasing order.
    @Test("Results are ranked highest score first")
    func resultsAreRanked() throws {
        let queryVec = try embedQuery("How do I reset my device?")
        let hits = retriever.search(query: queryVec, k: 10)

        for (a, b) in zip(hits, hits.dropFirst()) {
            #expect(a.score >= b.score,
                    "ranking violated: \(a.chunkId)@\(a.score) before \(b.chunkId)@\(b.score)")
        }
    }

    /// Tied scores must break in a stable, documented way (lower chunkId wins).
    /// This test only has teeth if your retriever implements the tiebreaker;
    /// it will otherwise document current behavior without asserting it.
    @Test("Ties break deterministically by chunk ID")
    func tiesBreakDeterministically() throws {
        let queryVec = try embedQuery("How do I reset my device?")
        let runs = (0..<10).map { _ in retriever.search(query: queryVec, k: 20) }
        let first = runs[0].map(\.chunkId)
        for (i, run) in runs.enumerated().dropFirst() {
            #expect(first == run.map(\.chunkId),
                    "run \(i) produced a different ordering than run 0")
        }
    }

    // MARK: - Helpers

    private func embedQuery(_ text: String) throws -> [Float] {
        let raw = try #require(
            embedder.vector(for: text),
            "embedder returned nil for: \(text)"
        )
        var v = raw.map { Float($0) }
        normalizeInPlace(&v)
        return v
    }
}

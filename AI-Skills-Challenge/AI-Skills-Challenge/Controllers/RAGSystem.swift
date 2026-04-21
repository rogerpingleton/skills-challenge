//
//  RAGSystem.swift
//  AI-Skills-Challenge
//
//  Created by Roger Pingleton on 4/20/26.
//  Copyright © 2026 Roger Pingleton. All rights reserved.
//

import Foundation
import NaturalLanguage
import FoundationModels

@MainActor
public final class RAGSystem {
    private let store: VectorStore
    private let retriever: BruteForceRetriever
    private let embedder: NLEmbedding

    public init() throws {
        guard let e = NLEmbedding.sentenceEmbedding(for: .english) else {
            throw RAGError.embeddingUnavailable
        }
        self.embedder = e
        self.store = try VectorStoreLoader.loadFromBundle(name: "rag")
        self.retriever = BruteForceRetriever(store)
    }

    /// Initializes the RAG system from a specific database file URL.
    /// Use this for command-line tools where Bundle.main is unavailable.
    public init(databaseURL: URL) throws {
        guard let e = NLEmbedding.sentenceEmbedding(for: .english) else {
            throw RAGError.embeddingUnavailable
        }
        self.embedder = e
        self.store = try VectorStoreLoader.load(from: databaseURL)
        self.retriever = BruteForceRetriever(store)
    }


    public struct RetrievedChunk {
        public let chunk: Chunk
        public let score: Float
    }

    public func retrieve(query: String, k: Int = 6) -> [RetrievedChunk] {
        guard let rawVec = embedder.vector(for: query) else { return [] }
        var q = rawVec.map { Float($0) }
        normalizeInPlace(&q)
        let hits = retriever.search(query: q, k: k)
        let byId = Dictionary(uniqueKeysWithValues: store.chunks.map { ($0.id, $0) })
        return hits.compactMap { hit in
            byId[hit.chunkId].map { RetrievedChunk(chunk: $0, score: hit.score) }
        }
    }

    public func answer(query: String, maxContextTokens: Int = 2400) async throws -> Answer {
        let retrieved = retrieve(query: query, k: 8)
        let packed = packContext(retrieved, budget: maxContextTokens)

//        let instructions = """
//        You answer questions using ONLY the provided passages. \
//        If the passages don't contain the answer, say so.
//        """
        
        let instructions = """
        You answer questions using ONLY the provided passages. \
        If the passages don't contain the answer, say so, \
        and suggest that perhaps the user's question may be too narrow.
        """
        
        
//        let instructions = """
//        You answer questions to the best of your ability using the provided passages. \
//        Speculate if no passages are provided.
//        """

        let contextBlock = packed.chunks.enumerated().map { idx, rc in
            "[\(idx + 1)] \(rc.chunk.text)"
        }.joined(separator: "\n\n")
        
        //print("RAG Context Debug: \(contextBlock)\n")
        
        let prompt = Prompt("""
        Passages:
        \(contextBlock)

        Question: \(query)
        """)
        
        let options = GenerationOptions(
            sampling: .greedy,
            temperature: 0.0
        )

        let session = LanguageModelSession { instructions }
        let response = try await session.respond(to: prompt, options: options)
        return Answer(text: response.content, citations: packed.chunks)
    }

    /// Greedily include retrieved chunks in descending-score order while
    /// respecting the token budget reserved for context.
    private func packContext(_ retrieved: [RetrievedChunk], budget: Int) -> Packed {
        let tokenizer = NLTokenizer(unit: .word)
        var total = 0
        var kept: [RetrievedChunk] = []
        for rc in retrieved {
            tokenizer.string = rc.chunk.text
            let tc = tokenizer.tokens(for: rc.chunk.text.startIndex..<rc.chunk.text.endIndex).count
            if total + tc > budget { continue }
            kept.append(rc); total += tc
        }
        return Packed(chunks: kept, tokenCount: total)
    }

    public struct Answer {
        public let text: String
        public let citations: [RetrievedChunk]
    }
    struct Packed { let chunks: [RetrievedChunk]; let tokenCount: Int }
}

public enum RAGError: Error { case embeddingUnavailable, databaseMissing }


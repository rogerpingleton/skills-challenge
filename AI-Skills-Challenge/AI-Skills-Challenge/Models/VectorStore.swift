//
//  VectorStore.swift
//  AI-Skills-Challenge
//
//  Created by Roger Pingleton on 4/20/26.
//  Copyright © 2026 Roger Pingleton. All rights reserved.
//

import Foundation
import Accelerate

public struct Chunk {
    public let id: Int64
    public let docId: Int64
    public let startOffset: Int
    public let endOffset: Int
    public let startSent: Int
    public let endSent: Int
    public let text: String
    public let embedding: [Float]         // 512, L2-normalized
    public let bits: [UInt8]              // 64 bytes, optional
}

public struct Hit {
    public let chunkId: Int64
    public let score: Float               // higher = better (cosine similarity)
}

/// Packs the entire corpus's vectors into a single contiguous [Float] of
/// length N*512. Contiguous memory is the single biggest factor in scan speed.
public final class VectorStore {
    public let dim: Int = 512
    public let count: Int
    /// Row-major N x 512 Float matrix.
    public let matrix: [Float]
    /// Parallel array of chunks (same order as matrix rows).
    public let chunks: [Chunk]
    /// Binary sketches packed contiguously: N x 64 bytes.
    public let bits: [UInt8]

    public init(chunks: [Chunk]) {
        self.chunks = chunks
        self.count = chunks.count
        var m = [Float](); m.reserveCapacity(chunks.count * 512)
        for c in chunks { m.append(contentsOf: c.embedding) }
        self.matrix = m
        var b = [UInt8](); b.reserveCapacity(chunks.count * 64)
        for c in chunks { b.append(contentsOf: c.bits) }
        self.bits = b
    }
}

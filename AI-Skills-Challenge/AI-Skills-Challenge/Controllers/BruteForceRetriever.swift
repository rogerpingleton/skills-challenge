//
//  BruteForceRetriever.swift
//  AI-Skills-Challenge
//
//  Created by Roger Pingleton on 4/20/26.
//  Copyright © 2026 Roger Pingleton. All rights reserved.
//

import Foundation
import Accelerate
import NaturalLanguage
import FoundationModels

public final class BruteForceRetriever {
    private let store: VectorStore
    public init(_ store: VectorStore) { self.store = store }

    /// Returns top-k chunks by cosine similarity.
    public func search(query: [Float], k: Int) -> [Hit] {
        precondition(query.count == store.dim)
        var q = query
        normalizeInPlace(&q)                      // defensive

        var scores = [Float](repeating: 0, count: store.count)
        // scores = M (N x D)  *  q (D)
        store.matrix.withUnsafeBufferPointer { mPtr in
            q.withUnsafeBufferPointer { qPtr in
                scores.withUnsafeMutableBufferPointer { sPtr in
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(store.count), Int32(store.dim),
                        1.0,
                        mPtr.baseAddress, Int32(store.dim),
                        qPtr.baseAddress, 1,
                        0.0,
                        sPtr.baseAddress, 1
                    )
                }
            }
        }
        return topK(scores: scores, k: k)
    }

    private func topK(scores: [Float], k: Int) -> [Hit] {
        // Partial selection: faster than full sort for small k.
        let n = scores.count
        let k = min(k, n)
        var idxs = Array(0..<n)
        idxs.withUnsafeMutableBufferPointer { buf in
            // Heap-based partial sort: O(n log k)
            var heap: [(Float, Int)] = []; heap.reserveCapacity(k)
            for i in 0..<n {
                let s = scores[i]
                if heap.count < k {
                    heap.append((s, i))
                    if heap.count == k { heap.sort { $0.0 < $1.0 } }   // min-heap by value
                } else if s > heap[0].0 {
                    heap[0] = (s, i)
                    // Re-heapify: cheap linear sift for small k
                    var j = 0
                    while true {
                        let l = 2*j+1, r = 2*j+2; var m = j
                        if l < k && heap[l].0 < heap[m].0 { m = l }
                        if r < k && heap[r].0 < heap[m].0 { m = r }
                        if m == j { break }
                        heap.swapAt(j, m); j = m
                    }
                }
            }
            heap.sort { $0.0 > $1.0 }   // descending
            buf.initialize(repeating: 0)
            for (i, h) in heap.enumerated() { buf[i] = h.1 }
        }
        return (0..<k).map { i in
            Hit(chunkId: store.chunks[idxs[i]].id, score: scores[idxs[i]])
        }
    }
}

func normalizeInPlace(_ v: inout [Float]) {
    var s: Float = 0
    v.withUnsafeBufferPointer { p in
        vDSP_svesq(p.baseAddress!, 1, &s, vDSP_Length(v.count))
    }
    var n = 1.0 / max(sqrt(s), 1e-12)
    vDSP_vsmul(v, 1, &n, &v, 1, vDSP_Length(v.count))
}


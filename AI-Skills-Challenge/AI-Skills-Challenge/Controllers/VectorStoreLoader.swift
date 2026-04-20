//
//  VectorStoreLoader.swift
//  AI-Skills-Challenge
//
//  Created by Roger Pingleton on 4/20/26.
//  Copyright © 2026 Roger Pingleton. All rights reserved.
//

import Foundation
import SQLite3

public enum VectorStoreError: Error {
    case databaseNotFound(name: String)
    case cannotOpen(message: String)
    case prepareFailed(message: String)
    case schemaVersionMismatch(found: String, expected: String)
    case dimensionMismatch(found: Int, expected: Int)
    case corruptBlob(chunkId: Int64, reason: String)
}

public enum VectorStoreLoader {

    private static let expectedDim = 512
    private static let expectedSchemaVersion = "1"

    /// Loads the prebuilt RAG database shipped in the main bundle.
    /// - Parameter name: Resource name without extension (e.g. "rag" for rag.sqlite).
    public static func loadFromBundle(name: String) throws -> VectorStore {
        guard let url = Bundle.main.url(forResource: name, withExtension: "sqlite") else {
            throw VectorStoreError.databaseNotFound(name: name)
        }
        return try load(from: url)
    }

    /// Loads from an arbitrary URL (useful for tests and previews).
    public static func load(from url: URL) throws -> VectorStore {
        var db: OpaquePointer?
        let flags = SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX
        let rc = sqlite3_open_v2(url.path, &db, flags, nil)
        guard rc == SQLITE_OK, let db else {
            let msg = db.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            if let db { sqlite3_close(db) }
            throw VectorStoreError.cannotOpen(message: msg)
        }
        defer { sqlite3_close(db) }

        try validateMeta(db: db)
        let chunks = try loadChunks(db: db)
        return VectorStore(chunks: chunks)
    }

    // MARK: - Private

    private static func validateMeta(db: OpaquePointer) throws {
        var meta: [String: String] = [:]
        var stmt: OpaquePointer?
        let sql = "SELECT key, value FROM meta"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw VectorStoreError.prepareFailed(
                message: String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        while sqlite3_step(stmt) == SQLITE_ROW {
            let k = String(cString: sqlite3_column_text(stmt, 0))
            let v = String(cString: sqlite3_column_text(stmt, 1))
            meta[k] = v
        }
        if let version = meta["schema_version"], version != expectedSchemaVersion {
            throw VectorStoreError.schemaVersionMismatch(
                found: version, expected: expectedSchemaVersion)
        }
        if let dim = meta["embedding_dim"], Int(dim) != expectedDim {
            throw VectorStoreError.dimensionMismatch(
                found: Int(dim) ?? 0, expected: expectedDim)
        }
    }

    private static func loadChunks(db: OpaquePointer) throws -> [Chunk] {
        let sql = """
            SELECT chunk_id, doc_id, start_offset, end_offset,
                   start_sent, end_sent, text, embedding, embedding_bits
            FROM chunks
            ORDER BY chunk_id
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw VectorStoreError.prepareFailed(
                message: String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }

        var chunks: [Chunk] = []
        chunks.reserveCapacity(4096)   // reasonable starting guess

        while sqlite3_step(stmt) == SQLITE_ROW {
            let chunkId = sqlite3_column_int64(stmt, 0)
            let docId = sqlite3_column_int64(stmt, 1)
            let startOffset = Int(sqlite3_column_int64(stmt, 2))
            let endOffset = Int(sqlite3_column_int64(stmt, 3))
            let startSent = Int(sqlite3_column_int64(stmt, 4))
            let endSent = Int(sqlite3_column_int64(stmt, 5))

            guard let textPtr = sqlite3_column_text(stmt, 6) else {
                throw VectorStoreError.corruptBlob(chunkId: chunkId, reason: "text is null")
            }
            let text = String(cString: textPtr)

            // Embedding (required)
            let embedBytes = sqlite3_column_bytes(stmt, 7)
            guard embedBytes == expectedDim * MemoryLayout<Float>.size else {
                throw VectorStoreError.corruptBlob(
                    chunkId: chunkId,
                    reason: "embedding is \(embedBytes) bytes, expected \(expectedDim * 4)")
            }
            guard let embedPtr = sqlite3_column_blob(stmt, 7) else {
                throw VectorStoreError.corruptBlob(chunkId: chunkId, reason: "embedding blob null")
            }
            let embedding = Array(UnsafeBufferPointer(
                start: embedPtr.assumingMemoryBound(to: Float.self),
                count: expectedDim))

            // Bits (optional — may be null for older builds)
            var bits: [UInt8] = []
            if sqlite3_column_type(stmt, 8) != SQLITE_NULL {
                let bitBytes = sqlite3_column_bytes(stmt, 8)
                if bitBytes == expectedDim / 8, let bitPtr = sqlite3_column_blob(stmt, 8) {
                    bits = Array(UnsafeBufferPointer(
                        start: bitPtr.assumingMemoryBound(to: UInt8.self),
                        count: expectedDim / 8))
                }
            }

            chunks.append(Chunk(
                id: chunkId,
                docId: docId,
                startOffset: startOffset,
                endOffset: endOffset,
                startSent: startSent,
                endSent: endSent,
                text: text,
                embedding: embedding,
                bits: bits
            ))
        }
        return chunks
    }
}

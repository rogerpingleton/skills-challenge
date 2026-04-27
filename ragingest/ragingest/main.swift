//
//  main.swift
//  ragingest
//
//  Created by Roger Pingleton on 4/19/26.
//

import Foundation
import NaturalLanguage
#if canImport(FoundationModels)
import FoundationModels
#endif

setbuf(stdout, nil)   // <-- to avoid pipe deadlock

// MARK: - Protocol
// All subcommands read one JSON object per line from stdin and emit one
// JSON object per line on stdout. This batch-mode protocol avoids process
// startup per call, which is the main performance pitfall when calling
// Swift from Python.

struct EmbedRequest: Decodable { let id: String; let text: String }
struct EmbedResponse: Encodable { let id: String; let vector: [Float] }

struct TokenRequest: Decodable { let id: String; let text: String }
struct TokenResponse: Encodable { let id: String; let tokenCount: Int }

struct SegmentRequest: Decodable { let id: String; let text: String }
struct Sentence: Encodable {
    let index: Int
    let startUTF16: Int
    let endUTF16: Int
    let text: String
}
struct SegmentResponse: Encodable { let id: String; let sentences: [Sentence] }

// MARK: - Embedding

final class Embedder {
    let embedding: NLEmbedding
    init() {
        guard let e = NLEmbedding.sentenceEmbedding(for: .english) else {
            fatalError("Sentence embedding unavailable for English")
        }
        self.embedding = e
    }
    /// Returns an L2-normalized Float32 vector.
    func embed(_ text: String) -> [Float]? {
        guard let v = embedding.vector(for: text) else { return nil }
        var f = v.map { Float($0) }
        var norm: Float = 0
        for x in f { norm += x * x }
        norm = max(norm.squareRoot(), 1e-12)
        for i in 0..<f.count { f[i] /= norm }
        return f
    }
}

// MARK: - Token count (Foundation Models on macOS 26+)

#if canImport(FoundationModels)
@available(macOS 26.0, *)
func tokenCount(for text: String) async throws -> Int {
    let model = SystemLanguageModel.default
    let prompt = Prompt(text)
    let usage = try await model.tokenCount(for: prompt)
    return usage
}
#endif

// MARK: - Sentence segmentation

func segment(_ text: String) -> [Sentence] {
    let tokenizer = NLTokenizer(unit: .sentence)
    tokenizer.setLanguage(.english)
    tokenizer.string = text
    let utf16 = text.utf16
    var out: [Sentence] = []
    var idx = 0
    tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
        let start = range.lowerBound.utf16Offset(in: text)
        let end   = range.upperBound.utf16Offset(in: text)
        let slice = String(utf16[utf16.index(utf16.startIndex, offsetBy: start)
                                ..< utf16.index(utf16.startIndex, offsetBy: end)]) ?? ""
        out.append(Sentence(index: idx, startUTF16: start, endUTF16: end, text: slice))
        idx += 1
        return true
    }
    return out
}

// MARK: - Line-oriented JSON driver

func runEmbed() {
    let embedder = Embedder()
    let encoder = JSONEncoder()
    while let line = readLine(), !line.isEmpty {
        let data = Data(line.utf8)
        guard let req = try? JSONDecoder().decode(EmbedRequest.self, from: data) else { continue }
        guard let vec = embedder.embed(req.text) else {
            // Emit a zero vector to keep IDs aligned; caller can detect norm=0.
            let resp = EmbedResponse(id: req.id, vector: Array(repeating: 0, count: 512))
            let out = try! encoder.encode(resp)
            print(String(data: out, encoding: .utf8)!)
            continue
        }
        let resp = EmbedResponse(id: req.id, vector: vec)
        let out = try! encoder.encode(resp)
        print(String(data: out, encoding: .utf8)!)
    }
}

@available(macOS 26.0, *)
func runCount() async {
    let encoder = JSONEncoder()
    while let line = readLine(), !line.isEmpty {
        let data = Data(line.utf8)
        guard let req = try? JSONDecoder().decode(TokenRequest.self, from: data) else { continue }
        let n = (try? await tokenCount(for: req.text)) ?? 0
        let resp = TokenResponse(id: req.id, tokenCount: n)
        let out = try! encoder.encode(resp)
        print(String(data: out, encoding: .utf8)!)
    }
}

func runSegment() {
    let encoder = JSONEncoder()
    while let line = readLine(), !line.isEmpty {
        let data = Data(line.utf8)
        guard let req = try? JSONDecoder().decode(SegmentRequest.self, from: data) else { continue }
        let resp = SegmentResponse(id: req.id, sentences: segment(req.text))
        let out = try! encoder.encode(resp)
        print(String(data: out, encoding: .utf8)!)
    }
}

// MARK: - Entry point

let args = CommandLine.arguments.dropFirst()
switch args.first {
case "embed":   runEmbed()
case "segment": runSegment()
case "count":
    if #available(macOS 26.0, *) {
        let sem = DispatchSemaphore(value: 0)
        Task { await runCount(); sem.signal() }
        sem.wait()
    } else {
        FileHandle.standardError.write(Data("tokenCount requires macOS 26+\n".utf8))
        exit(1)
    }
default:
    print("usage: ragtool embed|segment|count  (JSON-lines on stdin)")
    exit(2)
}

import Testing
import Foundation
@testable import AI_Skills_Challenge

@Suite("RAGSystem Tests")
struct RAGSystemTests {

    @Test("answer returns a non-empty response for 'what is RAG?'")
    @MainActor
    func answerWhatIsRAG() async throws {
        let rag = try RAGSystem()
        let result = try await rag.answer(query: "what is RAG?", maxContextTokens: 2400)
        print(result.text)

        #expect(!result.text.isEmpty, "Expected a non-empty answer")
        #expect(!result.citations.isEmpty, "Expected at least one citation")
    }
    
    @Test("answer returns a non-empty response for 'Why should I learn linear algebra as an AI Engineer?'")
    @MainActor
    func answerWhyLinearAlgebra() async throws {
        let rag = try RAGSystem()
        let result = try await rag.answer(query: "Why should I learn linear algebra as an AI Engineer?", maxContextTokens: 2400)
        print(result.text)

        #expect(!result.text.isEmpty, "Expected a non-empty answer")
        #expect(!result.citations.isEmpty, "Expected at least one citation")
    }
}

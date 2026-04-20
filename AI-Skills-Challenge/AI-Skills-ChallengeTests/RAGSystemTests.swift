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
        print("RAG Answer: \(result.text)\n\n")
        //print(result.citations[0].chunk.text)

        #expect(!result.text.isEmpty, "Expected a non-empty answer")
        #expect(!result.citations.isEmpty, "Expected at least one citation")
    }
    
    @Test("answer returns a non-empty response for 'Why should I learn linear algebra as an AI Engineer?'")
    @MainActor
    func answerWhyLinearAlgebra() async throws {
        let rag = try RAGSystem()
        let result = try await rag.answer(query: "Why should I learn linear algebra as an AI Engineer?", maxContextTokens: 2400)
        print("Why LinAlg Answer: \(result.text)\n\n")

        #expect(!result.text.isEmpty, "Expected a non-empty answer")
        #expect(!result.citations.isEmpty, "Expected at least one citation")
    }
    
    @Test("answer returns a non-empty response for 'How is Linear Algebra used in AI?'")
    @MainActor
    func answerHowLinearAlgebra() async throws {
        let rag = try RAGSystem()
        let result = try await rag.answer(query: "How is Linear Algebra used in AI?", maxContextTokens: 2400)
        print("How LinAlg Answer: \(result.text)\n\n")

        #expect(!result.text.isEmpty, "Expected a non-empty answer")
        #expect(!result.citations.isEmpty, "Expected at least one citation")
    }
    
    @Test("answer returns a non-empty response for 'Why is the sky blue?'")
    @MainActor
    func answerUnrelatedQuestion() async throws {
        let rag = try RAGSystem()
        let result = try await rag.answer(query: "Why is the sky blue?", maxContextTokens: 2400)
        print("Blue Sky Answer: \(result.text)\n\n")

        #expect(!result.text.isEmpty, "Expected a non-empty answer")
        #expect(!result.citations.isEmpty, "Expected at least one citation")
    }
}

import SwiftUI
import FoundationModels

struct ExplainView: View {
    let question: InterviewQuestion

    @State private var explanation = ""
    @State private var isLoading = false
    @State private var errorMessage: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(question.question)
                    .font(.headline)

                Divider()

                if isLoading {
                    HStack {
                        Spacer()
                        VStack(spacing: 12) {
                            ProgressView()
                                .controlSize(.large)
                            Text("Generating explanation...")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    .padding(.top, 32)
                } else if let error = errorMessage {
                    ContentUnavailableView {
                        Label("Error", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(error)
                    } actions: {
                        Button("Try Again") {
                            Task { await generateExplanation() }
                        }
                    }
                } else if !explanation.isEmpty {
                    Text(LocalizedStringKey(explanation))
                        .font(.body)
                }
            }
            .padding()
        }
        .navigationTitle("Explanation")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await generateExplanation()
        }
    }

    private func generateExplanation() async {
        isLoading = true
        errorMessage = nil
        explanation = ""

        let truncatedAnswer = String(question.answer.prefix(800))

        do {
            let model = SystemLanguageModel(guardrails: .permissiveContentTransformations)
            let session = LanguageModelSession(
                model: model,
                instructions: "Explain AI Engineering concepts simply for beginners. Use short sentences and analogies."
            )

            let response = try await session.respond(to: "Explain this simply: \(truncatedAnswer)")
            explanation = response.content
        } catch let error as LanguageModelSession.GenerationError {
            switch error {
            case .exceededContextWindowSize:
                errorMessage = "This answer is too long for the on-device model. Try a shorter question."
            case .guardrailViolation:
                errorMessage = "The model couldn't process this content due to safety restrictions."
            default:
                errorMessage = error.localizedDescription
            }
        } catch {
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }
}

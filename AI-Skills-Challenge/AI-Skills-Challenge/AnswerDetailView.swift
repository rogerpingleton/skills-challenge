import SwiftUI

struct AnswerDetailView: View {
    let question: InterviewQuestion

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(question.question)
                    .font(.headline)

                Text(question.answer)
                    .font(.body)

                NavigationLink {
                    ExplainView(question: question)
                } label: {
                    Text("Explain")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .padding()
        }
        .navigationTitle("Answer")
        .navigationBarTitleDisplayMode(.inline)
    }
}

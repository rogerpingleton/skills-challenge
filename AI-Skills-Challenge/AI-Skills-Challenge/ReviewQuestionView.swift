import SwiftUI

struct ReviewQuestionView: View {
    let questions: [InterviewQuestion]
    @State private var searchText = ""

    private var filteredQuestions: [InterviewQuestion] {
        if searchText.isEmpty {
            return questions
        }
        return questions.filter {
            $0.question.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        List(filteredQuestions) { question in
            NavigationLink(destination: AnswerDetailView(question: question)) {
                Text(question.question)
            }
        }
        .navigationTitle("Study")
        .searchable(text: $searchText, prompt: "Search questions")
    }
}

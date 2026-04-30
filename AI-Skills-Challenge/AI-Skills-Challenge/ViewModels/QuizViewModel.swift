import Foundation
import FoundationModels

@Observable
final class QuizViewModel {
    enum State: Equatable {
        case notStarted
        case answering
        case evaluating
        case reviewed
        case completed
    }

    struct QuestionResult {
        let question: InterviewQuestion
        let userAnswer: String?
        let evaluation: AnswerEvaluation?
        let passed: Bool
    }

    private(set) var state: State = .notStarted
    private(set) var results: [QuestionResult] = []
    private(set) var currentEvaluation: AnswerEvaluation?
    private(set) var errorMessage: String?

    var userAnswer: String = ""

    private var selectedQuestions: [InterviewQuestion] = []
    private var currentIndex = 0

    var currentQuestion: InterviewQuestion? {
        guard currentIndex < selectedQuestions.count else { return nil }
        return selectedQuestions[currentIndex]
    }

    var currentQuestionNumber: Int { currentIndex + 1 }
    var totalQuestions: Int { selectedQuestions.count }

    var totalScore: Int {
        results.compactMap { $0.evaluation?.score }.reduce(0, +)
    }

    var maxPossibleScore: Int {
        results.count * 10
    }

    var answeredCount: Int {
        results.filter { !$0.passed }.count
    }

    var passedCount: Int {
        results.filter { $0.passed }.count
    }

    func startQuiz(with allQuestions: [InterviewQuestion]) {
        selectedQuestions = Array(allQuestions.shuffled().prefix(10))
        currentIndex = 0
        results = []
        userAnswer = ""
        currentEvaluation = nil
        errorMessage = nil
        state = .answering
    }

    func submitAnswer() async {
        guard let question = currentQuestion else { return }
        let answer = userAnswer
        state = .evaluating
        errorMessage = nil

        do {
            let session = LanguageModelSession(instructions: """
                You are an AI engineering interview evaluator. \
                Score the user's answer against the reference answer. \
                Be fair but thorough. The topic is iOS development.
                """)

            let prompt = """
                Question: \(question.question)

                Reference answer: \(question.answer)

                User's answer: \(answer)

                Evaluate the user's answer.
                """

            let response = try await session.respond(to: prompt, generating: AnswerEvaluation.self)
            let evaluation = response.content

            currentEvaluation = evaluation
            results.append(QuestionResult(
                question: question,
                userAnswer: answer,
                evaluation: evaluation,
                passed: false
            ))
            state = .reviewed
        } catch {
            errorMessage = "Evaluation failed: \(error.localizedDescription)"
            state = .answering
        }
    }

    func passQuestion() {
        guard let question = currentQuestion else { return }
        results.append(QuestionResult(
            question: question,
            userAnswer: nil,
            evaluation: nil,
            passed: true
        ))
        advanceToNext()
    }

    func nextQuestion() {
        advanceToNext()
    }

    private func advanceToNext() {
        currentIndex += 1
        userAnswer = ""
        currentEvaluation = nil
        errorMessage = nil
        if currentIndex >= selectedQuestions.count {
            state = .completed
        } else {
            state = .answering
        }
    }

    func resetQuiz() {
        state = .notStarted
        selectedQuestions = []
        currentIndex = 0
        results = []
        userAnswer = ""
        currentEvaluation = nil
        errorMessage = nil
    }
}

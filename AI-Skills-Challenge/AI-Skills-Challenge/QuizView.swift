import SwiftUI
import FoundationModels

struct QuizView: View {
    let questions: [InterviewQuestion]
    @State private var viewModel = QuizViewModel()

    private var modelAvailability: SystemLanguageModel.Availability {
        SystemLanguageModel.default.availability
    }

    var body: some View {
        NavigationStack {
            Group {
                switch modelAvailability {
                case .available:
                    quizContent
                case .unavailable(.appleIntelligenceNotEnabled):
                    ContentUnavailableView(
                        "Apple Intelligence Required",
                        systemImage: "brain",
                        description: Text("Turn on Apple Intelligence in Settings to use the Quiz feature.")
                    )
                default:
                    ContentUnavailableView(
                        "Not Available",
                        systemImage: "exclamationmark.triangle",
                        description: Text("The on-device model is not available on this device.")
                    )
                }
            }
            .navigationTitle("Quiz")
        }
    }

    @ViewBuilder
    private var quizContent: some View {
        switch viewModel.state {
        case .notStarted:
            startView
        case .answering:
            answeringView
        case .evaluating:
            evaluatingView
        case .reviewed:
            reviewedView
        case .completed:
            completedView
        }
    }

    private var startView: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "brain.head.profile")
                .font(.system(size: 60))
                .foregroundStyle(.tint)

            Text("AI Engineering Quiz")
                .font(.largeTitle.bold())

            Text("Test your AI Engineering knowledge with 10 randomly selected questions. Your answers will be evaluated by Apple's on-device AI model, which will score each response and provide feedback.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Spacer()

            Button {
                viewModel.startQuiz(with: questions)
            } label: {
                Text("Start")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .padding(.horizontal)
            .padding(.bottom)
        }
    }

    private var answeringView: some View {
        VStack(spacing: 0) {
            ProgressView(
                value: Double(viewModel.currentQuestionNumber - 1),
                total: Double(viewModel.totalQuestions)
            )

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Question \(viewModel.currentQuestionNumber) of \(viewModel.totalQuestions)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if let question = viewModel.currentQuestion {
                        Text(question.question)
                            .font(.headline)
                    }

                    if let error = viewModel.errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }

                    TextEditor(text: $viewModel.userAnswer)
                        .frame(minHeight: 200)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(.tertiary)
                        )
                }
                .padding()
            }

            HStack(spacing: 12) {
                Button {
                    viewModel.passQuestion()
                } label: {
                    Text("Pass")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)

                Button {
                    Task {
                        await viewModel.submitAnswer()
                    }
                } label: {
                    Text("Submit")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(viewModel.userAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding()
        }
    }

    private var evaluatingView: some View {
        VStack(spacing: 16) {
            Spacer()
            ProgressView()
                .controlSize(.large)
            Text("Evaluating your answer...")
                .font(.headline)
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var reviewedView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if let evaluation = viewModel.currentEvaluation {
                    HStack {
                        Spacer()
                        Text("\(evaluation.score)/10")
                            .font(.system(size: 48, weight: .bold, design: .rounded))
                            .foregroundStyle(scoreColor(for: evaluation.score))
                        Spacer()
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Label("Critique", systemImage: "text.magnifyingglass")
                            .font(.headline)
                        Text(evaluation.critique)
                            .font(.body)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Label("Suggestions", systemImage: "lightbulb")
                            .font(.headline)
                        Text(evaluation.suggestions)
                            .font(.body)
                    }
                }

                Button {
                    viewModel.nextQuestion()
                } label: {
                    Text(viewModel.currentQuestionNumber < viewModel.totalQuestions ? "Next Question" : "See Results")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .padding()
        }
    }

    private var completedView: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("Quiz Complete!")
                    .font(.largeTitle.bold())

                Text("\(viewModel.totalScore)/\(viewModel.maxPossibleScore)")
                    .font(.system(size: 56, weight: .bold, design: .rounded))
                    .foregroundStyle(.tint)

                HStack(spacing: 24) {
                    VStack {
                        Text("\(viewModel.answeredCount)")
                            .font(.title.bold())
                        Text("Answered")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    VStack {
                        Text("\(viewModel.passedCount)")
                            .font(.title.bold())
                        Text("Passed")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Divider()

                ForEach(Array(viewModel.results.enumerated()), id: \.offset) { index, result in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Q\(index + 1)")
                                .font(.headline)
                            Spacer()
                            if result.passed {
                                Text("Passed")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            } else if let score = result.evaluation?.score {
                                Text("\(score)/10")
                                    .font(.headline)
                                    .foregroundStyle(scoreColor(for: score))
                            }
                        }
                        Text(result.question.question)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }

                Button {
                    viewModel.resetQuiz()
                } label: {
                    Text("Start Over")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .padding()
        }
    }

    private func scoreColor(for score: Int) -> Color {
        switch score {
        case 8...10: .green
        case 5...7: .orange
        default: .red
        }
    }
}

import FoundationModels

@Generable(description: "An evaluation of a user's answer to an AI Engineering interview question")
struct AnswerEvaluation {
    @Guide(description: "A brief critique noting what was correct and what was missing or incorrect")
    var critique: String

    @Guide(description: "Brief suggestions for how the user could improve their answer")
    var suggestions: String

    @Guide(description: "A score from 1 to 10", .range(1...10))
    var score: Int
}

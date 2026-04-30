import Foundation

struct InterviewQuestion: Codable, Identifiable {
    var id = UUID()
    let question: String
    let answer: String

    enum CodingKeys: String, CodingKey {
        case question
        case answer
    }
}

//
//  ChecklistCompletion.swift
//  AI-Skills-Challenge
//

import Foundation
import SwiftData

@Model
final class ChecklistCompletion {
    @Attribute(.unique) var checklistID: String
    var isComplete: Bool

    init(checklistID: String, isComplete: Bool = false) {
        self.checklistID = checklistID
        self.isComplete = isComplete
    }
}

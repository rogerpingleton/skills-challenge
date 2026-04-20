//
//  Data Models.swift
//  AI-Skills-Challenge
//
//  Created by Roger Pingleton on 4/19/26.
//

import Foundation
import SwiftUI

struct ChecklistSection: Codable, Identifiable {
    let id: String
    let title: String
    let checklists: [Checklist]
}

struct Checklist: Codable, Identifiable {
    let id: String
    let title: String
    let isComplete: Bool
}

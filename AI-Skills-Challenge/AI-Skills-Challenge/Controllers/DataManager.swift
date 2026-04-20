//
//  DataManager.swift
//  AI-Skills-Challenge
//

import Foundation

struct DataManager {
    /// Loads checklist sections from the bundled index.json file.
    static func loadChecklistSections() -> [ChecklistSection] {
        guard let url = Bundle.main.url(forResource: "index", withExtension: "json") else {
            print("DataManager: index.json not found in bundle")
            return []
        }

        do {
            let data = try Data(contentsOf: url)
            let sections = try JSONDecoder().decode([ChecklistSection].self, from: data)
            return sections
        } catch {
            print("DataManager: Failed to decode index.json - \(error)")
            return []
        }
    }
}

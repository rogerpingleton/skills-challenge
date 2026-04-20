//
//  DataManagerTests.swift
//  AI-Skills-ChallengeTests
//

import Foundation
import UIKit
import Testing
@testable import AI_Skills_Challenge

struct DataManagerTests {

    // MARK: - DataManager Loading

    @Test("DataManager loads sections from index.json")
    func loadChecklistSectionsReturnsData() {
        let sections = DataManager.loadChecklistSections()
        #expect(!sections.isEmpty)
    }

    @Test("DataManager loads the expected number of sections")
    func loadChecklistSectionsCount() {
        let sections = DataManager.loadChecklistSections()
        #expect(sections.count == 14)
    }

    @Test("First section has expected title")
    func firstSectionTitle() {
        let sections = DataManager.loadChecklistSections()
        #expect(sections.first?.title == "TECHNICAL FOUNDATION")
    }

    @Test("Last section has expected title")
    func lastSectionTitle() {
        let sections = DataManager.loadChecklistSections()
        #expect(sections.last?.title == "SECURITY, PRIVACY, ETHICS")
    }

    // MARK: - Section Integrity

    @Test("Every section has a non-empty id")
    func allSectionsHaveIds() {
        let sections = DataManager.loadChecklistSections()
        for section in sections {
            #expect(!section.id.isEmpty)
        }
    }

    @Test("Every section has a non-empty title")
    func allSectionsHaveTitles() {
        let sections = DataManager.loadChecklistSections()
        for section in sections {
            #expect(!section.title.isEmpty)
        }
    }

    @Test("Every section contains at least one checklist item")
    func allSectionsHaveChecklists() {
        let sections = DataManager.loadChecklistSections()
        for section in sections {
            #expect(!section.checklists.isEmpty, "Section '\(section.title)' has no checklists")
        }
    }

    @Test("All section ids are unique")
    func sectionIdsAreUnique() {
        let sections = DataManager.loadChecklistSections()
        let ids = sections.map(\.id)
        #expect(Set(ids).count == ids.count)
    }

    // MARK: - Checklist Item Integrity

    @Test("Every checklist item has a non-empty id")
    func allChecklistItemsHaveIds() {
        let sections = DataManager.loadChecklistSections()
        let allItems = sections.flatMap(\.checklists)
        for item in allItems {
            #expect(!item.id.isEmpty)
        }
    }

    @Test("Every checklist item has a non-empty title")
    func allChecklistItemsHaveTitles() {
        let sections = DataManager.loadChecklistSections()
        let allItems = sections.flatMap(\.checklists)
        for item in allItems {
            #expect(!item.title.isEmpty)
        }
    }

    @Test("All checklist item ids are unique across all sections")
    func checklistItemIdsAreGloballyUnique() {
        let sections = DataManager.loadChecklistSections()
        let allIds = sections.flatMap(\.checklists).map(\.id)
        #expect(Set(allIds).count == allIds.count)
    }

    // MARK: - JSON Decoding

    @MainActor
    @Test("ChecklistSection decodes from valid JSON")
    func decodeSectionFromJSON() throws {
        let json = """
        {
            "id": "test-id",
            "title": "TEST SECTION",
            "checklists": [
                { "id": "item-1", "title": "First item", "isComplete": false }
            ]
        }
        """
        let data = Data(json.utf8)
        let section = try JSONDecoder().decode(ChecklistSection.self, from: data)
        #expect(section.id == "test-id")
        #expect(section.title == "TEST SECTION")
        #expect(section.checklists.count == 1)
    }

    @MainActor
    @Test("Checklist decodes from valid JSON")
    func decodeChecklistFromJSON() throws {
        let json = """
        { "id": "item-1", "title": "Sample item", "isComplete": true }
        """
        let data = Data(json.utf8)
        let item = try JSONDecoder().decode(Checklist.self, from: data)
        #expect(item.id == "item-1")
        #expect(item.title == "Sample item")
        #expect(item.isComplete == true)
    }

    @MainActor
    @Test("Checklist decodes isComplete false correctly")
    func decodeChecklistNotComplete() throws {
        let json = """
        { "id": "item-2", "title": "Incomplete item", "isComplete": false }
        """
        let data = Data(json.utf8)
        let item = try JSONDecoder().decode(Checklist.self, from: data)
        #expect(item.isComplete == false)
    }

    @Test("Decoding fails for missing required fields")
    func decodeFailsForInvalidJSON() {
        let json = """
        { "id": "missing-title" }
        """
        let data = Data(json.utf8)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(Checklist.self, from: data)
        }
    }

    // MARK: - Asset Integrity

    @Test("Every section title has a corresponding image in Assets")
    func allSectionTitlesHaveImages() {
        let sections = DataManager.loadChecklistSections()
        for section in sections {
            #expect(
                UIImage(named: section.title) != nil,
                "Missing image asset for section '\(section.title)'"
            )
        }
    }
}

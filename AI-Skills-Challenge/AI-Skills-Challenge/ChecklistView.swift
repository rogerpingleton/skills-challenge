//
//  ChecklistView.swift
//  AI-Skills-Challenge
//

import SwiftUI

struct ChecklistView: View {
    let section: ChecklistSection

    var body: some View {
        List(section.checklists) { item in
            HStack {
                Image(systemName: item.isComplete ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(item.isComplete ? .green : .secondary)
                Text(item.title)
            }
        }
        .navigationTitle(section.title)
    }
}

#Preview {
    NavigationStack {
        ChecklistView(
            section: ChecklistSection(
                id: "preview",
                title: "SAMPLE SECTION",
                checklists: [
                    Checklist(id: "1", title: "First item", isComplete: false),
                    Checklist(id: "2", title: "Second item", isComplete: true)
                ]
            )
        )
    }
}

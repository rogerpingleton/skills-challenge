//
//  ChecklistView.swift
//  AI-Skills-Challenge
//

import SwiftUI

struct ChecklistView: View {
    let section: ChecklistSection

    var body: some View {
        List {
            Image(section.title)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: .infinity)
                .listRowInsets(EdgeInsets())
                .listRowSeparator(.hidden)

            Section {
                ForEach(section.checklists) { item in
                    HStack {
                        Image(systemName: item.isComplete ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(item.isComplete ? .green : .secondary)
                        Text(item.title)
                    }
                }
            } header: {
                Text(section.title)
                    .font(.largeTitle.bold())
                    .foregroundStyle(.primary)
                    .textCase(nil)
            }
        }
        .listStyle(.plain)
        .ignoresSafeArea(edges: .top)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
    }
}

#Preview {
    NavigationStack {
        ChecklistView(
            section: ChecklistSection(
                id: "preview",
                title: "TECHNICAL FOUNDATION",
                checklists: [
                    Checklist(id: "1", title: "First item", isComplete: false),
                    Checklist(id: "2", title: "Second item", isComplete: true)
                ]
            )
        )
    }
}

//
//  ChecklistView.swift
//  AI-Skills-Challenge
//

import SwiftUI
import SwiftData

struct ChecklistView: View {
    let section: ChecklistSection
    @Query private var completions: [ChecklistCompletion]
    @Environment(\.modelContext) private var modelContext

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
                        Button {
                            toggleCompletion(for: item)
                        } label: {
                            Image(systemName: isComplete(item) ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(isComplete(item) ? Color("AccentColor"): .secondary)
                        }
                        .buttonStyle(.plain)
                        NavigationLink(destination: StudySkillsView(itemTitle: item.title, sectionTitle: section.title)) {
                            Text(item.title)
                        }
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

    private func isComplete(_ item: Checklist) -> Bool {
        completions.first { $0.checklistID == item.id }?.isComplete ?? false
    }

    private func toggleCompletion(for item: Checklist) {
        if let existing = completions.first(where: { $0.checklistID == item.id }) {
            existing.isComplete.toggle()
        } else {
            let completion = ChecklistCompletion(checklistID: item.id, isComplete: true)
            modelContext.insert(completion)
        }
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
    .modelContainer(for: ChecklistCompletion.self, inMemory: true)
}

//
//  ChecklistSectionView.swift
//  AI-Skills-Challenge
//

import SwiftUI

struct ChecklistSectionView: View {
    let sections: [ChecklistSection]

    init() {
        self.sections = DataManager.loadChecklistSections()
    }

    var body: some View {
        NavigationStack {
            List(sections) { section in
                NavigationLink(destination: ChecklistView(section: section)) {
                    Text(section.title)
                }
            }
            .navigationTitle("AI Skills Challenge")
        }
    }
}

#Preview {
    ChecklistSectionView()
}

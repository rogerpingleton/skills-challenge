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
            List {
                Image("Main1")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .listRowInsets(EdgeInsets())
                    .listRowSeparator(.hidden)

                Section {
                    ForEach(sections) { section in
                        NavigationLink(destination: ChecklistView(section: section)) {
                            Text(section.title)
                        }
                    }
                } header: {
                    Text("AI Skills Checklist")
                        .font(.largeTitle.bold())
                        .foregroundStyle(Color("AccentColor"))
                        .textCase(nil)
                }
            }
            .listStyle(.plain)
            .ignoresSafeArea(edges: .top)
            .toolbar(.hidden, for: .navigationBar)
        }
    }
}

#Preview {
    ChecklistSectionView()
}

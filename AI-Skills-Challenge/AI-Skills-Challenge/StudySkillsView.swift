//
//  StudySkillsView.swift
//  AI-Skills-Challenge
//

import SwiftUI
import MarkdownView

struct StudySkillsView: View {
    let itemTitle: String
    let sectionTitle: String

    private var markdownContent: String {
        guard let url = Bundle.main.url(forResource: itemTitle, withExtension: "md"),
              let content = try? String(contentsOf: url, encoding: .utf8) else {
            return "Content not available."
        }
        return content.markdownWithTablesConverted
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                Image(sectionTitle)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(maxWidth: .infinity)
                    .frame(height: 250)
                    .clipped()

                MarkdownView(markdownContent)
                    .padding()
            }
        }
        .ignoresSafeArea(edges: .top)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
    }
}

#Preview {
    NavigationStack {
        StudySkillsView(
            itemTitle: "Attention mechanism",
            sectionTitle: "FOUNDATION MODELS AND MODEL SELECTION"
        )
    }
}

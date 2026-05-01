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
        return convertTablesToLists(content)
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

    // Converts markdown tables to lists to work around a crash
    // in MarkdownView's table coordinate space handling on iOS 26.
    private func convertTablesToLists(_ markdown: String) -> String {
        let lines = markdown.components(separatedBy: "\n")
        var result: [String] = []
        var i = 0

        while i < lines.count {
            let trimmed = lines[i].trimmingCharacters(in: .whitespaces)

            guard trimmed.hasPrefix("|") && trimmed.hasSuffix("|") else {
                result.append(lines[i])
                i += 1
                continue
            }

            var tableLines: [String] = []
            while i < lines.count {
                let t = lines[i].trimmingCharacters(in: .whitespaces)
                guard t.hasPrefix("|") && t.hasSuffix("|") else { break }
                tableLines.append(t)
                i += 1
            }

            guard tableLines.count >= 2 else {
                result.append(contentsOf: tableLines)
                continue
            }

            let headers = parseCells(tableLines[0])

            let separatorIndex = tableLines[1].trimmingCharacters(in: .whitespaces)
                .contains(where: { $0 == "-" }) ? 1 : -1
            let dataStart = separatorIndex >= 0 ? 2 : 1

            result.append("**\(headers.joined(separator: " | "))**")
            result.append("")

            for row in tableLines[dataStart...] {
                let cells = parseCells(row)
                let parts = zip(headers, cells).map { "\($0): \($1)" }
                result.append("- \(parts.joined(separator: " | "))")
            }

            result.append("")
        }

        return result.joined(separator: "\n")
    }

    private func parseCells(_ line: String) -> [String] {
        line
            .trimmingCharacters(in: .whitespaces)
            .dropFirst()
            .dropLast()
            .components(separatedBy: "|")
            .map { $0.trimmingCharacters(in: .whitespaces) }
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

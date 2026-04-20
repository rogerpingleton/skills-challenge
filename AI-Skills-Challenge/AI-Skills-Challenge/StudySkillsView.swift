//
//  StudySkillsView.swift
//  AI-Skills-Challenge
//

import SwiftUI

struct StudySkillsView: View {
    let itemTitle: String
    let sectionTitle: String

    private var markdownContent: String {
        guard let url = Bundle.main.url(forResource: itemTitle, withExtension: "md"),
              let content = try? String(contentsOf: url, encoding: .utf8) else {
            return "Content not available."
        }
        return content
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

                VStack(alignment: .leading, spacing: 12) {
                    ForEach(Array(parseBlocks().enumerated()), id: \.offset) { _, block in
                        renderBlock(block)
                    }
                }
                .padding()
            }
        }
        .ignoresSafeArea(edges: .top)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
    }

    // MARK: - Markdown Block Types

    private enum Block {
        case heading1(String)
        case heading2(String)
        case heading3(String)
        case paragraph(String)
        case codeBlock(String)
        case horizontalRule
        case unorderedListItem(String)
        case orderedListItem(String)
    }

    // MARK: - Parser

    private func parseBlocks() -> [Block] {
        var blocks: [Block] = []
        let lines = markdownContent.components(separatedBy: "\n")
        var i = 0
        var paragraphLines: [String] = []

        func flushParagraph() {
            let text = paragraphLines.joined(separator: " ").trimmingCharacters(in: .whitespaces)
            if !text.isEmpty {
                blocks.append(.paragraph(text))
            }
            paragraphLines = []
        }

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Code block
            if trimmed.hasPrefix("```") {
                flushParagraph()
                var codeLines: [String] = []
                i += 1
                while i < lines.count && !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    codeLines.append(lines[i])
                    i += 1
                }
                blocks.append(.codeBlock(codeLines.joined(separator: "\n")))
                i += 1
                continue
            }

            // Horizontal rule
            if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                flushParagraph()
                blocks.append(.horizontalRule)
                i += 1
                continue
            }

            // Headings (check ### before ## before #)
            if trimmed.hasPrefix("### ") {
                flushParagraph()
                blocks.append(.heading3(String(trimmed.dropFirst(4))))
                i += 1
                continue
            }
            if trimmed.hasPrefix("## ") {
                flushParagraph()
                blocks.append(.heading2(String(trimmed.dropFirst(3))))
                i += 1
                continue
            }
            if trimmed.hasPrefix("# ") {
                flushParagraph()
                blocks.append(.heading1(String(trimmed.dropFirst(2))))
                i += 1
                continue
            }

            // Unordered list items
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") {
                flushParagraph()
                blocks.append(.unorderedListItem(String(trimmed.dropFirst(2))))
                i += 1
                continue
            }

            // Ordered list items (e.g., "1. text")
            let numberPrefix = trimmed.prefix(while: { $0.isNumber })
            if !numberPrefix.isEmpty,
               trimmed.dropFirst(numberPrefix.count).hasPrefix(". ") {
                flushParagraph()
                let content = String(trimmed.dropFirst(numberPrefix.count + 2))
                blocks.append(.orderedListItem(content))
                i += 1
                continue
            }

            // Empty line = paragraph break
            if trimmed.isEmpty {
                flushParagraph()
                i += 1
                continue
            }

            // Regular text line
            paragraphLines.append(trimmed)
            i += 1
        }

        flushParagraph()
        return blocks
    }

    // MARK: - Rendering

    @ViewBuilder
    private func renderBlock(_ block: Block) -> some View {
        switch block {
        case .heading1(let text):
            Text(text)
                .font(.title.bold())
                .padding(.top, 8)
        case .heading2(let text):
            Text(text)
                .font(.title2.bold())
                .padding(.top, 6)
        case .heading3(let text):
            Text(text)
                .font(.title3.bold())
                .padding(.top, 4)
        case .paragraph(let text):
            inlineMarkdown(text)
        case .codeBlock(let code):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .padding(12)
            }
            .background(Color(.systemGray6))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        case .horizontalRule:
            Divider()
                .padding(.vertical, 4)
        case .unorderedListItem(let text):
            HStack(alignment: .top, spacing: 8) {
                Text("\u{2022}")
                inlineMarkdown(text)
            }
        case .orderedListItem(let text):
            inlineMarkdown(text)
        }
    }

    @ViewBuilder
    private func inlineMarkdown(_ text: String) -> some View {
        if let attributed = try? AttributedString(markdown: text) {
            Text(attributed)
                .font(.body)
        } else {
            Text(text)
                .font(.body)
        }
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

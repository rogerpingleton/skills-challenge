//
//  MarkdownTableConverter.swift
//  AI-Skills-Challenge
//

import Foundation

// Converts markdown tables to lists to work around a crash
// in MarkdownView's table coordinate space handling on iOS 26.
extension String {
    var markdownWithTablesConverted: String {
        let lines = components(separatedBy: "\n")
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

            let headers = Self.parseTableCells(tableLines[0])

            let hasSeparator = tableLines[1].contains(where: { $0 == "-" })
            let dataStart = hasSeparator ? 2 : 1

            result.append("**\(headers.joined(separator: " | "))**")
            result.append("")

            for row in tableLines[dataStart...] {
                let cells = Self.parseTableCells(row)
                let parts = zip(headers, cells).map { "\($0): \($1)" }
                result.append("- \(parts.joined(separator: " | "))")
            }

            result.append("")
        }

        return result.joined(separator: "\n")
    }

    private static func parseTableCells(_ line: String) -> [String] {
        line
            .trimmingCharacters(in: .whitespaces)
            .dropFirst()
            .dropLast()
            .components(separatedBy: "|")
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }
}

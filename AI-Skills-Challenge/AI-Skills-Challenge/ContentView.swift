//
//  ContentView.swift
//  AI-Skills-Challenge
//
//  Created by Roger Pingleton on 4/15/26.
//

import SwiftUI

struct ContentView: View {
    let questions: [InterviewQuestion] = {
        guard let url = Bundle.main.url(forResource: "ai_engineering_qna", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([InterviewQuestion].self, from: data) else {
            return []
        }
        return decoded
    }()
    
    var body: some View {
        TabView {
            Tab("Study", systemImage: "book.fill") {
                ChecklistSectionView()
            }

            Tab("Chat", systemImage: "bubble.left.and.bubble.right.fill") {
                ChatView()
            }
            
            Tab("Quiz", systemImage: "brain.head.profile") {
                QuizView(questions: questions)
            }

            Tab("Info", systemImage: "info.circle.fill") {
                InfoView()
            }
        }
    }
}

#Preview {
    ContentView()
}

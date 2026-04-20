//
//  ContentView.swift
//  AI-Skills-Challenge
//
//  Created by Roger Pingleton on 4/15/26.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            Tab("Study", systemImage: "book.fill") {
                ChecklistSectionView()
            }

            Tab("Chat", systemImage: "bubble.left.and.bubble.right.fill") {
                ChatView()
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

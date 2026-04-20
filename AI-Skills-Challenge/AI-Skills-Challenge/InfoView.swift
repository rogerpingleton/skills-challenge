//
//  InfoView.swift
//  AI-Skills-Challenge
//

import SwiftUI

struct InfoView: View {
    var body: some View {
        NavigationStack {
            List {
                Section("About") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("AI Skills Challenge")
                            .font(.headline)
                        Text("A learning companion to help you develop and track your AI skills. Work through checklists covering key AI topics, and chat with an on-device AI assistant to deepen your understanding. Based off of the _AI Engineering Skills Checklist_ by Marina Wyss, part of the [AI/ML Career Launchpad](https://aiml-career-launchpad.circle.so/), used with permission.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }

                Section("Features") {
                    Label("Study checklists for AI skill areas", systemImage: "checklist")
                    Label("Track your learning progress", systemImage: "chart.bar.fill")
                    Label("Chat with an on-device AI assistant", systemImage: "bubble.left.and.bubble.right.fill")
                }

                Section("Technology") {
                    Label("Built with SwiftUI", systemImage: "swift")
                    Label("Powered by Apple Intelligence", systemImage: "brain")
                    Label("Data stored with SwiftData", systemImage: "externaldrive.fill")
                }

                Section("Credits") {
                    Label {
                        Text("Developed by Roger Pingleton for [StreamSide AI](https://www.streamside.ai).")
                    } icon: {
                        Image("StreamSideAILogo")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 40, height: 40)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
            .navigationTitle("Info")
        }
    }
}

#Preview {
    InfoView()
}

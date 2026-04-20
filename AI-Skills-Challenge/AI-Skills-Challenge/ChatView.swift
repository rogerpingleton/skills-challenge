//
//  ChatView.swift
//  AI-Skills-Challenge
//

import SwiftUI
import FoundationModels

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var text: String

    enum Role {
        case user
        case assistant
    }
}

struct ChatView: View {
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var isGenerating = false
    @State private var session = LanguageModelSession(
        instructions: "You are a helpful AI assistant for the AI Skills Challenge app. Help users learn about AI concepts and skills. Keep responses concise and informative."
    )
    private var model = SystemLanguageModel.default

    var body: some View {
        NavigationStack {
            switch model.availability {
            case .available:
                chatContent
                    .navigationTitle("Chat")
            case .unavailable(.deviceNotEligible):
                unavailableView(message: "This device does not support Apple Intelligence.")
            case .unavailable(.appleIntelligenceNotEnabled):
                unavailableView(message: "Please enable Apple Intelligence in Settings to use this feature.")
            case .unavailable(.modelNotReady):
                unavailableView(message: "The AI model is still downloading. Please try again later.")
            case .unavailable:
                unavailableView(message: "The AI model is currently unavailable.")
            }
        }
    }

    private var chatContent: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: messages.count) {
                    if let lastMessage = messages.last {
                        withAnimation {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
            }

            Divider()

            HStack {
                TextField("Ask a question...", text: $inputText)
                    .textFieldStyle(.roundedBorder)
                    .disabled(isGenerating)
                    .onSubmit {
                        sendMessage()
                    }

                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || isGenerating)
            }
            .padding()
        }
    }

    private func unavailableView(message: String) -> some View {
        ContentUnavailableView {
            Label("AI Unavailable", systemImage: "brain")
        } description: {
            Text(message)
        }
    }

    private func sendMessage() {
        let userText = inputText.trimmingCharacters(in: .whitespaces)
        guard !userText.isEmpty else { return }
        inputText = ""

        messages.append(ChatMessage(role: .user, text: userText))
        messages.append(ChatMessage(role: .assistant, text: ""))
        isGenerating = true

        Task {
            do {
                let stream = session.streamResponse(to: userText)
                for try await snapshot in stream {
                    messages[messages.count - 1].text = snapshot.content
                }
            } catch {
                messages[messages.count - 1].text = "Sorry, an error occurred: \(error.localizedDescription)"
            }
            isGenerating = false
        }
    }
}

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer() }

            Text(message.text)
                .padding(12)
                .background(message.role == .user ? Color.blue : Color(.systemGray5))
                .foregroundStyle(message.role == .user ? .white : .primary)
                .clipShape(RoundedRectangle(cornerRadius: 16))

            if message.role == .assistant { Spacer() }
        }
    }
}

#Preview {
    ChatView()
}

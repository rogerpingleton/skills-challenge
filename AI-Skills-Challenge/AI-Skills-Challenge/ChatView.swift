//
//  ChatView.swift
//  AI-Skills-Challenge
//

import SwiftUI
import NaturalLanguage
import FoundationModels
import MarkdownView

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
    @State private var ragSystem: RAGSystem?
    @State private var initError: String?
    @State private var tokenCount = 0

    private let model = SystemLanguageModel.default
    private let tokenLimit = 100

    var body: some View {
        NavigationStack {
            switch model.availability {
            case .available:
                chatContent
                    .navigationTitle("Chat")
                    .task {
                        do {
                            ragSystem = try RAGSystem()
                        } catch {
                            initError = error.localizedDescription
                        }
                    }
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
            if let error = initError {
                ContentUnavailableView {
                    Label("RAG System Error", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(error)
                }
            } else {
                messageList
                Divider()
                inputArea
            }
        }
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(messages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }
                    if isGenerating {
                        HStack {
                            ProgressView()
                                .padding(12)
                                .background(Color(.systemGray5))
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                            Spacer()
                        }
                        .id("loading")
                    }
                }
                .padding()
            }
            .onChange(of: messages.count) {
                scrollToBottom(proxy: proxy)
            }
            .onChange(of: isGenerating) {
                scrollToBottom(proxy: proxy)
            }
        }
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        withAnimation {
            if isGenerating {
                proxy.scrollTo("loading", anchor: .bottom)
            } else if let lastMessage = messages.last {
                proxy.scrollTo(lastMessage.id, anchor: .bottom)
            }
        }
    }

    private var inputArea: some View {
        VStack(spacing: 4) {
            HStack {
                TextField("Ask a question...", text: $inputText)
                    .textFieldStyle(.roundedBorder)
                    .disabled(isGenerating)
                    .onSubmit {
                        sendMessage()
                    }
                    .onChange(of: inputText) { oldValue, newValue in
                        let count = countTokens(in: newValue)
                        if count >= tokenLimit && newValue.count > oldValue.count {
                            inputText = oldValue
                        } else {
                            tokenCount = count
                        }
                    }

                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || isGenerating)
            }

            if !inputText.isEmpty {
                HStack {
                    Text("\(tokenCount) / \(tokenLimit) tokens")
                        .font(.caption)
                        .foregroundStyle(tokenCount >= tokenLimit ? .red : .secondary)
                    Spacer()
                }
            }
        }
        .padding()
    }

    private func unavailableView(message: String) -> some View {
        ContentUnavailableView {
            Label("AI Unavailable", systemImage: "brain")
        } description: {
            Text(message)
        }
    }

    private func countTokens(in text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        return tokenizer.tokens(for: text.startIndex..<text.endIndex).count
    }

    private func sendMessage() {
        let userText = inputText.trimmingCharacters(in: .whitespaces)
        guard !userText.isEmpty, let rag = ragSystem else { return }
        inputText = ""
        tokenCount = 0

        messages.append(ChatMessage(role: .user, text: userText))
        isGenerating = true

        Task {
            do {
                let answer = try await rag.answer(query: userText)
                messages.append(ChatMessage(role: .assistant, text: answer.text))
            } catch {
                messages.append(ChatMessage(role: .assistant, text: "Sorry, an error occurred: \(error.localizedDescription)"))
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

            MarkdownView(message.text.markdownWithTablesConverted)
                .padding(12)
                .background(message.role == .user ? Color("AccentColor") : Color(.systemGray5))
                .foregroundStyle(message.role == .user ? .white : .primary)
                .clipShape(RoundedRectangle(cornerRadius: 16))

            if message.role == .assistant { Spacer() }
        }
    }
}

#Preview {
    ChatView()
}

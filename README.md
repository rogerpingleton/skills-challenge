# skills-challenge
A skills acquisition app which uses Apple's Foundation models to generate questions and assess answers.

## Disclaimer:

This app was originally inspired by Marina Wyss' AI Engineering Skills Checklist which is part of her AI/ML Career Launchpad system (https://aiml-career-launchpad.circle.so) and study techniques presented in her videos. The data however is decoupled, so that it can be generalized for other subject matter.

The app was also inspired by my work developing the OSHA-NIOSH Small Business Checklist while working as a contractor with Maximus for the CDC/NIOSH.

The suggested generated subject matter is stored in a separate repository: https://github.com/rogerpingleton/AI-ML-Engineering-Skills-Checklist

## Features:

### Stage 1
- A micro-RAG system which assesses answers and compares data via an on-device embedding model and on-device vector database
- Ability to question the database of training material (with limited context)

### Stage 2/3
- Ability to allow the user to be quized on specific topics.
- All answers are essay and judged on-device by Apple FM, which incorporates teach to learn mechanic
- Users can study the question/answer set ahead of time
- If a user doesn't understand a question, they can ask for an AI-generated simplification
- If simplification is still too complex, the user can tap "Retry" to generate a new simplification

### Stage 3 (current stage)
- Fallback system to meet Apple's requirements for Apple Intelligence

## Challenges:

- **Context window size**: Currently Apple's context window size is 4,096 tokens.
- **Privacy**: To maintain privacy, ideally all operations should be performed on the device.
- **Vector Similarity Testing**: All vector similarity operations need to be done in memory for efficiency purposes.
- **Embedded Material**: All training material must be preloaded and maintained along with its associated embeddings. The material is fixed at release. Maintaining a single source of truth for the subject matter is critical.
- **Prompt Engineering**: To be an effective educational tool, prompts must created which: 1) create pertinent questions 2) accurately evaluate answers 3) encourage users without being overly sycophantic. System prompts for this last goal are super important.
- **Guardrails**: Foundation model guardrails must be thoroughly tested systematically and repeatedly during testing to maintain accuracy and quality. (A frontier model will be used as a judge during development, but not production in order to maintain privacy.)
- **Observability and Feedback**: Without sacrificing a user's privacy, the user should be able to annonymously report content assessment failures in order to improve the system. Observability must protect the privacy of the users and comply with privacy laws.

## Project Layout:

```
├── AI-Skills-Challenge
│   ├── AI-Skills-Challenge
|   ├── ... # skills challenge app Xcode files
│   ├── AI-Skills-ChallengeTests
│   │   ├── AI_Skills_ChallengeTests.swift
│   │   ├── DataManagerTests.swift
│   │   ├── DeterminismTests.swift
│   │   └── RAGSystemTests.swift
│   └── rag-eval
│       └── main.swift # commandline rag evaluation tool
├── data_cleanup_helpers
│   ├── fix_md_heading.py      # makes the top heading the same as filename
│   ├── md_to_json.py.         # converts markdown to json
│   └── remove_code_blocks.py  # removes code blocks from md files
├── ingestion
│   └── ingest.py              # ingests data into sqlite db
├── evaluation
│   ├── .env.example              # .env example for judge.py
│   ├── ai_engineering_eval.json  # golden dataset for evaluation
│   ├── judge.py                  # judges results from evaluation 
│   ├── run_rag_eval.py           # runs a RAG evaluation
│   └── run_raw_eval.py           # runs a raw evaluation (no RAG)
├── LICENSE
├── Project_Document.pdf
├── ragingest                  # Swift command line ingest helper
│   ├── ragingest
│   │   └── main.swift
│   └── ragingest.xcodeproj
│       ├── project.pbxproj
│       ├── project.xcworkspace
│       │   ├── contents.xcworkspacedata
│       │   ├── xcshareddata
│       │   │   └── swiftpm
│       │   │       └── configuration
│       │   └── xcuserdata
│       │       └── jennywilder.xcuserdatad
│       │           └── UserInterfaceState.xcuserstate
│       └── xcuserdata
└── README.md
```

## Installation:
```
git clone https://github.com/rogerpingleton/skills-challenge.git
```

### Swift Package Dependencies

After cloning, open the Xcode project and add the following Swift Package dependency:

1. In Xcode, go to **File > Add Package Dependencies...**
2. Enter the package URL: `https://github.com/LiYanan2004/MarkdownView`
3. Set the dependency rule to **Up to Next Major Version** with minimum version **2.6.1**
4. Click **Add Package**, ensure **MarkdownView** is selected and the target is **AI-Skills-Challenge**, then click **Add Package**

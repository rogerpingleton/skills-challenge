# skills-challenge
A skills acquisition app which uses Apple's Foundation models to generate questions and assess answers.

## Disclaimer:

This app was originally inspired by Marina Wyss' AI Engineering Skills Checklist which is part of her AI/ML Career Launchpad system (https://aiml-career-launchpad.circle.so) and study techniques presented in her videos. The data however is decoupled, so that it can be generalized for other subject matter.

The app was also inspired by my work developing the OSHA-NIOSH Small Business Checklist while working as a contractor with Maximus for the CDC/NIOSH.

The suggested generated subject matter is stored in a separate repository: https://github.com/rogerpingleton/AI-ML-Engineering-Skills-Checklist

## Features:
- Either a micro-RAG system which assesses answers and compares data via an on-device embedding model and on-device vector database

AND/OR

- An adapter trained on the specific educational material

- Ability to question the database of training material (with limited context)
- Ability to let the user create their own explanation of what the subject matter is.

## Challenges:

- **Context window size**: Currently Apple's context window size is 4,096 tokens.
- **Privacy**: To maintain privacy, ideally all operations should be performed on the device.
- **Vector Similarity Testing**: All vector similarity operations need to be done in memory for efficiency purposes.
- **Embedded Material**: All training material must be preloaded and maintained along with its associated embeddings. The material is fixed at release. Maintaining a single source of truth for the subject matter is critical.
- **Prompt Engineering**: To be an effective educational tool, prompts must created which: 1) create pertinent questions 2) accurately evaluate answers 3) encourage users without being overly sycophantic. System prompts for this last goal are super important.
- **Guardrails**: Foundation model guardrails must be thoroughly tested systematically and repeatedly during testing to maintain accuracy and quality. (A frontier model will be used as a judge during development, but not production in order to maintain privacy.)
- **Observability and Feedback**: Without sacrificing a user's privacy, the user should be able to annonymously report content assessment failures in order to improve the system. Observability must protect the privacy of the users and comply with privacy laws.

## Project Layout:

TBD

## Installation:

TBD



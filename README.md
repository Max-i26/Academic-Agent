# Academic-Agent

A locally hosted, domain-specific AI agent designed to help university
students understand academic materials through grounded question answering.

## Project Vision

Academic-Agent aims to transform lecture materials into an interactive
academic knowledge assistant that can:

- Answer questions using university lecture materials.
- Retrieve relevant information from source documents.
- Provide page-level references for answers.
- Explain complex academic concepts.
- Generate quizzes and revision exercises.
- Support cross-topic reasoning.

## Architecture

The system is designed with modular components:

- Document ingestion
- Text processing and chunking
- Information retrieval
- Language model integration
- Agent orchestration
- API and user interface

## Technology Stack

- Python 3.11+
- PyMuPDF
- Pydantic
- Pytest

Additional retrieval and local language model components will be
introduced as development progresses.

## Development Status

<escape>🚧</escape> Early development — PDF extraction module implemented
with automated tests.

## Installation

Clone the repository:

```bash
git clone https://github.com/Max-i26/Academic-Agent.git
cd Academic-Agent

# Create virtual environment

python3.11 -m venv .venv
source .venv/bin/activate

# Install the project

python -m pip install -e ".[dev]"

#Run tests

python -m pytest

# Project Structure

Academic-Agent/
├── src/
│   └── academic_agent/
│       ├── agent/
│       ├── ingestion/
│       ├── llm/
│       ├── retrieval/
│       └── utils/
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── knowledge_base/
├── pyproject.toml
├── requirements.txt
└── README.md

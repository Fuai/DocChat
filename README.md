# RAG Chatbot

A simple document-based chatbot using RAG (Retrieval-Augmented Generation) with LangChain and OpenAI.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

3. Run:
```bash
streamlit run app.py
```

## Features
- Supports PDF, Word, Excel documents
- Real-time chat interface
- Document-based Q&A 
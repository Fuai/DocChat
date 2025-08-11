# Document Q&A Chatbot

A Gradio-based chatbot that can answer questions about uploaded documents using Hugging Face's models.

## Features
- Upload and process PDF, Word, and Excel documents
- Chat interface for asking questions about the documents
- Uses OpenAI-compatible API with Hugging Face models
- Fast document processing with optimized embeddings
- Local caching for better performance

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements_gradio.txt
```

3. Create a `.env` file with your Hugging Face token:
```
HF_TOKEN=your_huggingface_token
```

4. Run the app:
```bash
python app_gradio.py
```

5. Open http://127.0.0.1:7860 in your browser

## Configuration

- Default model: `openai/gpt-oss-20b`
- You can change models by setting `HF_MODEL` environment variable
- Supports PDF, DOCX, and Excel files
- Uses local caching for faster processing

## Requirements
- Python 3.8+
- See requirements_gradio.txt for full list
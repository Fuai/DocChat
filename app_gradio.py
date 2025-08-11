import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from openai import OpenAI

"""
Simplified Doc Chatbot using Hugging Face Inference API only.
Requires env var HUGGINGFACEHUB_API_TOKEN and optional HF_MODEL.
"""

# Initialize environment variables if needed
load_dotenv()

class DocumentChat:
    def __init__(self):
        self.vectorstore = None
        self.chat_history = []
        
        # Load environment variables
        load_dotenv()
        
        # Model configuration
        self.model_name = os.getenv("HF_MODEL", "openai/gpt-oss-20b")
        self.api_token = os.getenv("HF_TOKEN")
        if not self.api_token:
            raise ValueError("Please set HF_TOKEN in your .env file")
        
        # Initialize OpenAI client with HF endpoint
        print(f"Using model {self.model_name} via Hugging Face API...")
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.api_token
        )
        
        # Test API connection
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}]
            )
            print("Successfully connected to Hugging Face API")
        except Exception as e:
            print(f"API connection test failed: {str(e)}")

    def process_document(self, file):
        """Process the uploaded document and create vectorstore"""
        if file is None:
            return None, "Please upload a document first."

        try:
            # `file` is a path string when gr.File(type="filepath") is used
            file_path = file if isinstance(file, str) else getattr(file, "name", None)
            if not file_path:
                return None, "Invalid file input."
            print(f"Processing file: {os.path.basename(file_path)}")
            
            # Choose loader based on file extension
            lower_path = file_path.lower()
            if lower_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif lower_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif lower_path.endswith(('.xlsx', '.xls')):
                loader = UnstructuredExcelLoader(file_path)
            else:
                return None, f"Unsupported file type: {os.path.basename(file_path)}"

            # Load and split the document with progress updates
            print(f"Loading document...")
            documents = loader.load()
            
            # Use very small chunks for faster processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,  # Very small chunks
                chunk_overlap=30,  # Minimal overlap
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", " ", ""]  # More aggressive splitting
            )
            
            print(f"Splitting into chunks...")
            splits = text_splitter.split_documents(documents)
            print(f"Created {len(splits)} chunks")

            # Use the smallest, fastest embedding model
            print(f"Creating embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Tiny, fast model
                model_kwargs={'device': 'cpu'},  # Use CPU for consistent performance
                cache_folder="./.cache/huggingface"  # Local caching
            )
            
            # Create vectorstore with progress indication
            print(f"Building search index...")
            persist_directory = "./.cache/chroma"
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory  # Cache embeddings locally
            )
            print(f"Search index ready!")
            
            return self.vectorstore, f"Successfully processed {os.path.basename(file_path)}"
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return None, f"Error processing document: {str(e)}"

    def chat(self, message, history, file):
        """Handle chat interaction"""
        if self.vectorstore is None:
            vectorstore, status = self.process_document(file)
            if vectorstore is None:
                return status

        try:
            # Get relevant document chunks
            docs = self.vectorstore.similarity_search(message, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Format prompt with context and chat history
            # Build prompt parts separately
            history_text = ' '.join([f'{h[0]} {h[1]}' for h in self.chat_history[-3:]])
            
            # Combine all parts using regular string formatting
            prompt = (
                "Context from document:\n"
                f"{context}\n\n"
                "Chat History:\n"
                f"{history_text}\n\n"
                f"User Question: {message}\n"
                "Assistant: Let me help you with that based on the document content."
            )

            # Generate response using OpenAI-compatible API
            messages = [
                {"role": "system", "content": f"Context from document:\n{context}\n\nChat History:\n{history_text}"},
                {"role": "user", "content": message}
            ]
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512
                )
                response = completion.choices[0].message.content
            except Exception as e:
                error_msg = f"API Error: {str(e)}"
                print(error_msg)
                return error_msg
            
            # Update chat history
            self.chat_history.append((message, response))
            
            # Return just the new response part
            return response.split("Assistant: ")[-1].strip()
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return f"Error processing your question: {str(e)}"

# Create the document chat instance
doc_chat = DocumentChat()

# Create the Gradio interface
with gr.Blocks(
    title="Document Q&A Chatbot",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    ),
) as demo:
    # Header
    with gr.Column(variant="panel"):
        gr.Markdown(
            """
            # 📚 Document Q&A Chatbot
            ### Upload your document and ask questions about its content
            """
        )
    
    # Main content area with proper spacing
    with gr.Column():
        # File upload section with minimal controls
        with gr.Column(variant="panel"):
            file_upload = gr.File(
                label="Upload Your Document (PDF, Word, Excel)",
                file_types=[".pdf", ".docx", ".xlsx", ".xls"],
                type="filepath"
            )
            
            with gr.Row():
                status = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    container=True,
                    visible=True
                )
                progress = gr.Progress(track_tqdm=True)

            # Process document automatically when uploaded
            file_upload.upload(
                fn=lambda f: doc_chat.process_document(f)[1],
                inputs=[file_upload],
                outputs=[status]
            )
        
        # Add some spacing
        gr.Markdown("---")
        
        # Chat interface with better styling
        with gr.Column(variant="panel"):
            gr.Markdown("### Chat with your Document")
            chatbot = gr.ChatInterface(
                fn=lambda msg, history: doc_chat.chat(msg, history, file_upload.value),
                chatbot=gr.Chatbot(height=400),
                textbox=gr.Textbox(
                    placeholder="Ask a question about your document...",
                    container=False
                ),
                description="Ask any questions about your document's content"
            )
    
    # Footer
    with gr.Column(variant="panel"):
        gr.Markdown(
            """
            ---
            ### Tips:
            - Your document will be processed automatically after upload
            - Wait for the "Successfully processed" message
            - Ask specific questions about the document content
            """
        )

if __name__ == "__main__":
    demo.launch(show_error=True)
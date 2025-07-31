import os
import streamlit as st
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv, find_dotenv
import time

# Page config
st.set_page_config(
    page_title="RAG Chatbot Demo",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #1e1e1e;
}

.main {
    background-color: #1e1e1e;
}

/* Title */
h1 {
    color: white;
    font-size: 28px;
    font-weight: normal;
    margin: 20px;
}

/* Upload section */
[data-testid="stFileUploader"] {
    background-color: #2b313e;
    padding: 20px;
    border-radius: 8px;
    margin: 20px;
}

[data-testid="stFileUploader"] > div > div {
    color: white !important;
}

/* File list */
.file-item {
    background-color: #1b4332;
    margin: 5px 20px;
    padding: 10px 15px;
    border-radius: 4px;
    color: white;
}

.file-item .checkmark {
    color: #4caf50;
    margin-right: 10px;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Chat interface */
.chat-message {
    margin: 8px 20px;
    padding: 15px 20px;
    border-radius: 8px;
    font-size: 16px;
    line-height: 1.5;
    color: white;
}

.chat-message.user {
    background-color: #1a73e8;
}

.chat-message.assistant {
    background-color: #28a745;
}

/* Chat container */
.chat-container {
    display: flex;
    margin: 10px 20px;
    gap: 20px;
}

/* Message containers */
.message-left {
    flex: 1;
    background-color: #1a73e8;
    padding: 15px 20px;
    border-radius: 8px;
    color: white;
    position: relative;
}

.message-right {
    flex: 2;
    background-color: #28a745;
    padding: 15px 20px;
    border-radius: 8px;
    color: white;
    position: relative;
}

/* Icons */
.icon {
    position: absolute;
    top: -10px;
    left: -10px;
    background-color: #1e1e1e;
    border-radius: 50%;
    width: 25px;
    height: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid currentColor;
}

/* Message containers */
.message-container {
    margin: 0 20px 20px 20px;
}

/* Headers */
.message-header {
    background-color: #1a73e8;
    color: white;
    padding: 8px 15px;
    border-radius: 4px 4px 0 0;
    font-size: 14px;
    display: flex;
    align-items: center;
}

.message-header.assistant {
    background-color: #28a745;
}

.message-header .icon {
    margin-right: 8px;
}

/* Message content */
.message-content {
    background-color: white;
    color: #333;
    padding: 15px;
    border-radius: 0 0 4px 4px;
    font-size: 15px;
    line-height: 1.5;
    border: 1px solid rgba(0,0,0,0.1);
}

/* Input styling */
.stTextInput > div[data-baseweb="input"] {
    background-color: #1a73e8;
    border-radius: 4px;
    margin: 20px;
    border: none;
}

.stTextInput input {
    color: white !important;
    font-size: 16px !important;
    padding: 8px 12px !important;
}

.stTextInput input::placeholder {
    color: rgba(255, 255, 255, 0.7) !important;
}

/* Override Streamlit's default input styling */
.stTextInput div[data-baseweb="base-input"] {
    background-color: transparent !important;
    border-color: transparent !important;
}

.stTextInput [data-baseweb="input"] {
    border-color: transparent !important;
}

.stTextInput [data-baseweb="input"]:hover {
    border-color: transparent !important;
    background-color: #1a73e8 !important;
}

.stTextInput [data-baseweb="input"]:focus {
    border-color: transparent !important;
    background-color: #1a73e8 !important;
}

/* Hide duplicate elements */
[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
    display: none;
}

/* Hide success messages */
.stSuccess, .stSpinner {
    display: none;
}

/* Thinking indicator */
.thinking-container {
    margin: 0 20px;
}

.thinking {
    background-color: #28a745;
    color: white;
    padding: 15px 20px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.dots {
    display: flex;
    gap: 4px;
}

.dot {
    width: 8px;
    height: 8px;
    background-color: white;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    opacity: 0.5;
}

.dot:nth-child(2) {
    animation-delay: 0.2s;
}

.dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes pulse {
    0%, 100% {
        opacity: 0.5;
    }
    50% {
        opacity: 1;
    }
}

/* Hide default spinner */
.stSpinner {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_response' not in st.session_state:
    st.session_state.last_response = None
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'thinking' not in st.session_state:
    st.session_state.thinking = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = None

# Load environment variables
load_dotenv()

# Create storage directory
os.makedirs("storage", exist_ok=True)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:""")

QA_PROMPT = PromptTemplate.from_template("""You are a precise and concise AI assistant that can read serveral document at once. Analyze the context and provide a clear, focused answer.

Context:
{context}

Question: {question}

Instructions:
1. Provide a concise answer (2-5 sentences maximum as bullet points)
2. Focus on the most relevant information from both the context and your knowledge
3. If adding external knowledge, mark it with [Additional Context]
4. If the question cannot be answered from the context, say so
5. If the answer requires clarification, ask for it

Answer:""")

def process_document(file):
    # Save uploaded file temporarily
    temp_path = f"storage/temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())
    
    try:
        # Load document based on file type
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(temp_path)
        elif file.name.endswith(('.xlsx', '.xls')):
            loader = UnstructuredExcelLoader(temp_path)
        else:
            raise ValueError("Unsupported file format")
        
        documents = loader.load()
        if not documents:
            raise ValueError(f"No content found in {file.name}")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            raise ValueError(f"Could not split content from {file.name}")
        
        # Add source filename to metadata
        for split in splits:
            split.metadata["source"] = file.name
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Initialize or update vector store
        if not st.session_state.vector_store:
            st.session_state.vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.session_state.uploaded_files.clear()  # Clear any stale file records
        else:
            st.session_state.vector_store.add_documents(splits)
        
        # Store file information
        st.session_state.uploaded_files[file.name] = {
            'path': temp_path,
            'chunks': len(splits),
            'processed': True
        }
        
        # Initialize QA chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        
        # Add existing chat history to memory if any
        if hasattr(st.session_state, 'chat_history'):
            for q, a in st.session_state.chat_history:
                memory.chat_memory.add_user_message(q)
                memory.chat_memory.add_ai_message(a)
        
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                temperature=0.2,
                model="gpt-4-turbo-preview",
            ),
            retriever=st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            rephrase_question=True,
            return_generated_question=True,
            verbose=True
        )
        
        return True
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        if file.name in st.session_state.uploaded_files:
            del st.session_state.uploaded_files[file.name]
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_response(question: str) -> dict:
    if not st.session_state.qa_chain or not st.session_state.vector_store:
        raise ValueError("Please upload a document first")
    
    if not st.session_state.uploaded_files:
        raise ValueError("No documents are currently loaded. Please upload documents first.")
    
    try:
        # Add the current question to chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        result = st.session_state.qa_chain({
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        
        # Update chat history with the new Q&A pair
        if result.get("answer"):
            st.session_state.chat_history.append((question, result["answer"]))
        
        return {
            "answer": result.get("answer", "No answer found"),
            "sources": [{
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            } for doc in result.get("source_documents", [])]
        }
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return {"answer": "Error processing your question", "sources": []}

def handle_input():
    if st.session_state.question_input and not st.session_state.thinking:
        if not st.session_state.uploaded_files:
            st.error("Please upload some documents first.")
            return
        
        question = st.session_state.question_input
        st.session_state.question_input = ""  # Clear input
        st.session_state.thinking = True
        
        try:
            # Set a maximum processing time
            start_time = time.time()
            max_processing_time = 30  # 30 seconds timeout
            
            response = get_response(question)
            
            # Check if processing took too long
            if time.time() - start_time > max_processing_time:
                raise TimeoutError("Response took too long. Please try again.")
            
            if response and "answer" in response:
                # Only add to chat history if it's a new question
                if not st.session_state.chat_history or st.session_state.chat_history[-1][0] != question:
                    st.session_state.chat_history.append((question, response["answer"]))
                st.session_state.last_response = response
            else:
                raise ValueError("Invalid response format")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.thinking = False

# Main UI
st.markdown("<h1>RAG Chatbot Demo</h1>", unsafe_allow_html=True)

# File upload section
uploaded_files = st.file_uploader(
    "Upload documents (PDF, Word, or Excel)\nLimit 200MB per file • PDF, DOCX, XLSX, XLS",
    type=['pdf', 'docx', 'xlsx', 'xls'],
    accept_multiple_files=True
)

# Process and display files
if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                if process_document(file):
                    st.session_state.uploaded_files[file.name] = {
                        'size': f"{file.size / 1024:.1f}KB",
                        'processed': True
                    }

# Display processed files
for filename, info in st.session_state.uploaded_files.items():
    if info.get('processed'):
        st.markdown(f"""
        <div class="file-item">
            <span class="checkmark">✓</span>{filename} {info['size']}
        </div>
        """, unsafe_allow_html=True)

# Chat interface
if st.session_state.chat_history:
    for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
        # User message
        st.markdown(f"""
        <div class="message-container">
            <div class="message-header">
                <span class="icon">👤</span> You
            </div>
            <div class="message-content">
                {question}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="message-container">
            <div class="message-header assistant">
                <span class="icon">🤖</span> Assistant
            </div>
            <div class="message-content">
                {answer}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Show thinking indicator
if st.session_state.thinking:
    st.markdown("""
    <div class="thinking-container">
        <div class="thinking">
            <span>🤖 Assistant is thinking</span>
            <div class="dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Question input
st.text_input(
    "Question Input",
    key="question_input",
    placeholder="Ask a question...",
    label_visibility="hidden",
    on_change=handle_input
)

# Add buttons in a row
if st.session_state.thinking or st.session_state.chat_history:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.session_state.thinking and st.button("Cancel"):
            st.session_state.thinking = False
            st.session_state.question_input = ""
            st.rerun()
    
    with col2:
        if st.session_state.chat_history and st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.last_response = None
            st.rerun()
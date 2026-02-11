import os
import hashlib
import json
from pathlib import Path
from typing import List, TypedDict
import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide")
st_callback = StreamlitCallbackHandler(st.container())


# Environment setup
def set_env(var: str):
    if not os.environ.get(var):
        return


load_dotenv()

# Disable ALL telemetry and tracing to prevent data leakage
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["DO_NOT_TRACK"] = "1"
# Remove API keys from environment to prevent accidental use
for key in ["LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"]:
    os.environ.pop(key, None)


# =============================================================================
# Auto-detect and vectorize new documents from data/ directory
# =============================================================================

DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
MANIFEST_FILE = os.path.join(FAISS_DIR, ".manifest.json")


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_data_directory(data_dir: str) -> dict:
    """Scan data directory and return dict of {filepath: hash}."""
    if not os.path.exists(data_dir):
        return {}

    file_hashes = {}
    data_path = Path(data_dir)
    exclude_patterns = ['.DS_Store', '.ipynb_checkpoints', '*.pyc', '__pycache__']

    for file in data_path.rglob('*'):
        if not file.is_file():
            continue
        if any(file.match(pattern) for pattern in exclude_patterns):
            continue
        # Only process PDF files
        if file.suffix.lower() == '.pdf':
            file_hashes[str(file.absolute())] = compute_file_hash(str(file))

    return file_hashes


def load_manifest() -> dict:
    """Load the manifest of previously indexed files."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    """Save the manifest of indexed files."""
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def load_documents_from_files(file_paths: List[str]) -> List[Document]:
    """Load PDF documents from a list of file paths."""
    documents = []
    for filepath in file_paths:
        file = Path(filepath)
        try:
            if file.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file))
                documents.extend(loader.load())
                print(f"Loaded: {file.name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 128) -> List[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked = []
    for doc in documents:
        chunked.extend(splitter.split_documents([doc]))
    return chunked


def rebuild_faiss_index(embedding_model, data_dir: str, faiss_dir: str) -> FAISS:
    """Rebuild the entire FAISS index from the data directory."""
    file_hashes = scan_data_directory(data_dir)
    if not file_hashes:
        print("No documents found in data directory.")
        return None

    print(f"Building FAISS index from {len(file_hashes)} files...")
    documents = load_documents_from_files(list(file_hashes.keys()))
    if not documents:
        print("No documents could be loaded.")
        return None

    chunked = chunk_documents(documents)
    print(f"Created {len(chunked)} chunks from {len(documents)} documents.")

    db = FAISS.from_documents(chunked, embedding_model)
    db.save_local(faiss_dir)
    save_manifest(file_hashes)
    print(f"FAISS index saved to {faiss_dir}")
    return db


def check_and_update_faiss_index(dev_mode: bool) -> bool:
    """
    Check if data/ directory has new or modified files.
    Returns True if index was rebuilt, False otherwise.
    """
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        print("Create 'data/' folder and add PDF/CSV/TXT files to auto-index them.")
        return False

    current_files = scan_data_directory(DATA_DIR)
    if not current_files:
        print("No indexable files found in data/ directory.")
        return False

    old_manifest = load_manifest()

    # Check for changes
    new_files = set(current_files.keys()) - set(old_manifest.keys())
    removed_files = set(old_manifest.keys()) - set(current_files.keys())
    modified_files = {
        f for f in current_files
        if f in old_manifest and current_files[f] != old_manifest[f]
    }

    if not (new_files or removed_files or modified_files):
        print("No changes detected in data/ directory.")
        return False

    # Report changes
    if new_files:
        print(f"New files detected: {len(new_files)}")
        for f in new_files:
            print(f"  + {Path(f).name}")
    if modified_files:
        print(f"Modified files detected: {len(modified_files)}")
        for f in modified_files:
            print(f"  ~ {Path(f).name}")
    if removed_files:
        print(f"Removed files detected: {len(removed_files)}")
        for f in removed_files:
            print(f"  - {Path(f).name}")

    # Rebuild index
    print("Rebuilding FAISS index...")
    embedding_model = get_embedding_model(dev_mode)
    rebuild_faiss_index(embedding_model, DATA_DIR, FAISS_DIR)
    return True


def get_embedding_model(dev_mode: bool, model_name: str = 'nomic-embed-text'):
    if dev_mode:
        try:
            return HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        except Exception:
            # Fallback to Ollama embeddings if local HF load fails (avoids PyTorch meta tensor issues)
            return OllamaEmbeddings(model='nomic-embed-text')
    else:
        return HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        #return OllamaEmbeddings(model=model_name)


def get_retriever(db, k: int = 3):
    return db.as_retriever(k=k, search_type="mmr")


def filter_documents_by_role(documents: List[Document], role: str) -> List[Document]:
    filtered_docs = []
    for doc in documents:
        access_roles = doc.metadata.get("access_roles", [])
        if role in access_roles:
            filtered_docs.append(doc)
    return filtered_docs


def get_llm(dev_mode: bool):
    # Always use local Ollama - no cloud LLM to prevent data leakage
    return ChatOllama(model="llama3.2:3b", temperature=0)


def get_llm_json(dev_mode: bool):
    # Always use local Ollama with JSON format - no cloud LLM
    return ChatOllama(model="llama3.2:3b", temperature=0, format="json")

def to_json_dict(x, default=None):
    if isinstance(x, dict):
        return x
    else:
        try:
            return json.loads(x.content)
        except json.JSONDecodeError:
            return default if default is not None else {"error": "Invalid JSON format"}

# Define our graph state type
class MyGraphState(TypedDict):
    question: str
    max_retries: int
    documents: List[Document]
    generation: str
    loop_step: int


# Build and cache the agent graph
def build_agent(dev_mode: bool, ac: bool, persist_directory: str, k: int):
    st.text("Loading model...")
    embedding_model = get_embedding_model(dev_mode)
    db = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
    retriever = get_retriever(db, k)
    llm = get_llm(dev_mode)
    llm_json = get_llm_json(dev_mode)
    role = st.session_state.get("role", "manager")

    doc_grader_instructions = (
        "You are assessing whether a document is relevant to a given question.\n"
        "- If the document provides information that helps answer the question, mark it as relevant.\n"
        "- Otherwise, mark it as not relevant.\n"
        "Respond with a JSON object: {\"relevant\": \"yes\"} or {\"relevant\": \"no\"}.\n"
        "Do NOT include any additional commentary or explanation."
    )

    hallucination_instructions = (
        "Determine whether the given answer is fully supported by the provided facts.\n"
        "- If the answer contains any information not present in the facts, set binary_score to \"no\".\n"
        "- If the answer is fully grounded in the facts, set binary_score to \"yes\".\n"
        "Respond with a JSON object in the format:\n"
        "{\"binary_score\": \"yes\" or \"no\", \"explanation\": \"...\"}"
    )

    answer_instructions = (
        "Evaluate whether the generated answer directly and clearly addresses the user's question.\n"
        "- If the answer responds appropriately to the question, set 'answered' to \"yes\".\n"
        "- If it misses the point, is vague, or irrelevant, set 'answered' to \"no\".\n"
        "Return a JSON object: {\"answered\": \"yes\" or \"no\", \"explanation\": \"...\"}"
    )

    rag_prompt = (
        "You are an AI assistant. Use the information from the CONTEXT to answer the user's QUESTION in no more than 10 sentences. And try to include all relevant information.\n"
        "Only include facts that are explicitly mentioned in the CONTEXT. Do not speculate or invent information. Also don't repeat yourself.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve(state):
        print(f"Retrieving documents for question: {state['question']}")
        docs = retriever.invoke(state["question"])
        if ac:
            docs = filter_documents_by_role(docs, role)
        return {"documents": docs}

    def grade_documents(state):
        filtered = []
        for d in state.get("documents", []):
            prompt = f"Document: {d.page_content}\nQuestion: {state['question']}"
            resp_gd = llm_json.invoke([SystemMessage(content=doc_grader_instructions), HumanMessage(content=prompt)])
            resp_gd = to_json_dict(resp_gd, {"relevant": "no"})
            if resp_gd["relevant"] == "yes":
                filtered.append(d)
        if not filtered:
            filtered = state.get("documents", [])
        return {"documents": filtered}

    def generate(state):
        ctx = format_docs(state["documents"])
        prompt = rag_prompt.format(context=ctx, question=state["question"])
        gen = llm.invoke([HumanMessage(content=prompt)])
        return {"generation": gen.content, "loop_step": state.get("loop_step", 0) + 1}

    def grade_generation(state):
        facts = format_docs(state["documents"])
        hall_prompt = f"FACTS:\n{facts}\nANSWER:{state['generation']}"
        hall = llm_json.invoke([SystemMessage(content=hallucination_instructions), HumanMessage(content=hall_prompt)])
        hall = to_json_dict(hall, {"binary_score": "no", "explanation": "No explanation provided."})
        if hall["binary_score"] == "no" and state.get("loop_step", 0) < state.get("max_retries", 3):
            return "not_supported"
        ans_prompt = f"QUESTION:\n{state['question']}\nANSWER:{state['generation']}"
        ans = llm_json.invoke([SystemMessage(content=answer_instructions), HumanMessage(content=ans_prompt)])
        ans = to_json_dict(ans, {"answered": "no", "explanation": "No explanation provided."})
        score = ans["answered"]
        exp = ans["explanation"]
        print(f"The score was {score} and {exp}")
        if score == "yes":
            return "useful"
        if state.get("loop_step", 0) < state.get("max_retries", 3):
            return "not_supported"
        return "max_retries"

    # Build graph - PDF-only RAG
    graph = StateGraph(MyGraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)

    # Simple linear flow: retrieve → grade → generate
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("grade_documents", "generate")
    graph.add_conditional_edges("generate", grade_generation, {
        "not_supported": "generate",
        "useful": END,
        "max_retries": END
    })
    return graph.compile()


# Streamlit UI

st.title("PDF RAG Assistant 📄")


def save_chat_history(filename=os.path.join(BASE_DIR, "chat_history.json")):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)


def load_chat_history(filename=os.path.join(BASE_DIR, "chat_history.json")):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.session_state.messages = []


# Sidebar controls
with st.sidebar:
    st.header("Options")
    if st.button("Delete Chat History"):
        filename = os.path.join(BASE_DIR, "chat_history.json")
        if os.path.exists(filename):
            os.remove(filename)
            st.success("Chat history file deleted!")
        else:
            st.warning("No chat history file found to delete.")

    st.divider()
    st.subheader("Document Index")
    # Show data directory status
    if os.path.exists(DATA_DIR):
        file_count = len(scan_data_directory(DATA_DIR))
        st.caption(f"📁 {file_count} files in data/ folder")
    else:
        st.caption("📁 No data/ folder found")
        st.info(f"Create folder: {DATA_DIR}")

    if st.button("🔄 Reindex Documents"):
        with st.spinner("Checking for document changes..."):
            if check_and_update_faiss_index(st.session_state.get("dev_mode", False)):
                st.success("Index rebuilt! Reloading agent...")
                # Force agent reload
                if "agent_graph" in st.session_state:
                    del st.session_state.agent_graph
                st.rerun()
            else:
                st.info("No changes detected.")

dev_mode = st.sidebar.checkbox("Developer mode", False)
st.session_state.dev_mode = dev_mode
role = st.sidebar.selectbox("Role", ["manager", "employee"])
if role != st.session_state.get("role", "manager"):
    st.session_state.role = role
    st.session_state.agent_graph = build_agent(dev_mode, False, os.path.join(BASE_DIR, "faiss_index"), 3)
    #st.session_state.agent_graph = build_agent(dev_mode, True, "faiss_index_ac", 3)
    st.session_state.messages = []
k = st.sidebar.number_input("Retriever top-k", min_value=1, max_value=10, value=3)

# Auto-check for new documents on first load
if "index_checked" not in st.session_state:
    st.session_state.index_checked = True
    with st.spinner("Checking for new documents..."):
        if check_and_update_faiss_index(dev_mode):
            st.toast("Document index updated!", icon="✅")

# Initialize session state
if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = build_agent(dev_mode, False, os.path.join(BASE_DIR, "faiss_index"), k)
    #st.session_state.agent_graph = build_agent(True, True, "faiss_index_ac", 3)
agent = st.session_state.agent_graph

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f'<div class="source-text">{source}</div>', unsafe_allow_html=True)
                        st.markdown("---")
# User input
if question := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        st.write("Processing your request...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke({"question": question})
        print(f"Response from agent: {response}")
        docs = response.get("documents", [])
        st.write(response.get("generation", "No generation found."))
        if docs:
            with st.expander("View Sources"):
                st.markdown("PDF Sources")
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"**File:** {doc.metadata.get('source', 'unknown')}")
                    st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                    st.markdown("---")
                    st.markdown(f'<div class="source-text">{doc.page_content}</div>',
                                unsafe_allow_html=True)
                    st.markdown("---")
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("generation", "No generation found."),
            "sources": [doc.page_content for doc in docs] if docs else []
        })
        save_chat_history()

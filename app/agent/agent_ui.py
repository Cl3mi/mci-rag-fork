import os
import hashlib
import getpass
import json
from pathlib import Path
from typing import List, TypedDict
import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
#from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
# DISABLED: Web search - we run fully offline
# from langchain_community.tools.tavily_search import TavilySearchResults
# DISABLED: Cloud LLM - we use local Ollama only
# from langchain_groq import ChatGroq
import pandas as pd
# DISABLED: PandasAI may phone home - use plain pandas instead
# import pandasai as pdai
from sqlalchemy import text, create_engine

from db_helper import build_sql_prompt, get_db_schema

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Electronic.Inc Assistant",
    page_icon="🤖",
    layout="wide")
st_callback = StreamlitCallbackHandler(st.container())


# Environment setup
def set_env(var: str):
    if not os.environ.get(var):
        return


load_dotenv()
# DISABLED: External API keys - we run fully offline
# set_env("TAVILY_API_KEY")
# set_env("LANGSMITH_API_KEY")
# set_env("GROQ_API_KEY")

# Disable ALL telemetry and tracing to prevent data leakage
# These MUST be set after load_dotenv() to override any .env values
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Disable LangSmith tracing v2
os.environ["LANGCHAIN_TRACING"] = "false"     # Disable LangSmith tracing v1
os.environ["LANGSMITH_TRACING"] = "false"     # Another LangSmith flag
os.environ["DO_NOT_TRACK"] = "1"              # Disable LangChain anonymous telemetry
# Remove API keys from environment to prevent accidental use
for key in ["LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "TAVILY_API_KEY", "GROQ_API_KEY"]:
    os.environ.pop(key, None)
# NOTE: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE removed to allow model downloads
# After models are cached, you can re-enable these for fully offline operation:
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"


# =============================================================================
# Auto-detect and vectorize new documents from data/ directory
# =============================================================================

DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_ei")
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
        # Only process supported file types
        if file.suffix.lower() in ['.pdf', '.csv', '.txt', '.md']:
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
    """Load documents from a list of file paths."""
    documents = []
    for filepath in file_paths:
        file = Path(filepath)
        try:
            if file.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file))
            elif file.suffix.lower() == '.csv':
                loader = CSVLoader(str(file), encoding='utf-8')
            else:  # .txt, .md, etc.
                loader = TextLoader(str(file), encoding='utf-8')
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
    web_search: str


# Build and cache the agent graph
def build_agent(dev_mode: bool, ac: bool, persist_directory: str, k: int):
    st.text("Loading model...")
    # Load resources - fully offline, no external API calls
    embedding_model = get_embedding_model(dev_mode)
    db = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
    retriever = get_retriever(db, k)
    # DISABLED: Web search removed - we run fully offline
    # tavily_api_key = os.getenv("TAVILY_API_KEY")
    # web_search_tool = TavilySearchResults(k=3, tavily_api_key=tavily_api_key) if tavily_api_key else None
    llm = get_llm(dev_mode)
    llm_json = get_llm_json(dev_mode)
    role = st.session_state.get("role", "manager")

    DB_URL = os.getenv("DB_URL", "sqlite:///./electronic_inc.db")
    engine = create_engine(DB_URL + "?mode=ro", future=True)
    schemes = get_db_schema(engine)
    router_instructions = (
        "You are a smart router that decides whether a user question should be answered using one of the following LOCAL data sources:\n"
        "1. internal company documents (vectorstore)\n"
        "2. structured CSV data (csv)\n"
        "3. sql database (sqldatabase)\n\n"

        "Routing Rules:\n"
        "- If the question is about Electronic Inc.'s company history, strategy, values, or internal documentation, or person information return: {\"result\": \"vectorstore\"}.\n"
        "- If the question involves numerical data, trends, sales figures, costs, or anything typically found in spreadsheets or tables, return: {\"result\": \"csv\"}.\n"
        "- If the question is about aggregations, or insights that can be obtained from a SQL database of staff and employees and infos about persons, like email, department, country, return: {\"result\": \"sqldatabase\"}.\n"
        "- For any other question, default to: {\"result\": \"vectorstore\"}.\n\n"

        "Respond ONLY with a valid JSON object in the form: {\"result\": \"vectorstore\"}, {\"result\": \"csv\"}, or {\"result\": \"sqldatabase\"}."
    )

    doc_grader_instructions = (
        "You are assessing whether a document is relevant to a given question.\n"
        "- If the document provides information that helps answer the question AND is related to Electronic Inc., mark it as relevant.\n"
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
        "- If the answer responds appropriately to the question or is from a CSV, set 'answered' to \"yes\".\n"
        "- If it misses the point, is vague, or irrelevant, set 'answered' to \"no\".\n"
        "Return a JSON object: {\"answered\": \"yes\" or \"no\", \"explanation\": \"...\"}"
    )

    rag_prompt = (
        "You are an AI assistant. Use the information from the CONTEXT to answer the user's QUESTION in no more than 10 sentences. And try to include all relevant information.\n"
        "Only include facts that are explicitly mentioned in the CONTEXT. Do not speculate or invent information. Also don't repeat yourself.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"
    )

    sql_prompt = (
        "You are an expert SQL generator. Given a user question, generate an appropriate SQL query to retrieve the necessary data from the database.\n"
        "Only generate SELECT queries. Do not include any explanations or additional text, only the SQL query.\n"
        "If the question is not answerable with SQL, respond with: NO_SQL.\n\n"
        "User Question: {question}\n\nSQL Query:"
    )

    # Define action functions

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def route_question(state):
        msg = llm_json.invoke([
            SystemMessage(content=router_instructions),
            HumanMessage(content=state["question"]),
        ])
        msg = to_json_dict(msg, {"result": "vectorstore"})
        print(f"Routing decision: {msg}")
        return msg.get("result", "vectorstore")

    def execute_query(query: str):
        with engine.connect() as conn:
            result = conn.execute(text(query))
            if result.returns_rows:  # Check if the query returns rows
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
            else:
                return None

    def format_sql_result(df: pd.DataFrame) -> str:
        if df.shape[1] == 1:
            return "\n".join(map(str, df.iloc[:, 0].tolist()))
        # Multi-column → pretty table without index
        return df.to_string(index=False)

    def sql_analysis(state):
        prompt = build_sql_prompt(state["question"], schemes)
        sql_response = llm.invoke([HumanMessage(content=prompt)])
        sql_query = sql_response.content.strip().rstrip(";")
        print(f"Generated SQL query: {sql_query}")
        if sql_query == "NO_SQL":
            return {
                "documents": state.get("documents", []) + [Document(page_content="No relevant SQL query could be generated.")],
                "question": state["question"],
                "max_retries": state["max_retries"],
                "generation": "",
                "loop_step": state.get("loop_step", 0),
                "web_search": "no"
            }
        try:
            df = execute_query(sql_query)
            if df is None or df.empty:
                answer = "The SQL query returned no results."
            else:
                answer = format_sql_result(df.head(5))
        except Exception as e:
            answer = f"Error executing SQL query: {e}"
        print(f"SQL analysis answer: {answer}")
        answer_string = f"SQL analysis answer to the question {state['question']} is: {answer}"
        return {
            "documents": state.get("documents", []) + [Document(page_content=answer_string)]
        }


    def analyze_csv(csv_file: str, question: str):
        """Analyze CSV using local LLM without PandasAI (which may phone home)."""
        data = pd.read_csv(csv_file)
        schema = ", ".join(data.columns)
        # Provide sample data and stats to the LLM for analysis
        sample_rows = data.head(10).to_string(index=False)
        stats = data.describe().to_string() if data.select_dtypes(include='number').shape[1] > 0 else "No numeric columns"

        csv_prompt = (
            f"You are analyzing a CSV file with the following columns: {schema}\n\n"
            f"Sample data (first 10 rows):\n{sample_rows}\n\n"
            f"Statistics:\n{stats}\n\n"
            f"Question: {question}\n\n"
            "Provide a concise answer in no more than 10 sentences based on the data above."
        )
        response = llm.invoke([HumanMessage(content=csv_prompt)])
        return response.content

    def csv_search(state):

        hits = retriever.invoke(state["question"])
        if not hits:
            print("No relevant CSV documents found.")
            return {
                "documents": [Document(page_content="No relevant CSV found.")],
                "question": state["question"],
                "max_retries": state["max_retries"],
                "generation": "",
                "loop_step": state.get("loop_step", 0),
                "web_search": "no"
            }
        source = hits[0].metadata.get("source", "unknown")
        print(f"CSV search source: {source}")
        try:
            answer = analyze_csv(source, state["question"])
        except Exception as e:
            answer = f"Error analyzing CSV: {e}"
        print(f"CSV search answer: {answer}")
        answer_string = f"CSV search answer to the question {state['question']} is: {answer}"
        return {
            "documents": state.get("documents", []) + [Document(page_content=answer_string, metadata={"source": source})]
        }

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
        # If no relevant docs found, keep original docs and let generate handle it
        # NO web search fallback - we stay fully offline
        if not filtered:
            filtered = state.get("documents", [])
        return {"documents": filtered, "web_search": "no"}

    def generate(state):
        ctx = format_docs(state["documents"])
        prompt = rag_prompt.format(context=ctx, question=state["question"])
        gen = llm.invoke([HumanMessage(content=prompt)])
        return {"generation": gen.content, "loop_step": state.get("loop_step", 0) + 1}

    # REMOVED: web_search function - we run fully offline

    def decide_generate(state):
        # Always generate - no web search fallback
        return "generate"

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
        # NO web search fallback - retry generation or end
        if state.get("loop_step", 0) < state.get("max_retries", 3):
            return "not_supported"  # Retry generation instead of websearch
        return "max_retries"

    # Build graph - FULLY OFFLINE, no websearch node
    graph = StateGraph(MyGraphState)
    # REMOVED: graph.add_node("websearch", web_search)
    graph.add_node("retrieve", retrieve)
    graph.add_node("csv_search", csv_search)
    graph.add_node("sql_analysis", sql_analysis)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)

    # Route only to local data sources - no websearch
    graph.set_conditional_entry_point(route_question, {
        "vectorstore": "retrieve",
        "csv": "csv_search",
        "sqldatabase": "sql_analysis"
    })
    # REMOVED: graph.add_edge("websearch", "generate")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("csv_search", "generate")
    graph.add_edge("sql_analysis", "generate")
    # Always go to generate - no websearch fallback
    graph.add_edge("grade_documents", "generate")
    graph.add_conditional_edges("generate", grade_generation, {
        "not_supported": "generate",  # Retry generation
        "useful": END,
        "max_retries": END
        # REMOVED: "not_useful": "websearch"
    })
    return graph.compile()


# Streamlit UI

st.title("Electronic.Inc Assistant 🤖")


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
    st.session_state.agent_graph = build_agent(dev_mode, False, os.path.join(BASE_DIR, "faiss_ei"), 3)
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
    st.session_state.agent_graph = build_agent(dev_mode, False, os.path.join(BASE_DIR, "faiss_ei"), k)
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
        web_search = response.get("web_search", "no")
        st.write(response.get("generation", "No generation found."))
        if docs:
            with st.expander("View Sources"):
                st.markdown("Web Search" if web_search == "yes" else "Vectorstore Results")
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"**File:** {doc.metadata.get('source', 'unknown')}")
                    st.markdown(f"**Date:** {doc.metadata.get('moddate', 'N/A')}")
                    st.markdown("---")
                    st.markdown(f'<div class="source-text">{doc.page_content}</div>',
                                unsafe_allow_html=True)
                    st.markdown("---")
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("generation", "No generation found."),
            "sources": [doc.page_content for doc in docs] if docs else []
        })
        # Autosave chat history after each assistant response
        save_chat_history()
        #for event in agent.stream({"question": question, "max_retries": 3}, stream_mode="values"):
            #st.write(event)


# What is the email address of the HR manager at Electronic Inc.?
# What is the min Total Amount in retail sales and what Gender has the client?
# What is the biggest country in Europe?
# What is the GDPR?
# Has Electronic Inc a Orientation Session sheduled?

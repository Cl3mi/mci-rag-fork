import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, SeleniumURLLoader, RecursiveUrlLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
# Set development mode
dev_mode = True

def load_documents(docs_path: str, exclude_patterns=None) -> list:
    if exclude_patterns is None:
        exclude_patterns = ['.DS_Store', '.ipynb_checkpoints']

    base_path = Path(docs_path)
    all_files = base_path.rglob('*')

    documents = []
    for file in all_files:
        if not file.is_file():
            continue

        if any(file.match(pattern) for pattern in exclude_patterns):
            continue

        # Decide loader by extension
        if file.suffix.lower() == '.pdf':
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(str(file.absolute()))
        elif file.suffix.lower() == '.csv':
            print(f"Loading CSV: {file}")
            loader = CSVLoader(str(file.absolute()), encoding='utf-8')
        else:
            print(f"Loading text: {file}")
            loader = TextLoader(str(file.absolute()), encoding='utf-8')

        documents.extend(loader.load())

    print(f"Total documents loaded: {len(documents)}")
    return documents

# Load website data
def decode_website(urls: list):
    loader = SeleniumURLLoader(urls)
    docs = loader.load()
    web_text = ""

    for page in docs:
        web_text += page.page_content + " "
        print(web_text[:1000])
    return docs

def recursive_decode_website(url):
    loader = RecursiveUrlLoader(url, max_depth=2, use_async=False, prevent_outside=True)
    docs = loader.load()
    web_text = ""

    for page in docs:
        web_text += page.page_content + " "
        print(web_text[:1000])
    return docs

def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        chunked_docs.extend(doc_chunks)

    print(f"Total documents after chunking: {len(chunked_docs)}")
    return chunked_docs

def remove_directory(directory: str) -> None:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Removed existing Chroma directory: {directory}")
    else:
        print(f"No existing Chroma directory found at: {directory}")

def get_embedding_model(model_name: str = 'llama3.2:3b'):
    print(f"Loading embedding model: {model_name}")
    if dev_mode:
        return HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    else:
        return OllamaEmbeddings(model=model_name)

def build_vectorstore(docs: list, embeddings, persist_dir: str, collection: str):
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection
    )
    print(f"Chroma DB persisted at: {persist_dir}")
    return db

def build_faiss_vectorstore(docs: list, embeddings, persist_dir: str):
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    vectorstore.save_local(persist_dir)
    print(f"FAISS index saved at: {persist_dir}")
    return vectorstore


def main():
    chroma = False
    db_dir = 'faiss_ei'
    db_name = 'inc-support'

    # 1) Remove old Chroma data
    # remove_directory(db_dir)

    # 2) Get embedding model
    embedding_llm = get_embedding_model('llama3.2:3b')

    # 3) Load documents
    raw_documents = load_documents('./data')

    # 4) Chunk documents
    chunked_documents = chunk_documents(raw_documents, chunk_size=512, chunk_overlap=128)

    if not chunked_documents:
        print('No documents to add to the vectorstore. Exiting.')
        return
    # 5) Build and persist vectorstore
    if chroma:
        db = build_vectorstore(
            docs=chunked_documents,
            embeddings=embedding_llm,
            persist_dir=db_dir,
            collection=db_name
        )
    else:
        db = build_faiss_vectorstore(
            docs=chunked_documents,
            embeddings=embedding_llm,
            persist_dir=db_dir
        )


if __name__ == '__main__':
    main()
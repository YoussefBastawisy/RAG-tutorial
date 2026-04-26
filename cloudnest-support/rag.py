from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"


# --- Config ---
KB_DIR = Path(__file__).parent / "data" / "kb"
PERSIST_DIR = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "cloudnest_kb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_kb(kb_dir: Path = KB_DIR) -> List[Document]:
    """Load all .md files in the knowledge base directory."""
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No markdown files found in {kb_dir}")
    return docs


def load_pdf(pdf_path: str) -> List[Document]:
    """Load a single PDF (used for user uploads)."""
    return PyPDFLoader(pdf_path).load()


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the embedding model. Downloads ~90MB on first run."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document], embeddings) -> Chroma:
    """Build a fresh persistent vectorstore from the given chunks."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )


def _load_persisted_vectorstore(embeddings) -> Chroma:
    """Open the on-disk vectorstore without re-embedding."""
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def build_default_vectorstore() -> Chroma:
    """Return the KB vectorstore.

    On first run we embed all KB markdown files and persist to .chroma/.
    On subsequent runs we just open the existing on-disk store, which is
    instant and survives Streamlit hot-reloads (avoiding the well-known
    'no such table: collections' issue with in-memory Chroma).
    """
    embeddings = get_embeddings()

    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        vs = _load_persisted_vectorstore(embeddings)
        try:
            if vs._collection.count() > 0:
                return vs
        except Exception:
            pass

    docs = load_kb()
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks, embeddings)


def add_pdf_to_vectorstore(vectorstore: Chroma, pdf_path: str) -> int:
    """Add a user-uploaded PDF to an existing vectorstore. Returns chunks added."""
    docs = load_pdf(pdf_path)
    chunks = chunk_documents(docs)
    vectorstore.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    # Quick sanity check when run directly
    print("Loading KB...")
    docs = load_kb()
    print(f"  Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    print("Building vectorstore (downloads model on first run)...")
    vs = build_default_vectorstore()

    print("\nTest query: 'how do I get a refund?'")
    results = vs.similarity_search("how do I get a refund?", k=3)
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "?").split("/")[-1]
        print(f"\n  [{i}] {source}")
        print(f"      {r.page_content[:150]}...")

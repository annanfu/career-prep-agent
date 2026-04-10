"""Ingestion pipeline: loads, chunks, embeds, stores documents in ChromaDB."""

import json
import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_data"
COLLECTION_NAME = "knowledge_base"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".tex", ".docx"}


def _load_file(file_path: Path) -> List[Document]:
    """Load a single file using the appropriate LangChain document loader.

    Args:
        file_path: Path to the file to load.

    Returns:
        List of Document objects loaded from the file.
    """
    ext = file_path.suffix.lower()
    path_str = str(file_path)

    if ext == ".pdf":
        loader = PyPDFLoader(path_str)
    elif ext == ".docx":
        loader = Docx2txtLoader(path_str)
    elif ext in (".txt", ".md", ".tex"):
        loader = TextLoader(path_str, encoding="utf-8")
    else:
        return []

    return loader.load()


def _load_star_stories(stories_dir: str = "output/stories") -> List[Document]:
    """Load accumulated STAR story JSON files from the stories directory.

    Args:
        stories_dir: Path to directory containing STAR story JSON files.

    Returns:
        List of Document objects, one per story entry.
    """
    stories_path = Path(stories_dir)
    if not stories_path.exists():
        return []

    documents: List[Document] = []
    for json_file in stories_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                stories = json.load(f)
            if not isinstance(stories, list):
                stories = [stories]
            for story in stories:
                content = json.dumps(story, indent=2)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "star_story",
                        "source_file": json_file.name,
                    },
                )
                documents.append(doc)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not load {json_file.name}: {e}")

    return documents


def ingest_documents(directory: str = "knowledge_base") -> Tuple[int, int]:
    """Recursively ingest documents from a directory into ChromaDB.

    Scans the given directory for supported file types (PDF, TXT, MD, TEX,
    DOCX), splits them into chunks, embeds with a local sentence-transformer
    model, and stores in a persistent ChromaDB collection. Also ingests any
    accumulated STAR story JSON files from output/stories/.

    Args:
        directory: Root directory to scan for documents. Defaults to
            'knowledge_base'.

    Returns:
        Tuple of (files_processed, total_chunks_created).
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    print(f"Scanning '{directory}' for documents...")

    # --- Load documents from knowledge_base ---
    all_docs: List[Document] = []
    files_processed = 0

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            docs = _load_file(file_path)
            if not docs:
                continue
            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["source_path"] = str(
                    file_path.relative_to(root)
                )
            all_docs.extend(docs)
            files_processed += 1
            rel = file_path.relative_to(root)
            print(f"  Loaded: {rel} ({len(docs)} page(s))")
        except Exception as e:
            print(f"  Warning: could not load {file_path.name}: {e}")

    # --- Load STAR stories ---
    star_docs = _load_star_stories()
    if star_docs:
        print(
            f"  Loaded {len(star_docs)} STAR story/stories"
            " from output/stories/"
        )
    all_docs.extend(star_docs)

    if not all_docs:
        print("No documents found to ingest.")
        return 0, 0

    # --- Split into chunks ---
    # Custom separators: prefer splitting at story/section boundaries
    # before falling back to paragraphs and sentences.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n\n",        # triple newline (story gaps)
            "\n## ",          # markdown H2
            "\n# ",           # markdown H1
            "\nStory ",       # STAR story headers
            "\nSituation",    # STAR story start
            "\n\n",           # double newline (paragraphs)
            "\n",             # single newline
            ". ",             # sentence boundary
            " ",              # word boundary
        ],
    )
    chunks = splitter.split_documents(all_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(
        f"\nSplitting complete: {len(chunks)} chunks"
        f" from {files_processed} file(s)."
    )

    # --- Embed and store in ChromaDB ---
    print(
        f"Loading embedding model '{EMBED_MODEL}'"
        " (local, no API key needed)..."
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(
        f"Storing in ChromaDB collection '{COLLECTION_NAME}'"
        f" at '{CHROMA_DIR}/'..."
    )
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Clear old collection to avoid duplicates on re-ingest
    _old = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    _old.delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    total_chunks = vectorstore._collection.count()
    print("\nIngestion complete.")
    print(f"  Files processed : {files_processed}")
    print(f"  Total chunks    : {total_chunks}")

    return files_processed, total_chunks


if __name__ == "__main__":
    files, chunks = ingest_documents("knowledge_base")
    print(f"\nSummary: processed {files} file(s), created {chunks} chunk(s).")

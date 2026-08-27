import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from fastembed import TextEmbedding

from app import config

_embedder = None
_index = None
_metadata = []

INDEX_FILE = config.INDEX_DIR / "index.faiss"
META_FILE = config.INDEX_DIR / "metadata.pkl"


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=config.EMBEDDING_MODEL)
    return _embedder


def _embed(texts):
    embedder = get_embedder()
    vecs = np.array(list(embedder.embed(texts)), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vecs / norms


def load_index():
    global _index, _metadata
    if INDEX_FILE.exists() and META_FILE.exists():
        _index = faiss.read_index(str(INDEX_FILE))
        with open(META_FILE, "rb") as f:
            _metadata = pickle.load(f)
    else:
        # BAAI/bge-small-en-v1.5 has an explicit vector dimension of 384
        dim = 384
        _index = faiss.IndexFlatIP(dim)
        _metadata = []
    return _index, _metadata


def save_index():
    faiss.write_index(_index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(_metadata, f)


def chunk_text(text, chunk_size=None, overlap=None):
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def extract_pdf_text(path):
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def ingest_pdf(path, source_name):
    global _index, _metadata
    if _index is None:
        load_index()

    text = extract_pdf_text(path)
    chunks = chunk_text(text)
    if not chunks:
        return 0, len(_metadata)

    vectors = _embed(chunks)
    _index.add(vectors)

    start_id = len(_metadata)
    for i, chunk in enumerate(chunks):
        _metadata.append({
            "text": chunk,
            "source": source_name,
            "chunk_id": start_id + i,
        })

    save_index()
    return len(chunks), len(_metadata)

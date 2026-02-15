# indexing.py
import json
import math
import pickle
from collections import Counter, defaultdict
from typing import Callable, Dict, Any


def _iter_corpus(corpus_path: str, doc_mode: str):
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = str(doc["_id"])

            title = (doc.get("title", "") or "").strip()
            text  = (doc.get("text", "")  or "").strip()

            if doc_mode == "title":
                raw = title
            else:  # "title+text"
                raw = (title + " " + text).strip()

            yield doc_id, raw



def build_inverted_index(
    corpus_path: str,
    preprocess_fn: Callable[[str], list],
    use_log_tf=False,
    doc_mode="title+text"
) -> Dict[str, Any]:
    """
    Builds:
      - index: term -> {doc_id: tf}
      - idf: term -> idf
      - doc_lengths: doc_id -> ||doc|| (vector magnitude under tf-idf)
    Compatible with your retrieval.rank_documents().
    """

    # ---------- PASS 1: compute document frequencies (df) ----------
    df = Counter()
    N = 0  # number of docs seen (non-empty after preprocessing)

    for doc_id, text in _iter_corpus(corpus_path, doc_mode):
        tokens = preprocess_fn(text)
        if not tokens:
            continue
        tf = Counter(tokens)
        N += 1
        for term in tf.keys():
            df[term] += 1

    if N == 0:
        raise ValueError("No documents were indexed. Check corpus_path or preprocessing.")

    # Smooth IDF to avoid division by zero
    # Common choice: log((N+1)/(df+1)) + 1
    idf = {term: math.log((N + 1) / (dft + 1)) + 1.0 for term, dft in df.items()}

    # ---------- PASS 2: build postings + precompute doc vector lengths ----------
    index = defaultdict(dict)
    doc_lengths = {}

    for doc_id, text in _iter_corpus(corpus_path, doc_mode):
        tokens = preprocess_fn(text)
        if not tokens:
            continue

        raw_tf = Counter(tokens)
        length_sq = 0.0

        for term, freq in raw_tf.items():
            tf_w = (1.0 + math.log(freq)) if use_log_tf else float(freq)
            index[term][doc_id] = tf_w

            w = tf_w * idf.get(term, 0.0)
            length_sq += w * w

        doc_lengths[doc_id] = math.sqrt(length_sq) if length_sq > 0 else 0.0

    return {
        "N_docs": N,
        "vocab_size": len(index),
        "use_log_tf": use_log_tf,
        "index": dict(index),
        "idf": idf,
        "doc_lengths": doc_lengths,
    }


def save_index(index_data: Dict[str, Any], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(index_data, f)


def load_index(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def sample_vocabulary(index_data: Dict[str, Any], k: int = 100):
    vocab = sorted(index_data["index"].keys())
    return vocab[:k]

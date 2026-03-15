import subprocess
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# Resolve project root reliably (.. /Method2/rerank_biencoder.py -> project root)
# ----------------------------
THIS_FILE = Path(__file__).resolve()
METHOD2_DIR = THIS_FILE.parent
PROJECT_ROOT = METHOD2_DIR.parent

CORPUS_PATH = PROJECT_ROOT / "corpus.jsonl"
QUERIES_PATH = PROJECT_ROOT / "queries.jsonl"

BASELINE_RESULTS = METHOD2_DIR / "baseline_method2.txt"
FINAL_RESULTS = METHOD2_DIR / "Results_method2_biencoder"

TOP_K = 100

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
REBUILD_BASELINE = True  # set False if you already have baseline_method2.txt


def run_baseline():
    print("Running Assignment 1 baseline retrieval (TF-IDF)...")

    cmd = [
        sys.executable,  # use the same python as your venv
        "run_ir.py",
        "--doc_mode",
        "title+text",
        "--run_name",
        "baseline_method2",
        "--results_path",
        str(BASELINE_RESULTS.relative_to(PROJECT_ROOT)),
    ]

    if REBUILD_BASELINE:
        cmd.insert(2, "--rebuild")

    # Run from project root so run_ir.py is found and relative paths make sense
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def load_corpus():
    corpus = {}
    with open(CORPUS_PATH, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = str(doc["_id"])
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            corpus[doc_id] = (title + " " + text).strip()
    return corpus


def load_queries():
    queries = {}
    with open(QUERIES_PATH, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            qid = str(q["_id"])
            title = q.get("title", "") or ""
            text = q.get("text", "") or ""
            queries[qid] = (title + " " + text).strip()
    return queries


def load_baseline():
    # baseline lines: qid Q0 docid rank score runname
    results = defaultdict(list)
    with open(BASELINE_RESULTS, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            doc_id = parts[2]
            if len(results[qid]) < TOP_K:
                results[qid].append(doc_id)
    return results


def rerank():
    corpus = load_corpus()
    queries = load_queries()
    baseline = load_baseline()

    print("Loading bi-encoder model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    doc_emb_cache = {}

    with open(FINAL_RESULTS, "w", encoding="utf8") as out:
        for i, (qid, doc_ids) in enumerate(baseline.items(), start=1):
            query = queries.get(qid, "")
            if not query:
                continue

            q_emb = model.encode(query, normalize_embeddings=True)

            missing = [d for d in doc_ids if d in corpus and d not in doc_emb_cache]
            if missing:
                texts = [corpus[d] for d in missing]
                embs = model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for d, e in zip(missing, embs):
                    doc_emb_cache[d] = e

            kept_docs = [d for d in doc_ids if d in doc_emb_cache]
            if not kept_docs:
                continue

            doc_mat = np.stack([doc_emb_cache[d] for d in kept_docs], axis=0)
            scores = doc_mat @ q_emb  # cosine similarity (normalized vectors)

            ranked = list(zip(kept_docs, scores.tolist()))
            ranked.sort(key=lambda x: x[1], reverse=True)

            for rank, (doc, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {doc} {rank} {score:.6f} method2\n")

            if i % 50 == 0:
                print(f"Reranked {i} queries...")

    print("Method 2 results written to:", FINAL_RESULTS)


def main():
    run_baseline()
    print("Baseline retrieval complete.")
    print("Running bi-encoder reranking...")
    rerank()


if __name__ == "__main__":
    main()
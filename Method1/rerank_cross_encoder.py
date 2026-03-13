import subprocess
import json
from collections import defaultdict
from sentence_transformers.cross_encoder import CrossEncoder


# ----------------------------
# FILE PATHS
# ----------------------------

CORPUS_PATH = "../corpus.jsonl"
QUERIES_PATH = "../queries.jsonl"

BASELINE_RESULTS = "baseline_method1.txt"
FINAL_RESULTS = "Results_method1_bert"

TOP_K = 100

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ----------------------------
# STEP 1: RUN ASSIGNMENT 1
# ----------------------------

def run_baseline():

    print("Running Assignment 1 baseline retrieval...")

    cmd = [
        "python",
        "run_ir.py",
        "--rebuild",
        "--doc_mode",
        "title+text",
        "--run_name",
        "baseline_method1",
        "--results_path",
        "Method1/baseline_method1.txt"
    ]

    subprocess.run(cmd, check=True, cwd="..")


# ----------------------------
# LOAD CORPUS
# ----------------------------

def load_corpus():

    corpus = {}

    with open(CORPUS_PATH, "r", encoding="utf8") as f:
        for line in f:
            doc = json.loads(line)

            doc_id = str(doc["_id"])
            title = doc.get("title", "")
            text = doc.get("text", "")

            corpus[doc_id] = title + " " + text

    return corpus


# ----------------------------
# LOAD QUERIES
# ----------------------------

def load_queries():

    queries = {}

    with open(QUERIES_PATH, "r", encoding="utf8") as f:
        for line in f:
            q = json.loads(line)

            qid = str(q["_id"])

            title = q.get("title", "")
            text = q.get("text", "")

            queries[qid] = title + " " + text

    return queries


# ----------------------------
# LOAD BASELINE RESULTS
# ----------------------------

def load_baseline():

    results = defaultdict(list)

    with open(BASELINE_RESULTS, "r") as f:

        for line in f:

            parts = line.split()

            qid = parts[0]
            doc_id = parts[2]

            if len(results[qid]) < TOP_K:
                results[qid].append(doc_id)

    return results


# ----------------------------
# BERT RERANK
# ----------------------------

def rerank():

    corpus = load_corpus()
    queries = load_queries()
    baseline = load_baseline()

    print("Loading BERT model...")
    model = CrossEncoder(MODEL_NAME)

    with open(FINAL_RESULTS, "w") as out:

        for qid in baseline:

            query = queries[qid]

            pairs = []
            docs = []

            for doc_id in baseline[qid]:

                if doc_id not in corpus:
                    continue

                doc_text = corpus[doc_id]

                pairs.append((query, doc_text))
                docs.append(doc_id)

            scores = model.predict(pairs)

            ranked = list(zip(docs, scores))
            ranked.sort(key=lambda x: x[1], reverse=True)

            for rank, (doc, score) in enumerate(ranked, start=1):

                out.write(f"{qid} Q0 {doc} {rank} {score:.6f} method1\n")


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def main():

    run_baseline()

    print("Baseline retrieval complete.")

    print("Running BERT reranking...")

    rerank()

    print("Method 1 results written to:", FINAL_RESULTS)


if __name__ == "__main__":
    main()
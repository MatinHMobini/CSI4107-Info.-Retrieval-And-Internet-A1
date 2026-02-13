# run_ir.py
import argparse
import json
import os

from preprocessing import preprocess_text
from retrieval import rank_documents
from indexing import build_inverted_index, save_index, load_index, sample_vocabulary


def is_odd_query_id(qid: str) -> bool:
    """Assignment says: only odd-numbered test queries (1,3,5,...)"""
    try:
        return int(qid) % 2 == 1
    except:
        # If ID isn't numeric, we can't apply odd/even; keep it.
        return True


def iter_queries(queries_path: str):
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            qid = str(q.get("_id", "")).strip()
            title = (q.get("title", "") or "").strip()
            text = (q.get("text", "") or "").strip()
            yield qid, title, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--queries", default="queries.jsonl")
    ap.add_argument("--index_path", default="index.pkl")
    ap.add_argument("--results_path", default="Results")
    ap.add_argument("--run_name", default="tfidf_run")
    ap.add_argument("--mode", choices=["title", "title+text"], default="title")
    ap.add_argument("--use_log_tf", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    # Build/load index
    if (not args.rebuild) and os.path.exists(args.index_path):
        index_data = load_index(args.index_path)
    else:
        index_data = build_inverted_index(
            corpus_path=args.corpus,
            preprocess_fn=preprocess_text,
            use_log_tf=args.use_log_tf
        )
        save_index(index_data, args.index_path)

    index = index_data["index"]
    idf = index_data["idf"]
    doc_lengths = index_data["doc_lengths"]

    # Quick sanity prints (useful for README stats)
    print("Indexed docs:", index_data["N_docs"])
    print("Vocab size:", index_data["vocab_size"])
    print("Vocab sample (first 20):", sample_vocabulary(index_data, 20))

    # Run retrieval for odd queries only
    with open(args.results_path, "w", encoding="utf-8") as out:
        for qid, title, text in iter_queries(args.queries):
            if not qid:
                continue
            if not is_odd_query_id(qid):
                continue

            if args.mode == "title":
                query_str = title if title else text  # SciFact may not have titles
            else:
                query_str = (title + " " + text).strip()
                if not query_str:
                    query_str = text

            ranked = rank_documents(
                query=query_str,
                index=index,
                idf=idf,
                preprocess=preprocess_text,
                doc_lengths=doc_lengths,
                top_k=100
            )

            for rank, (doc_id, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {args.run_name}\n")

    print(f"\nWrote results to: {args.results_path}")


if __name__ == "__main__":
    main()

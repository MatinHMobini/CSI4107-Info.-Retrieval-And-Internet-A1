import argparse
from collections import defaultdict

def is_odd_qid(qid: str) -> bool:
    try:
        return int(qid) % 2 == 1
    except ValueError:
        # If qid isn't numeric, don't odd-filter it
        return True

def load_qrels(path: str):
    """
    Supports typical qrels formats like:
      qid 0 docid rel
    Skips header lines like: query-id 0 corpus-id score
    Stores only rel > 0 as relevant.
    Returns: dict[qid] -> set(docid)
    """
    qrels = defaultdict(set)
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # Skip header if present
            if parts[0].lower().startswith("query"):
                continue

            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) == 3:
                qid, docid, rel = parts[0], parts[1], parts[2]
            else:
                continue

            try:
                rel_i = int(float(rel))
            except ValueError:
                continue

            if rel_i > 0:
                qrels[str(qid)].add(str(docid))

    return dict(qrels)

def load_run(path: str):
    """
    Reads run files like:
      qid Q0 docid rank score runname
    Returns: dict[qid] -> list(docid) in rank order
    """
    rows = defaultdict(list)
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            docid = parts[2]
            try:
                rank = int(parts[3])
            except ValueError:
                rank = len(rows[qid]) + 1
            rows[qid].append((rank, docid))

    run = {}
    for qid, lst in rows.items():
        lst.sort(key=lambda x: x[0])
        run[qid] = [docid for _, docid in lst]
    return run

def average_precision(ranked_docs, rel_set):
    if not rel_set:
        return None
    hit = 0
    sum_prec = 0.0
    for i, d in enumerate(ranked_docs, start=1):
        if d in rel_set:
            hit += 1
            sum_prec += hit / i
    return sum_prec / len(rel_set)

def precision_at_k(ranked_docs, rel_set, k=10):
    if not rel_set:
        return None
    topk = ranked_docs[:k]
    return sum(1 for d in topk if d in rel_set) / k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default="qrels.txt")
    ap.add_argument("--run", required=True)

    # IMPORTANT: matches your pipeline behavior (odd queries only)
    ap.add_argument("--odd_only", action="store_true", default=True,
                    help="Evaluate only odd-numbered query IDs (default: True).")
    ap.add_argument("--no_odd_only", action="store_true",
                    help="Disable odd-only filtering and evaluate all qrels query IDs.")

    # Optional: evaluate only queries that appear in the run
    ap.add_argument("--only_run_qids", action="store_true",
                    help="Evaluate only qrels queries that also appear in the run (intersection).")

    args = ap.parse_args()

    odd_only = args.odd_only and not args.no_odd_only

    qrels_all = load_qrels(args.qrels)
    run = load_run(args.run)

    qrels_qids_total = len(qrels_all)

    # Apply odd-only filter to qrels
    if odd_only:
        qrels = {qid: rel for qid, rel in qrels_all.items() if is_odd_qid(qid)}
    else:
        qrels = dict(qrels_all)

    qrels_qids_after_filter = len(qrels)

    # Optionally restrict to intersection with run qids
    if args.only_run_qids:
        qrels = {qid: rel for qid, rel in qrels.items() if qid in run}

    qrels_qids_final = len(qrels)

    aps = []
    p10s = []
    evaluated = 0

    missing_in_run = []

    for qid, rel_set in qrels.items():
        ranked = run.get(qid, [])
        if not ranked:
            missing_in_run.append(qid)
            # If you're evaluating intersection only, this won't happen.
            # Otherwise, treat as zero for both metrics (standard IR evaluation).
            aps.append(0.0)
            p10s.append(0.0)
            evaluated += 1
            continue

        apv = average_precision(ranked, rel_set)
        p10v = precision_at_k(ranked, rel_set, k=10)

        # If no relevant docs exist in qrels for this qid (rare), skip it
        if apv is None or p10v is None:
            continue

        aps.append(apv)
        p10s.append(p10v)
        evaluated += 1

    map_score = sum(aps) / evaluated if evaluated else 0.0
    p10_score = sum(p10s) / evaluated if evaluated else 0.0

    print(f"Run file: {args.run}")
    print(f"Qrels queries total: {qrels_qids_total}")
    if odd_only:
        print(f"Qrels queries after odd-only filter: {qrels_qids_after_filter}")
    else:
        print(f"Qrels queries after filter: {qrels_qids_after_filter}")

    if args.only_run_qids:
        print(f"Qrels queries after intersecting with run: {qrels_qids_final}")

    print(f"Queries evaluated: {evaluated}")

    if missing_in_run:
        # print only a few so output isn't huge
        preview = ", ".join(missing_in_run[:15])
        print(f"Missing queries in run (count={len(missing_in_run)}). First few: {preview}")

    print(f"MAP:  {map_score:.6f}")
    print(f"P@10: {p10_score:.6f}")

if __name__ == "__main__":
    main()
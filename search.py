"""Hybrid BM25 + vector retrieval over the Chroma index, then answer via local LLM."""
import argparse
import logging
import math
import re
import sys
from collections import Counter

import chromadb

from config import CHROMA_PATH, COLLECTION, Settings

log = logging.getLogger("search")


def load_index() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(COLLECTION)
    except chromadb.errors.NotFoundError:
        raise SystemExit(f"Index '{COLLECTION}' missing in {CHROMA_PATH}. Run `make index` first.")


collection = load_index()
all_docs = collection.get(include=["documents", "metadatas"])
docs = all_docs["documents"]
metas = all_docs["metadatas"]

if not docs:
    raise SystemExit("Index is empty. Run `make index` first.")

def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

DOC_TOKENS = [tokens(d) for d in docs]
DF = Counter()
for t in DOC_TOKENS:
    DF.update(set(t))
N = len(DOC_TOKENS)
AVG_DL = sum(len(t) for t in DOC_TOKENS) / N

# ponytail: precomputed df/avgdl, O(N * q) per query; fine at this corpus size,
# upgrade to an inverted index if the collection grows past a few thousand docs.
def bm25_scores(query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    terms = set(tokens(query))
    scores = []
    for t in DOC_TOKENS:
        s = 0.0
        for term in terms:
            df = DF[term]
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf = t.count(term)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / AVG_DL))
        scores.append(s)
    return scores


def retrieve(query: str, top_k: int = 10) -> list[tuple[str, dict, float]]:
    emb = Settings.embed_model.get_query_embedding(query)
    res = collection.query(
        query_embeddings=[emb],
        n_results=20,
        include=["documents", "metadatas", "distances"],
    )
    vec = [(d, m, 1.0 - dist) for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    bm = bm25_scores(query)
    bm_max = max(bm) or 1.0
    keyword = [(d, m, s / bm_max) for d, m, s in zip(docs, metas, bm)]

    merged: dict[str, tuple[dict, float]] = {}
    for d, m, s in vec + keyword:
        if s > merged.get(d, (m, 0.0))[1]:
            merged[d] = (m, s)

    return sorted(((d, m, s) for d, (m, s) in merged.items()), key=lambda x: x[2], reverse=True)[:top_k]


def list_all(entity: str) -> str:
    """Enumerate every matching row: top-k retrieval cannot list all rows."""
    needle = entity.lower()
    rows: dict[str, tuple[int, str]] = {}
    for d, m in zip(docs, metas):
        if needle not in d.lower():
            continue
        for line in d.splitlines():
            line = line.strip()
            if needle not in line.lower():
                continue
            key = re.sub(r"\s+", " ", line)[:60]
            if key not in rows or len(line) > len(rows[key][1]):
                rows[key] = (m.get("page"), line)
    if not rows:
        return ""
    return "\n".join(
        f"[p{page}] {line}" for page, line in sorted(rows.values(), key=lambda x: (x[0], x[1]))
    )


STOP = {
    "list", "all", "the", "a", "an", "of", "to", "from", "and", "in", "by", "for", "with",
    "transactions", "transaction", "items", "entries",
}


def answer(query: str) -> str:
    if re.search(r"\blist all\b", query, re.I):
        for candidate in reversed(re.findall(r"\b([A-Za-z][A-Za-z0-9]+)\b", query)):
            if candidate.lower() in STOP:
                continue
            result = list_all(candidate)
            if result:
                return result

    hits = retrieve(query)
    context = "\n\n".join(f"[page {m['page']}] {d}" for d, m, _ in hits)
    prompt = (
        "Answer only from the context below. Cite the page number for each claim.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    try:
        text = Settings.llm.complete(prompt).text
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e
    if not text:
        raise RuntimeError("LLM returned an empty answer (reasoning-only output; raise max_tokens)")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about the indexed PDFs.")
    parser.add_argument("query", nargs="?", help="One-shot question; omit for interactive mode")
    args = parser.parse_args()

    if args.query:
        try:
            print(answer(args.query))
        except RuntimeError as e:
            log.error("%s", e)
            return 1
        return 0

    print("Ready. Ask questions about the PDFs (type 'exit' to quit)")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            return 0
        if not q:
            continue
        try:
            print("\nAnswer:")
            print(answer(q))
        except RuntimeError as e:
            log.error("%s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())

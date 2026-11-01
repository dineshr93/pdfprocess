"""Runnable self-checks for search.py. Plain asserts, no framework."""
import search

assert search.tokens("Bank BHD-44") == ["bank", "bhd", "44"]

q = "petitioner"
assert search.bm25_scores(q) and max(search.bm25_scores(q)) > 0

hits = search.retrieve(q, top_k=5)
assert hits and all(h[2] >= 0 for h in hits)
assert hits[0][1]["page"] >= 1

rows = search.list_all("DINESH")
assert rows and rows.count("\n") >= 1
assert rows.startswith("[p")

# answer's list-all branch is LLM-free; proves that path without a model call.
assert search.answer("list all DINESH").startswith("[p")
assert search._list_answer("list all zzz") is None

# Unknown entity: list-all yields "", and question falls back to retrieve.
assert search.list_all("zzz") == ""

print("self-check passed")

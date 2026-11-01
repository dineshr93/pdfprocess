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

print("self-check passed")

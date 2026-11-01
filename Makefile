PY := .venv/bin/python

setup:
	test -d .venv || uv venv .venv
	uv pip install -r requirements.txt

index:
	$(PY) process_pdfs.py

run:
	$(PY) search.py

query:
	$(PY) search.py "$(Q)"

test:
	$(PY) test_search.py

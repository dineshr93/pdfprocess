# pdfprocess

Local RAG over PDFs: extracts page text, indexes chunks in Chroma with hybrid
BM25 + vector retrieval, and answers questions with a local LLM. Both the
embedding and LLM servers are OpenAI-compatible endpoints on localhost.

## Requirements

- Python 3.13 and `uv`
- An embedding server at `http://localhost:8001/v1`
- A chat model server at `http://localhost:8888/v1`
- `ocrmypdf` for scanned PDFs (uses Ghostscript)

## Setup

```bash
make setup   # creates .venv with uv and installs requirements.txt
```

Put your PDFs in `pdfs/`, then build the index:

```bash
make index
```

## Usage

```bash
make run                          # interactive Q&A
make query Q="Who is the petitioner?"
make test                         # assert-based self-check
```

The index is rebuilt from scratch on each `make index` run: a staging
collection is built first, then swapped in, so a crash never leaves a
half-built index.

## Notes

Expected chunks are stored in `chroma_db/`; that directory, `pdfs/`, and
`.venv/` are not tracked by git.

`list all <entity>` enumerates lines mentioning an entity, but it is a text
matcher, not a table parser — for transaction tables it falls back to
retrieval rather than listing every row.

## License

MIT — see `LICENSE`.

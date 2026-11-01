"""Build the Chroma index from PDFs. Exit non-zero on any failure."""
from pathlib import Path
import logging
import os
import subprocess
import sys
import tempfile

import chromadb
import pymupdf
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from config import CHROMA_PATH, COLLECTION, PDF_DIR, Settings

log = logging.getLogger("index")


def extract_pages(pdf_path: Path) -> list[Document]:
    """Page-level text. OCR only when the PDF has no text layer."""
    pages: list[Document] = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc, 1):
            text = page.get_text().strip()
            if text:
                pages.append(Document(text=text, metadata={"file_name": pdf_path.name, "page": i}))
    finally:
        doc.close()

    if pages:
        return pages

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ocrmypdf", "--skip-text", "--deskew", "--clean", "--quiet", str(pdf_path), tmp_path],
            check=True,
        )
        doc = pymupdf.open(tmp_path)
        try:
            for i, page in enumerate(doc, 1):
                text = page.get_text().strip()
                if text:
                    pages.append(Document(text=text, metadata={"file_name": pdf_path.name, "page": i}))
        finally:
            doc.close()
    finally:
        os.unlink(tmp_path)
    return pages


def main() -> int:
    pdfs = sorted(Path(PDF_DIR).glob("*.pdf"))
    if not pdfs:
        log.error("No PDFs in %s", PDF_DIR)
        return 1

    documents: list[Document] = []
    failed = 0
    for pdf in tqdm(pdfs, desc="Processing"):
        try:
            documents.extend(extract_pages(pdf))
        except Exception as e:  # one bad PDF must not abort the whole run
            log.error("Failed %s: %s", pdf.name, e)
            failed += 1

    if not documents:
        log.error("No text extracted from any PDF")
        return 1

    staging = COLLECTION + "_staging"
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    for name in (staging,):
        try:
            client.delete_collection(name)
        except chromadb.errors.NotFoundError:
            pass

    collection = client.get_or_create_collection(staging)
    try:
        VectorStoreIndex.from_documents(
            documents,
            storage_context=StorageContext.from_defaults(
                vector_store=ChromaVectorStore(chroma_collection=collection)
            ),
            show_progress=True,
        )
        # Swap only after the new index is fully built: a crash leaves staging intact.
        try:
            client.delete_collection(COLLECTION)
        except chromadb.errors.NotFoundError:
            pass
        collection.modify(name=COLLECTION)
    except Exception:
        client.delete_collection(staging)
        raise

    log.info("Index ready: %s (%d chunks, %d pages)", CHROMA_PATH, collection.count(), len(documents))
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())

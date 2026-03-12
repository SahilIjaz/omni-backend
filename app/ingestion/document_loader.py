# app/ingestion/document_loader.py
import requests
from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader
from bs4 import BeautifulSoup


@dataclass
class Document:
    """
    Core data structure representing a loaded document.
    
    WHY a dataclass?
    Every chunk, every embedding, every vector DB entry
    traces back to a source document. We need metadata
    (where did this come from?) for citations in answers.
    """
    content: str          # Raw text
    source: str           # File path or URL
    doc_type: str         # 'pdf', 'web', 'text'
    metadata: dict        # Page numbers, titles, etc.


class DocumentLoader:
    """Loads documents from multiple sources into a unified format."""

    def load_pdf(self, file_path: str) -> Document:
        """
        Load text from a PDF file.
        
        KEY CONCEPT: PDFs are complex — they store text as positioned
        characters, not flowing paragraphs. PyPDF extracts it page by
        page. We track page numbers for citations later.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        reader = PdfReader(file_path)
        pages_text = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():          # Skip blank pages
                pages_text.append({
                    "page": page_num + 1,
                    "text": text.strip()
                })

        # Join all pages — we'll chunk later
        full_text = "\n\n".join([p["text"] for p in pages_text])

        return Document(
            content=full_text,
            source=file_path,
            doc_type="pdf",
            metadata={
                "filename": path.name,
                "total_pages": len(reader.pages),
                "pages": pages_text          # Keep per-page data for citations
            }
        )

    def load_web(self, url: str) -> Document:
        """
        Scrape and clean text from a webpage.
        
        KEY CONCEPT: Raw HTML is full of noise (nav bars, footers, ads).
        BeautifulSoup lets us extract only the meaningful content.
        We remove script/style tags that would pollute our chunks.
        """
        headers = {"User-Agent": "Mozilla/5.0 (RAG Knowledge Agent)"}

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "advertisement"]):
            tag.decompose()

        # Extract title
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else url

        # Get main content — prefer article/main tags, fallback to body
        main_content = (
            soup.find("article") or
            soup.find("main") or
            soup.find("div", class_=lambda c: c and "content" in c.lower()) or
            soup.body
        )

        text = main_content.get_text(separator="\n", strip=True) if main_content else ""

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_text = "\n".join(lines)

        return Document(
            content=clean_text,
            source=url,
            doc_type="web",
            metadata={
                "title": title_text,
                "url": url,
                "char_count": len(clean_text)
            }
        )

    def load_text(self, file_path: str) -> Document:
        """Load plain text or markdown files."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        return Document(
            content=content,
            source=file_path,
            doc_type="text",
            metadata={"filename": path.name}
        )

    def load_many(self, sources: list[dict]) -> list[Document]:
        """
        Load multiple sources at once.
        
        sources = [
            {"type": "pdf",  "path": "docs/manual.pdf"},
            {"type": "web",  "url": "https://docs.python.org"},
            {"type": "text", "path": "docs/notes.txt"}
        ]
        """
        documents = []
        for source in sources:
            try:
                if source["type"] == "pdf":
                    doc = self.load_pdf(source["path"])
                elif source["type"] == "web":
                    doc = self.load_web(source["url"])
                elif source["type"] == "text":
                    doc = self.load_text(source["path"])
                else:
                    print(f"⚠️  Unknown type: {source['type']}")
                    continue

                documents.append(doc)
                print(f"✅ Loaded: {source.get('path') or source.get('url')}")

            except Exception as e:
                print(f"❌ Failed to load {source}: {e}")

        return documents
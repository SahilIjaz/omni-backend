# app/ingestion/chunker.py
from dataclasses import dataclass, field
from app.ingestion.document_loader import Document


@dataclass
class Chunk:
    """
    A single chunk — the atomic unit of your RAG system.
    Everything downstream (embeddings, vector DB, retrieval)
    works with Chunks, not raw Documents.
    """
    content: str            # The actual text
    chunk_id: str           # Unique ID for vector DB
    source: str             # Original document source
    doc_type: str           # pdf / web / text
    chunk_index: int        # Position in original document
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.content)


class TextChunker:
    """
    Implements multiple chunking strategies.
    
    RULE OF THUMB for chunk sizes:
    - Too small (< 100 chars):  loses context, poor retrieval
    - Too large (> 1500 chars): retrieves too much noise
    - Sweet spot: 300–800 chars with 50–100 char overlap
    
    Overlap is critical — it prevents losing information
    at chunk boundaries (like cutting a sentence in half).
    """

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ─────────────────────────────────────────────
    # Strategy 1: Fixed Size (baseline, rarely best)
    # ─────────────────────────────────────────────
    def fixed_size_split(self, document: Document) -> list[Chunk]:
        """
        Split every N characters regardless of content.
        
        WHEN TO USE: Quick prototyping only.
        PROBLEM: Cuts mid-sentence, destroying semantic meaning.
        """
        text = document.content
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_id=f"{document.source}::fixed::{idx}",
                    source=document.source,
                    doc_type=document.doc_type,
                    chunk_index=idx,
                    metadata={**document.metadata, "strategy": "fixed_size"}
                ))
                idx += 1

            start = end - self.chunk_overlap   # Overlap slides window back

        return chunks

    # ─────────────────────────────────────────────────────
    # Strategy 2: Recursive Split (best general-purpose ✅)
    # ─────────────────────────────────────────────────────
    def recursive_split(self, document: Document) -> list[Chunk]:
        """
        Split by hierarchy: paragraphs → sentences → words → chars.
        
        WHEN TO USE: Most documents — this is your default.
        WHY IT WORKS: Text has natural boundaries. This respects them
        by trying the largest separator first, then falling back.
        
        Separators tried in order:
        1. Double newline (paragraph break)
        2. Single newline  
        3. Period + space (sentence end)
        4. Single space (word boundary)
        5. Individual characters (last resort)
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        raw_chunks = self._recursive_split_text(
            document.content, separators, self.chunk_size
        )

        # Apply overlap and build Chunk objects
        chunks = []
        for idx, text in enumerate(raw_chunks):
            # Add tail of previous chunk as prefix (overlap)
            if idx > 0 and self.chunk_overlap > 0:
                prev = raw_chunks[idx - 1]
                overlap_text = prev[-self.chunk_overlap:].strip()
                text = overlap_text + " " + text

            text = text.strip()
            if len(text) > 50:                # Skip tiny fragments
                chunks.append(Chunk(
                    content=text,
                    chunk_id=f"{document.source}::recursive::{idx}",
                    source=document.source,
                    doc_type=document.doc_type,
                    chunk_index=idx,
                    metadata={**document.metadata, "strategy": "recursive"}
                ))

        return chunks

    def _recursive_split_text(self,
                               text: str,
                               separators: list[str],
                               max_size: int) -> list[str]:
        """Internal recursive helper."""
        if not separators:
            return [text[i:i+max_size] for i in range(0, len(text), max_size)]

        separator = separators[0]
        splits = text.split(separator) if separator else list(text)

        chunks = []
        current = ""

        for split in splits:
            candidate = current + (separator if current else "") + split

            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If single split is too large, recurse with next separator
                if len(split) > max_size:
                    sub = self._recursive_split_text(split, separators[1:], max_size)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = split

        if current:
            chunks.append(current)

        return chunks

    # ─────────────────────────────────────────────────────────
    # Strategy 3: Document-Aware Split (best for PDFs/Markdown)
    # ─────────────────────────────────────────────────────────
    def document_aware_split(self, document: Document) -> list[Chunk]:
        """
        Split based on document structure markers.
        
        WHEN TO USE: PDFs with clear sections, Markdown docs,
        technical documentation with headers.
        
        WHY IT WORKS: Headers signal topic changes. Keeping content
        under its header preserves semantic grouping naturally.
        """
        import re

        # Detect structure markers
        header_pattern = re.compile(
            r'^(#{1,6}\s+.+|[A-Z][A-Z\s]{3,}:?\s*$|\d+\.\s+[A-Z].+)',
            re.MULTILINE
        )

        text = document.content
        sections = []
        last_end = 0
        current_header = "Introduction"

        for match in header_pattern.finditer(text):
            if match.start() > last_end:
                section_text = text[last_end:match.start()].strip()
                if section_text:
                    sections.append((current_header, section_text))

            current_header = match.group().strip()
            last_end = match.end()

        # Don't forget the last section
        if last_end < len(text):
            sections.append((current_header, text[last_end:].strip()))

        # Now chunk each section (sections can still be large)
        chunks = []
        global_idx = 0

        for header, section_text in sections:
            if len(section_text) <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(Chunk(
                    content=f"{header}\n{section_text}",
                    chunk_id=f"{document.source}::aware::{global_idx}",
                    source=document.source,
                    doc_type=document.doc_type,
                    chunk_index=global_idx,
                    metadata={
                        **document.metadata,
                        "strategy": "document_aware",
                        "section_header": header
                    }
                ))
                global_idx += 1
            else:
                # Section too large — recursively split it
                sub_doc = Document(
                    content=section_text,
                    source=document.source,
                    doc_type=document.doc_type,
                    metadata=document.metadata
                )
                sub_chunks = self.recursive_split(sub_doc)
                for sub in sub_chunks:
                    sub.content = f"{header}\n{sub.content}"
                    sub.chunk_id = f"{document.source}::aware::{global_idx}"
                    sub.chunk_index = global_idx
                    sub.metadata["section_header"] = header
                    chunks.append(sub)
                    global_idx += 1

        return chunks

    # ─────────────────────────────────────────────
    # Smart dispatcher — picks strategy automatically
    # ─────────────────────────────────────────────
    def chunk(self, document: Document, strategy: str = "auto") -> list[Chunk]:
        """
        Main entry point. Choose strategy or let it auto-select.
        
        AUTO LOGIC:
        - PDF with many pages       → document_aware (respects structure)
        - Web page                  → recursive (unstructured HTML text)
        - Markdown / structured     → document_aware
        - Everything else           → recursive (safe default)
        """
        if strategy == "auto":
            if document.doc_type == "pdf":
                strategy = "document_aware"
            elif document.doc_type == "web":
                strategy = "recursive"
            else:
                strategy = "recursive"

        if strategy == "fixed":
            return self.fixed_size_split(document)
        elif strategy == "recursive":
            return self.recursive_split(document)
        elif strategy == "document_aware":
            return self.document_aware_split(document)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
import os
from typing import List, Dict, Any
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None


class DocumentChunker:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from file (.txt, .md, .pdf, .docx).
        Returns a list of dicts: [{"text": str, "metadata": dict}]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        file_name = path.name

        if ext in [".txt", ".md", ".py", ".json", ".csv"]:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [{"text": content, "metadata": {"source": file_name, "page": 1}}]

        elif ext == ".pdf":
            if PdfReader is None:
                raise ImportError("pypdf is required to parse PDF files. Run `pip install pypdf`.")
            reader = PdfReader(str(path))
            pages_data = []
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_data.append({
                        "text": text,
                        "metadata": {"source": file_name, "page": idx + 1}
                    })
            return pages_data

        elif ext in [".docx", ".doc"]:
            if docx is None:
                raise ImportError("python-docx is required to parse DOCX files. Run `pip install python-docx`.")
            doc = docx.Document(str(path))
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return [{"text": full_text, "metadata": {"source": file_name, "page": 1}}]

        else:
            # Fallback to plain text read
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [{"text": content, "metadata": {"source": file_name, "page": 1}}]

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Recursive character text splitting based on paragraphs, sentences, and words.
        """
        if not text:
            return []

        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        
        def _split(s: str, seps: List[str]) -> List[str]:
            if not seps:
                return [s]
            
            sep = seps[0]
            remaining_seps = seps[1:]
            
            if sep:
                parts = s.split(sep)
            else:
                parts = list(s)
                
            chunks = []
            current = ""
            
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    if len(part) > self.chunk_size and remaining_seps:
                        sub_chunks = _split(part, remaining_seps)
                        for sc in sub_chunks:
                            if len(sc) > self.chunk_size:
                                chunks.append(sc[:self.chunk_size])
                            else:
                                chunks.append(sc)
                        current = ""
                    else:
                        current = part
            if current and current.strip():
                chunks.append(current.strip())
            return chunks

        # Do primary split
        raw_chunks = _split(text, separators)
        
        # Apply sliding window overlap
        final_chunks = []
        for i, ch in enumerate(raw_chunks):
            if i > 0 and self.chunk_overlap > 0:
                prev_overlap = raw_chunks[i-1][-self.chunk_overlap:]
                combined = prev_overlap + " ... " + ch
                final_chunks.append(combined)
            else:
                final_chunks.append(ch)
                
        return [c for c in final_chunks if len(c.strip()) > 20]

    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract and chunk document, returning list of chunk items with metadata.
        """
        pages = self.extract_text(file_path)
        all_chunks = []
        chunk_idx = 0
        for p in pages:
            text = p["text"]
            meta = p["metadata"]
            chunks = self.split_text_into_chunks(text)
            for c in chunks:
                chunk_idx += 1
                all_chunks.append({
                    "text": c,
                    "metadata": {
                        "source": meta["source"],
                        "page": meta.get("page", 1),
                        "chunk_id": f"{meta['source']}_chunk_{chunk_idx}",
                        "chunk_index": chunk_idx
                    }
                })
        return all_chunks

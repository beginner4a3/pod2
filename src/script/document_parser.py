"""
Document Parser - Extract text from PDF, DOCX, and TXT files

Supports:
- PDF: Using PyPDF2
- DOCX: Using python-docx
- TXT: Direct file read
"""

import os
from pathlib import Path
from typing import Optional, Tuple


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text.strip())
        
        return "\n\n".join(text_parts)
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF parsing. Install with: pip install PyPDF2")
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text content
    """
    try:
        from docx import Document
        
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        
        return "\n\n".join(paragraphs)
    except ImportError:
        raise ImportError("python-docx is required for DOCX parsing. Install with: pip install python-docx")
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {e}")


def extract_text_from_txt(file_path: str, encoding: str = "utf-8") -> str:
    """
    Extract text from a TXT file.
    
    Args:
        file_path: Path to the TXT file
        encoding: File encoding (default: utf-8)
        
    Returns:
        Text content
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"Failed to read TXT file: {e}")


def extract_text(file_path: str) -> Tuple[str, str]:
    """
    Extract text from a file (auto-detect format).
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (extracted_text, file_type)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = path.suffix.lower()
    
    if extension == ".pdf":
        return extract_text_from_pdf(file_path), "pdf"
    elif extension == ".docx":
        return extract_text_from_docx(file_path), "docx"
    elif extension == ".txt":
        return extract_text_from_txt(file_path), "txt"
    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported: .pdf, .docx, .txt")


def clean_text(text: str) -> str:
    """
    Clean extracted text for LLM processing.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Trim
    text = text.strip()
    
    return text


def parse_document(file_path: str) -> dict:
    """
    Parse a document and return structured result.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Dictionary with text, file_type, word_count, etc.
    """
    text, file_type = extract_text(file_path)
    cleaned = clean_text(text)
    
    words = cleaned.split()
    
    return {
        "text": cleaned,
        "file_type": file_type,
        "word_count": len(words),
        "char_count": len(cleaned),
        "file_name": Path(file_path).name,
    }


if __name__ == "__main__":
    print("Document Parser ready.")
    print("Supported formats: PDF, DOCX, TXT")
    print("Usage: parse_document('path/to/file.pdf')")

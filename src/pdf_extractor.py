# src/pdf_extractor.py
# PDF text extraction with support for multiple methods.
#   extract_text_from_pdf  – main entry point, delegates to appropriate method
#   extract_with_pymupdf   – fast extraction using PyMuPDF
#   extract_with_pdfplumber – better for tables and structured content
#   extract_with_ocr       – OCR for scanned/image PDFs

from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from .utils.cli_utils import print_warning, spinner, progress_iter


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails."""
    pass


def extract_text_from_pdf(
    pdf_path: Path,
    method: str = "pymupdf",
    ocr_enabled: bool = False,
    ocr_language: str = "eng",
    extract_images: bool = False,
    extract_tables: bool = False,
) -> List[Dict[str, Any]]:
    """Extract text from PDF using specified method.
    
    Args:
        pdf_path: Path to PDF file
        method: Extraction method ('pymupdf', 'pdfplumber', 'ocr')
        ocr_enabled: Enable OCR fallback for scanned pages
        ocr_language: Tesseract language code
        extract_images: Extract embedded images
        extract_tables: Extract tables separately
    
    Returns:
        List of page dictionaries with text and metadata
        
    Raises:
        PDFExtractionError: If extraction fails
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Choose extraction method
    if method == "pymupdf":
        return extract_with_pymupdf(
            pdf_path,
            ocr_fallback=ocr_enabled,
            ocr_language=ocr_language,
            extract_images=extract_images,
        )
    elif method == "pdfplumber":
        if not PDFPLUMBER_AVAILABLE:
            print_warning("pdfplumber not installed, falling back to pymupdf")
            return extract_with_pymupdf(pdf_path)
        return extract_with_pdfplumber(
            pdf_path,
            extract_tables=extract_tables,
        )
    elif method == "ocr":
        if not OCR_AVAILABLE:
            raise PDFExtractionError(
                "OCR not available. Install: uv add pytesseract pdf2image"
            )
        return extract_with_ocr(pdf_path, language=ocr_language)
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def extract_with_pymupdf(
    pdf_path: Path,
    ocr_fallback: bool = False,
    ocr_language: str = "eng",
    extract_images: bool = False,
) -> List[Dict[str, Any]]:
    """Extract text using PyMuPDF (fastest method).
    
    Args:
        pdf_path: Path to PDF file
        ocr_fallback: Use OCR if page has no text
        ocr_language: Tesseract language
        extract_images: Extract embedded images
    
    Returns:
        List of page data dictionaries
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFExtractionError(f"Failed to open PDF: {e}")
    
    pages = []

    for page_num in progress_iter(range(len(doc)), total=len(doc), desc="Extracting pages (pymupdf)"):
        page = doc[page_num]

        text = page.get_text()
        
        # OCR fallback for scanned pages
        if ocr_fallback and not text.strip() and OCR_AVAILABLE:
            text = _ocr_page_pymupdf(page, ocr_language)
        
        page_data = {
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text),
            "metadata": {
                "width": page.rect.width,
                "height": page.rect.height,
            }
        }
        
        # Extract images if requested
        if extract_images:
            images = []
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                images.append({
                    "index": img_index,
                    "xref": xref,
                })
            page_data["images"] = images
        
        pages.append(page_data)
    
    doc.close()
    return pages


def extract_with_pdfplumber(
    pdf_path: Path,
    extract_tables: bool = False,
) -> List[Dict[str, Any]]:
    """Extract text using pdfplumber (better for tables).
    
    Args:
        pdf_path: Path to PDF file
        extract_tables: Extract tables separately
    
    Returns:
        List of page data dictionaries
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []

            for page_num, page in progress_iter(enumerate(pdf.pages), total=len(pdf.pages), desc="Extracting pages (pdfplumber)"):
                text = page.extract_text() or ""
                
                page_data = {
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "width": page.width,
                        "height": page.height,
                    }
                }
                
                # Extract tables if requested
                if extract_tables:
                    tables = page.extract_tables()
                    if tables:
                        page_data["tables"] = tables
                
                pages.append(page_data)
            
            return pages
    except Exception as e:
        raise PDFExtractionError(f"pdfplumber extraction failed: {e}")


def extract_with_ocr(
    pdf_path: Path,
    language: str = "eng",
    dpi: int = 300,
) -> List[Dict[str, Any]]:
    """Extract text using OCR (for scanned PDFs).
    
    Args:
        pdf_path: Path to PDF file
        language: Tesseract language code
        dpi: Image DPI for conversion
    
    Returns:
        List of page data dictionaries
    """
    try:
        # Convert PDF to images
        with spinner(f"Converting PDF to images (DPI={dpi})..."):
            images = convert_from_path(str(pdf_path), dpi=dpi)
        
        pages = []
        
        for page_num, image in progress_iter(enumerate(images), total=len(images), desc="OCR pages"):
                # Run OCR
                text = pytesseract.image_to_string(image, lang=language)
                
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "ocr": True,
                        "language": language,
                        "dpi": dpi,
                    }
                })
        
        return pages
    except Exception as e:
        raise PDFExtractionError(f"OCR extraction failed: {e}")


def _ocr_page_pymupdf(page, language: str = "eng") -> str:
    """OCR a single PyMuPDF page."""
    try:
        # Render page as image
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        
        # Convert to PIL Image for Tesseract
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(img_bytes))
        
        # Run OCR
        text = pytesseract.image_to_string(image, lang=language)
        return text
    except Exception:
        return ""


def get_pdf_info(pdf_path: Path) -> Dict[str, Any]:
    """Get PDF metadata and information.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Dictionary with PDF metadata
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        info = {
            "filename": pdf_path.name,
            "pages": len(doc),
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
        }
        
        doc.close()
        return info
    except Exception as e:
        raise PDFExtractionError(f"Failed to get PDF info: {e}")

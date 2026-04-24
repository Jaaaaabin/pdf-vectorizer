# src/pipeline.py
# Main PDF vectorization pipeline.
#   PDFVectorizationPipeline – orchestrates extraction, chunking, and vectorization
#   process_single_pdf       – process one PDF file
#   process_directory        – batch process all PDFs in a directory

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .configuration import (
    load_config,
    get_paths_config,
    get_extraction_config,
    get_chunking_config,
    get_vectorization_config,
)
from .pdf_extractor import extract_text_from_pdf, get_pdf_info
from .text_chunker import chunk_text, get_chunk_statistics
from .vectorizer import vectorize_chunks, save_embeddings
from .utils import (
    list_pdf_files,
    get_output_path,
    print_info,
    print_success,
    print_error,
    print_warning,
    progress_iter,
    format_size,
    get_file_size,
)


class PDFVectorizationPipeline:
    """Complete PDF vectorization pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize pipeline with configuration.
        
        Args:
            config_path: Path to config file (uses default if None)
        """
        self.config = load_config(config_path)
        self.paths = get_paths_config(self.config)
        self.extraction_config = get_extraction_config(self.config)
        self.chunking_config = get_chunking_config(self.config)
        self.vectorization_config = get_vectorization_config(self.config)
        
        # Ensure output directories exist
        self.paths["processed"].mkdir(parents=True, exist_ok=True)
    
    def process_pdf(
        self,
        pdf_path: Path,
        save_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """Process a single PDF through the full pipeline.
        
        Args:
            pdf_path: Path to PDF file
            save_intermediate: Save intermediate results (chunks, text)
        
        Returns:
            Dictionary with processing results
        """
        print_info(f"Processing: {pdf_path.name}")
        
        results = {
            "pdf_path": str(pdf_path),
            "filename": pdf_path.name,
            "status": "processing",
        }
        
        try:
            # Step 1: Extract text
            print_info("Step 1/3: Extracting text from PDF...")
            pages = extract_text_from_pdf(
                pdf_path,
                method=self.extraction_config.get("method", "pymupdf"),
                ocr_enabled=self.extraction_config.get("ocr_enabled", False),
                ocr_language=self.extraction_config.get("ocr_language", "eng"),
                extract_images=self.extraction_config.get("extract_images", False),
                extract_tables=self.extraction_config.get("extract_tables", False),
            )
            
            total_chars = sum(page.get("char_count", 0) for page in pages)
            print_info(f"  Extracted {len(pages)} pages, {total_chars:,} characters")
            
            results["pages_extracted"] = len(pages)
            results["total_chars"] = total_chars
            
            # Save extracted text if requested
            if save_intermediate:
                text_path = get_output_path(
                    pdf_path,
                    self.paths["processed"] / "extracted_text",
                    suffix="_text",
                    extension="json",
                )
                text_path.parent.mkdir(parents=True, exist_ok=True)
                
                with text_path.open("w", encoding="utf-8") as f:
                    json.dump(pages, f, indent=2, ensure_ascii=False)
                
                results["text_file"] = str(text_path)
            
            # Step 2: Chunk text
            print_info("Step 2/3: Chunking text...")
            chunks = chunk_text(
                pages,
                strategy=self.chunking_config.get("strategy", "recursive"),
                chunk_size=self.chunking_config.get("chunk_size", 1000),
                chunk_overlap=self.chunking_config.get("chunk_overlap", 200),
                separators=self.chunking_config.get("separators"),
            )
            
            chunk_stats = get_chunk_statistics(chunks)
            print_info(f"  Created {chunk_stats['total_chunks']} chunks")
            print_info(f"  Avg length: {chunk_stats['avg_length']:.0f} chars")
            
            results["chunks_created"] = chunk_stats["total_chunks"]
            results["chunk_stats"] = chunk_stats
            
            # Save chunks if requested
            if save_intermediate:
                chunks_path = get_output_path(
                    pdf_path,
                    self.paths["processed"] / "chunks",
                    suffix="_chunks",
                    extension="json",
                )
                chunks_path.parent.mkdir(parents=True, exist_ok=True)
                
                with chunks_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        [chunk.to_dict() for chunk in chunks],
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                
                results["chunks_file"] = str(chunks_path)
            
            # Step 3: Vectorize
            print_info("Step 3/3: Generating embeddings...")
            embeddings = vectorize_chunks(
                chunks,
                model_type=self.vectorization_config.get("model_type", "sentence_transformers"),
                model_name=self.vectorization_config.get("model_name", "all-MiniLM-L6-v2"),
                batch_size=self.vectorization_config.get("batch_size", 32),
                openai_api_key=self.vectorization_config.get("openai_api_key"),
            )
            
            print_info(f"  Generated {len(embeddings)} embeddings")
            print_info(f"  Embedding dimension: {embeddings.shape[1] if len(embeddings) > 0 else 0}")
            
            results["embeddings_count"] = len(embeddings)
            results["embedding_dim"] = embeddings.shape[1] if len(embeddings) > 0 else 0
            
            # Save embeddings
            output_format = self.vectorization_config.get("output_format", "numpy")
            embeddings_path = get_output_path(
                pdf_path,
                self.paths["processed"] / "embeddings",
                suffix="_embeddings",
                extension="npz" if output_format == "numpy" else "json",
            )
            
            save_embeddings(
                embeddings,
                chunks,
                embeddings_path,
                format=output_format,
                include_metadata=self.vectorization_config.get("include_metadata", True),
            )
            
            results["embeddings_file"] = str(embeddings_path)
            results["status"] = "success"
            
            print_success(f"Successfully processed {pdf_path.name}")
            
        except Exception as e:
            print_error(f"Failed to process {pdf_path.name}: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def process_directory(
        self,
        input_dir: Optional[Path] = None,
        recursive: bool = False,
        save_intermediate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process all PDFs in a directory.
        
        Args:
            input_dir: Input directory (uses config default if None)
            recursive: Search subdirectories
            save_intermediate: Save intermediate results
        
        Returns:
            List of result dictionaries
        """
        input_dir = input_dir or self.paths["raw"]
        
        print_info(f"Scanning for PDFs in: {input_dir}")
        pdf_files = list_pdf_files(input_dir, recursive=recursive)
        
        if not pdf_files:
            print_warning(f"No PDF files found in {input_dir}")
            return []
        
        print_info(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        results = []
        for pdf_path in progress_iter(pdf_files, desc="Processing PDFs"):
            result = self.process_pdf(pdf_path, save_intermediate=save_intermediate)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        
        print_info("\n=== Processing Summary ===")
        print_success(f"Successful: {successful}/{len(results)}")
        if failed > 0:
            print_error(f"Failed: {failed}/{len(results)}")
        
        # Save summary
        summary_path = self.paths["processed"] / "processing_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print_info(f"Summary saved to: {summary_path}")
        
        return results


def process_single_pdf(
    pdf_path: Path,
    config_path: Optional[Path] = None,
    save_intermediate: bool = True,
) -> Dict[str, Any]:
    """Convenience function to process a single PDF.
    
    Args:
        pdf_path: Path to PDF file
        config_path: Path to config file
        save_intermediate: Save intermediate results
    
    Returns:
        Processing results dictionary
    """
    pipeline = PDFVectorizationPipeline(config_path)
    return pipeline.process_pdf(pdf_path, save_intermediate)


def process_directory(
    input_dir: Path,
    config_path: Optional[Path] = None,
    recursive: bool = False,
    save_intermediate: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience function to process directory of PDFs.
    
    Args:
        input_dir: Input directory path
        config_path: Path to config file
        recursive: Search subdirectories
        save_intermediate: Save intermediate results
    
    Returns:
        List of processing results
    """
    pipeline = PDFVectorizationPipeline(config_path)
    return pipeline.process_directory(input_dir, recursive, save_intermediate)

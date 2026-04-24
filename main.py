#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse

from src.utils import print_info, print_success, print_error


def main():
    parser = argparse.ArgumentParser(
        description="PDF Text Extraction, Chunking, and Vectorization Pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("check", help="Verify imports and config")
    p.add_argument("--config", type=Path)

    p = sub.add_parser("extract", help="Extract text from a PDF (no chunking/embedding)")
    p.add_argument("input", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("-o", "--output", type=Path, help="Override output path")

    p = sub.add_parser("process", help="Full pipeline: extract → chunk → embed")
    p.add_argument("input", type=Path, help="PDF file or directory")
    p.add_argument("--config", type=Path)
    p.add_argument("--recursive", "-r", action="store_true")

    p = sub.add_parser("init", help="Create data directories")
    p.add_argument("--config", type=Path)

    p = sub.add_parser("info", help="Show current configuration")
    p.add_argument("--config", type=Path)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    try:
        return {
            "check":   run_check,
            "extract": run_extract,
            "process": run_process,
            "init":    run_init,
            "info":    run_info,
        }[args.command](args)
    except Exception as e:
        print_error(f"Error: {e}")
        return 1


def run_check(args):
    from src.configuration import load_config, get_paths_config
    print_success("  src.configuration  OK")

    from src.pdf_extractor import extract_text_from_pdf  # noqa: F401
    print_success("  src.pdf_extractor  OK")

    print_info("\nConfig paths:")
    config = load_config(args.config)
    for name, path in get_paths_config(config).items():
        print_info(f"  {name}: {path}")

    print_success("\nAll checks passed.")
    return 0


def run_extract(args):
    import json
    from src.configuration import load_config, get_extraction_config, get_paths_config
    from src.pdf_extractor import extract_text_from_pdf, get_pdf_info

    pdf_path = args.input
    if not pdf_path.exists():
        print_error(f"File not found: {pdf_path}")
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print_error("Input must be a .pdf file")
        return 1

    config = load_config(args.config)
    ext_cfg = get_extraction_config(config)

    info = get_pdf_info(pdf_path)
    print_info(f"File  : {pdf_path.name}")
    print_info(f"Pages : {info.get('page_count', '?')}")
    print_info(f"Method: {ext_cfg.get('method', 'pymupdf')}")

    pages = extract_text_from_pdf(
        pdf_path,
        method=ext_cfg.get("method", "pymupdf"),
        ocr_enabled=ext_cfg.get("ocr_enabled", False),
    )

    total_chars = sum(len(p.get("text", "")) for p in pages)
    print_success(f"Extracted {len(pages)} pages, {total_chars:,} characters")

    out_path = args.output or (
        Path(get_paths_config(config).get("processed", "data/processed"))
        / pdf_path.stem / "text.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
    print_info(f"Saved to: {out_path}")
    return 0


def run_process(args):
    from src.pipeline import PDFVectorizationPipeline

    input_path = args.input
    if not input_path.exists():
        print_error(f"Path not found: {input_path}")
        return 1

    pipeline = PDFVectorizationPipeline(args.config)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print_error("Input must be a PDF file")
            return 1
        result = pipeline.process_pdf(input_path)
        if result["status"] == "success":
            print_success("Processing complete!")
            print_info(f"Embeddings: {result['embeddings_file']}")
            return 0
        print_error(f"Failed: {result.get('error', 'unknown error')}")
        return 1

    if input_path.is_dir():
        results = pipeline.process_directory(input_path, recursive=args.recursive)
        ok = sum(1 for r in results if r["status"] == "success")
        return 0 if ok == len(results) else 1

    print_error(f"Invalid path: {input_path}")
    return 1


def run_init(args):
    from src.configuration import load_config, ensure_paths_exist
    load_config(args.config)
    for name, path in ensure_paths_exist().items():
        print_info(f"  {name}: {path}")
    print_success("Directories ready.")
    return 0


def run_info(args):
    from src.configuration import load_config, get_paths_config, get_extraction_config, get_chunking_config, get_vectorization_config
    config = load_config(args.config)
    sections = {
        "Paths":         get_paths_config(config),
        "Extraction":    get_extraction_config(config),
        "Chunking":      get_chunking_config(config),
        "Vectorization": get_vectorization_config(config),
    }
    for title, data in sections.items():
        print_info(f"\n{title}:")
        for k, v in data.items():
            print_info(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

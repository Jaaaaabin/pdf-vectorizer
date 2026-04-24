# PDF Vectorization Pipeline

A Python project for extracting text from PDFs, chunking it intelligently, and generating embeddings for semantic search and RAG applications.

## Features

- **Multi-method PDF extraction**: PyMuPDF (fast), pdfplumber (tables), OCR (scanned documents)
- **Smart text chunking**: Recursive character splitting with configurable overlap
- **Multiple embedding options**: Local (sentence-transformers) or cloud (OpenAI)
- **Batch processing**: Process entire directories of PDFs
- **Progress tracking**: Rich progress bars and colored console output
- **Flexible configuration**: YAML config with environment variable support
- **Multiple output formats**: NumPy, JSON, JSONL

## Project Structure

```
pdf-vectorizer/
├── config/
│   └── config.yaml          # Pipeline configuration
├── data/
│   ├── raw/                 # Input PDFs (place your PDFs here)
│   └── processed/           # Output directory
│       ├── extracted_text/  # Raw extracted text (JSON)
│       ├── chunks/          # Text chunks (JSON)
│       └── embeddings/      # Vector embeddings (NPZ/JSON)
├── src/
│   ├── configuration.py     # Config loader with env-var substitution
│   ├── pdf_extractor.py     # PDF text extraction
│   ├── text_chunker.py      # Text chunking strategies
│   ├── vectorizer.py        # Embedding generation
│   ├── pipeline.py          # Main orchestration pipeline
│   └── utils/
│       ├── cli_utils.py     # Progress bars, colored output
│       └── file_utils.py    # File handling utilities
├── examples/
│   └── basic_usage.py       # Example scripts
├── main.py                  # CLI entry point
└── pyproject.toml           # UV dependencies
```

## Quick Start

### 1. Install UV (if needed)

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies and Activate

```bash
cd pdf-vectorizer
uv sync                          # creates .venv/ and installs all packages
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Initialize and Verify

```bash
python main.py init              # create data/ directories
python main.py check             # smoke-test: verify imports and config
python main.py info              # display loaded configuration
```

### 4. Process PDFs

```bash
# Copy PDFs to input directory
cp /path/to/your/pdfs/*.pdf data/raw/

# Quick extraction check (no embedding step)
python main.py extract data/raw/document.pdf

# Full pipeline: extract → chunk → embed
python main.py process data/raw/document.pdf

# Process entire directory (add -r for recursive)
python main.py process data/raw/
```

### Success Indicators

- `uv sync` completes without errors
- `python main.py check` prints "All checks passed"
- `python main.py extract file.pdf` shows page count and character count
- `python main.py process file.pdf` creates files in `data/processed/`

## CLI Reference

```bash
python main.py check                            # verify imports & config
python main.py extract data/raw/doc.pdf         # extraction only
python main.py extract data/raw/doc.pdf -o out.json  # save extracted text
python main.py process data/raw/doc.pdf         # full pipeline
python main.py process data/raw/ --recursive    # batch, recursive
python main.py process data/raw/ --no-intermediate  # skip saving text/chunks
python main.py init                             # create directories
python main.py info                             # show configuration
```

## Python API

```python
from pathlib import Path
from src.pipeline import PDFVectorizationPipeline

pipeline = PDFVectorizationPipeline()
result = pipeline.process_pdf(Path("data/raw/document.pdf"))
print(f"Created {result['chunks_created']} chunks")
print(f"Generated {result['embeddings_count']} embeddings")
print(f"Output: {result['embeddings_file']}")

# Batch
results = pipeline.process_directory(Path("data/raw/"))
```

### Step-by-step API

```python
from src.pdf_extractor import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.vectorizer import vectorize_chunks, save_embeddings
from pathlib import Path

pages = extract_text_from_pdf(Path("document.pdf"), method="pymupdf")
chunks = chunk_text(pages, strategy="recursive", chunk_size=1000, chunk_overlap=200)
embeddings = vectorize_chunks(chunks, model_type="sentence_transformers", model_name="all-MiniLM-L6-v2")
save_embeddings(embeddings, chunks, Path("output.npz"), format="numpy")
```

## Configuration

Edit `config/config.yaml`:

```yaml
data:
  raw: data/raw
  processed: data/processed

extraction:
  method: pymupdf        # pymupdf | pdfplumber | ocr
  ocr_enabled: false
  ocr_language: eng

chunking:
  strategy: recursive    # recursive | fixed
  chunk_size: 1000
  chunk_overlap: 200

vectorization:
  model_type: sentence_transformers   # sentence_transformers | openai
  model_name: all-MiniLM-L6-v2
  batch_size: 32
  output_format: numpy   # numpy | json | jsonl
```

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Reference in config:
```yaml
vectorization:
  openai_api_key: ${OPENAI_API_KEY:}
  model_name: ${EMBEDDING_MODEL:all-MiniLM-L6-v2}
```

## Embedding Models

**Local (free)**
- `all-MiniLM-L6-v2` — fast, 384 dimensions (recommended)
- `all-mpnet-base-v2` — higher quality, 768 dimensions
- `multi-qa-MiniLM-L6-cos-v1` — optimized for Q&A

**OpenAI (paid)**
- `text-embedding-3-small` — 1536 dimensions, $0.02/1M tokens
- `text-embedding-3-large` — 3072 dimensions, $0.13/1M tokens

## Output Formats

**NumPy (`.npz`)**
```python
import numpy as np
data = np.load("embeddings.npz")
embeddings = data["embeddings"]  # shape: (num_chunks, embedding_dim)
texts = data["texts"]
```

**JSON**
```json
{
  "embeddings": [[0.1, 0.2, ...], ...],
  "chunks": [{"chunk_id": 0, "text": "...", "page": 1, "metadata": {}}],
  "embedding_dim": 384,
  "num_chunks": 100
}
```

## UV Package Management

```bash
uv sync                      # install / sync dependencies
uv sync --upgrade            # upgrade all packages
uv add package-name          # add a package
uv add --dev pytest ruff     # add dev-only packages
uv remove package-name       # remove a package
uv pip list                  # list installed packages
uv pip tree                  # show dependency tree
uv pip list --outdated       # check for updates
uv run python main.py ...    # run without activating venv

# Export for pip users
uv pip freeze > requirements.txt

# Recreate environment from scratch
rm -rf .venv uv.lock && uv sync

# Troubleshooting
uv sync --reinstall          # force reinstall all packages
uv sync --verbose            # verbose output
uv --version                 # check UV version
```

## Optional: OCR Support

For scanned PDFs:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
# Mac
brew install tesseract poppler
# Then add Python packages
uv add pytesseract pdf2image
```

Enable in `config/config.yaml`:
```yaml
extraction:
  method: pymupdf
  ocr_enabled: true
```

## Optional: OpenAI Embeddings

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
uv add openai
```

Update `config/config.yaml`:
```yaml
vectorization:
  model_type: openai
  model_name: text-embedding-3-small
```

## Performance Tips

1. Use local sentence-transformers for free, fast inference
2. Increase `batch_size` for faster embedding generation
3. Enable `parallel` processing in config for large directories
4. Only enable OCR when needed (scanned documents)
5. Tune `chunk_size` — larger gives more context, smaller gives more precision

## Troubleshooting

**`No module named 'fitz'`** → `uv add pymupdf`

**`Tesseract not found`** → install system package first (see OCR section above)

**`OpenAI API key not set`** → add `OPENAI_API_KEY` to `.env`

**`Command not found: uv`** → reinstall UV and add to PATH:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

**`Virtual environment not activated`** → either activate with `.venv\Scripts\activate` (Windows) / `source .venv/bin/activate` (Linux/Mac), or prefix commands with `uv run`:
```bash
uv run python main.py process data/raw/
```

## Contributing

Areas for improvement:
- Semantic chunking implementation
- Vector database integration (ChromaDB, Pinecone)
- Multi-language OCR support
- Table extraction enhancement

## License

MIT License

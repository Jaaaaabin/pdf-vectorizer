# PDF Vectorization Pipeline

Extract text from PDFs, chunk it, and generate embeddings for semantic search and RAG.

## Stack

- **Extraction**: PyMuPDF (fast), pdfplumber (tables), OCR via Tesseract
- **Chunking**: LangChain recursive splitter
- **Embeddings**: sentence-transformers (local) or OpenAI (cloud)
- **Output**: NumPy `.npz` or JSON, one subfolder per PDF

## Setup

```bash
uv sync
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

## Scripts

| Script | Purpose |
|---|---|
| `extract.py` | PDF → `text.json` |
| `chunk.py` | `text.json` → `chunks.json` |
| `vect.py` | `chunks.json` → `embeddings.npz` |
| `process.py` | full pipeline + check + info |

## Running

```bash
uv run python extract.py data/raw/doc.pdf      # step 1
uv run python chunk.py   data/raw/doc.pdf      # step 2
uv run python vect.py    data/raw/doc.pdf      # step 3

uv run python process.py run   data/raw/doc.pdf  # full pipeline
uv run python process.py run   data/raw/ -r      # batch, recursive
uv run python process.py check                   # verify imports and config
uv run python process.py info                    # show loaded configuration
```

## Output Structure

Each PDF gets its own subfolder:

```
data/processed/
└── my-document/
    ├── text.json        # extracted pages
    ├── chunks.json      # text chunks with metadata
    └── embeddings.npz   # vectors
```

`process.py run` skips steps whose output already exists — run `extract.py` and `chunk.py` individually to validate before embedding.

## Configuration

Edit `config/config.yaml`. All active settings are documented inline.

Key knobs:

| Setting | Default | Notes |
|---|---|---|
| `extraction.method` | `pymupdf` | `pdfplumber` for tables, `ocr` for scanned |
| `chunking.chunk_size` | `512` | chars per chunk |
| `chunking.chunk_overlap` | `128` | ~10–20% of chunk_size |
| `vectorization.model_type` | `sentence_transformers` | `openai` requires API key |
| `vectorization.model_name` | `all-MiniLM-L6-v2` | override via `EMBEDDING_MODEL` in `.env` |

## Optional: OpenAI Embeddings

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

```yaml
vectorization:
  model_type: openai
  model_name: text-embedding-3-small
```

## Optional: OCR

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
# Mac
brew install tesseract

uv add pytesseract pdf2image
```

Then set `extraction.method: ocr` or `extraction.ocr_enabled: true` in config.

## UV Reference

```bash
uv sync                  # install dependencies
uv sync --upgrade        # upgrade all packages
uv add <pkg>             # add a package
uv remove <pkg>          # remove a package
uv pip list              # list installed
uv run python process.py … # run without activating venv
```

## Load Embeddings

```python
import numpy as np
data = np.load("data/processed/my-document/embeddings.npz")
embeddings = data["embeddings"]  # shape: (n_chunks, dim)
texts = data["texts"]
```

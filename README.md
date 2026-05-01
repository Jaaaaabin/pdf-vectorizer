# PDF Vectorization Pipeline

Extract text from PDFs, chunk it, and generate embeddings for semantic search and RAG.

## Stack

- **Extraction**: PyMuPDF (fast), pdfplumber (tables)
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
| `extract.py` | PDF → `text.json` + `figures/` |
| `chunk.py` | `text.json` → `chunks.json` |
| `vect.py` | `chunks.json` → `embeddings.npz` |
| `process.py` | full pipeline + check + info |

## Running

```bash
uv run python extract.py data/raw/1.pdf      # step 1
uv run python chunk.py   data/raw/1.pdf      # step 2
uv run python vect.py    data/raw/1.pdf      # step 3

uv run python process.py extract data/raw/1.pdf  # extract only, single file
uv run python process.py extract data/raw/ -r    # extract only, batch
uv run python process.py run     data/raw/1.pdf  # full pipeline
uv run python process.py run     data/raw/ -r    # full pipeline, batch
uv run python process.py check                   # verify imports and config
uv run python process.py info                    # show loaded configuration
```

## Output Structure

Each PDF gets its own subfolder:

```
data/processed/
└── 1/
    ├── text.json        # extracted pages
    ├── chunks.json      # text chunks with metadata
    ├── embeddings.npz   # vectors
    └── figures/         # extracted images, keyed by page and index
```

`process.py run` skips steps whose output already exists — run `extract.py` and `chunk.py` individually to validate before embedding.

Figures are extracted automatically by `extract.py` and saved as `page_{N}_fig_{I}.{ext}` (e.g. `figures/page_3_fig_0.jpeg`). Each page entry in `text.json` includes a `"figures"` key listing the filenames for that page, so downstream code can link embeddings back to their source images.

## Configuration

Edit `config/config.yaml`. All active settings are documented inline.

Key knobs:

| Setting | Default | Notes |
|---|---|---|
| `extraction.method` | `pymupdf` | `pdfplumber` for tables |
| `extraction.min_figure_px` | `100` | skip figures smaller than this in either dimension |
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

## Known Issues / TODO

### Chunking
- [ ] Explore optimal `chunk_size` / `chunk_overlap` for this dataset (baseline: 512/128 — try 500/100 for precision, 2000/400 for context)
- [ ] Compare recursive vs fixed strategy on a sample PDF
- [ ] Evaluate chunk quality: check if chunks cut mid-sentence frequently

### Vectorization — Performance
- [ ] Profile where time is spent: import vs model load vs encode (add timestamps around each phase in `run_embed`)
- [ ] Test larger `batch_size` (64, 128) to see if encoding speeds up
- [ ] Check if GPU/CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Investigate faster model alternatives (e.g. `paraphrase-MiniLM-L3-v2`)

### Vectorization — Simplification
- [ ] Remove `jsonl` format from `save_embeddings` (dead code, not exposed in config)
- [ ] Consider lazy model loading — only instantiate `SentenceTransformer` when `encode()` is called
- [ ] Review whether `load_embeddings` is needed from CLI (currently unreachable)

### Vectorization — PyTorch Compatibility
- [ ] Verify torch version matches sentence-transformers requirements: `python -c "import torch; print(torch.__version__)"`
- [ ] Check for version conflicts: `uv pip list | grep -E "torch|transformers|sentence"`
- [ ] Test on a clean environment to rule out install-order issues

## UV Reference

```bash
uv sync                  # install dependencies
uv sync --upgrade        # upgrade all packages
uv add <pkg>             # add a package
uv remove <pkg>          # remove a package
uv pip list              # list installed
uv run python process.py … # run without activating venv
```

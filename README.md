# semanticcache

Framework-agnostic semantic caching for APIs and LLMs: embeddings, vector storage (pgvector), and response caching (Redis).

The PyPI distribution name is **`semanticcache-py`** (the import package remains `semanticcache`).

## Install

```bash
pip install semanticcache-py
```

Optional extras:

- `embed-local` / `embed-local-cpu`: local embeddings via Sentence Transformers with **CPU** PyTorch (default PyPI wheels)
- `embed-local-gpu`: same libraries; install with PyTorch’s **CUDA** wheel index so `torch` resolves to a GPU build (see below)
- `embed-openai`: OpenAI embeddings and tiktoken

**CPU (Sentence Transformers + CPU PyTorch)**

```bash
pip install "semanticcache-py[embed-local-cpu]"
# or: pip install "semanticcache-py[embed-local]"
```

**GPU (Sentence Transformers + CUDA PyTorch)**

Pick a CUDA version that matches your driver and OS from [PyTorch Get Started](https://pytorch.org/get-started/locally/), then install with that index so pip selects CUDA wheels (example for CUDA 12.4):

```bash
pip install "semanticcache-py[embed-local-gpu]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Use the URL shown on the PyTorch site for your platform; the extra name is the same (`embed-local-gpu`).

## Requirements

Python 3.12+.

## Links

- Repository: https://github.com/axm1647/SemanticCache-py

## License

Apache-2.0. See `LICENSE`.

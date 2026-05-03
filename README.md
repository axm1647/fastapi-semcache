# semanticcache

Framework-agnostic semantic caching for APIs and LLMs: embeddings, vector storage (pgvector), and response caching (Redis).

The PyPI distribution name is **`semanticcache-py`** (the import package remains `semanticcache`).

## Install

```bash
pip install semanticcache-py
```

Optional extras:

- `embed-local`: local embeddings via Sentence Transformers
- `embed-openai`: OpenAI embeddings and tiktoken

```bash
pip install "semanticcache-py[embed-local]"
```

## Requirements

Python 3.12+.

## Links

- Repository: https://github.com/axm1647/SemanticCache-py

## License

Apache-2.0. See `LICENSE`.

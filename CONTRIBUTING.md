# Contributing to fastapi-semcache

Thank you for helping improve this project. This document describes how to set up a development environment, run checks, and submit changes.

## Contributing

This is currently a **solo project** and I would love to have more people working on it. If you would like to contribute, opening a discussion or reaching out through an issue would be great.

Feel free to open an **issue** or a **pull request**. For planned features and larger themes, see the [Roadmap](README.md#future-support) in the README (streaming work in particular is outlined under [Streaming and chunked responses](README.md#streaming-and-chunked-responses)).

## Development setup

You need **Python 3.12 or newer**. This repo uses **[uv](https://docs.astral.sh/uv/)** for environments and locking (`uv.lock`).

From the repository root:

```bash
uv sync --extra test
```

That installs the package in editable mode plus the `test` optional dependency group (`pytest`, `pytest-asyncio`, `numpy`, and related pins).

To exercise semantic caching against Postgres and Redis locally, copy `.env.example` to `.env`, adjust connection strings and options, and load those variables in your shell or process manager. Variable names are prefixed with `SEMANTIC_CACHE_`. See the README and `semanticcache.config` for behavior.

## Running tests

Run the full suite:

```bash
uv run pytest
```

Some tests live under `tests/embedders/test_sbert_integration.py` and are marked **`integration`**. They import **`sentence_transformers`** and may download a small model on first run. If that package is not installed, pytest skips that file entirely.

To run integration tests after installing an embedding extra (for example CPU Sentence Transformers):

```bash
uv sync --extra embed-huggingface-cpu --extra test
uv run pytest -m integration
```

To exclude integration tests when you have heavy extras installed:

```bash
uv run pytest -m "not integration"
```

## Type checking

The project configures **[basedpyright](https://github.com/DetachHead/basedpyright)** in `pyproject.toml` under `[tool.basedpyright]` (strict mode on the `semanticcache` package). Install the CLI in your environment if you want to run it locally:

```bash
uv pip install basedpyright
uv run basedpyright semanticcache
```

CI or local workflows may add other linters over time. Match whatever is already configured in the repo when you open a pull request.

## Code style

- Follow existing patterns in the tree (imports, naming, error handling, async style).
- Prefer **Google-style docstrings** for new or substantially edited public APIs (module, class, and function docstrings), consistent with the rest of the package.
- The PyPI distribution name is **`fastapi-semcache`**. THe import package is **`semanticcache`**. Keep that distinction clear in docs and examples.

## Documentation

If your change affects install steps, configuration, embedders, or public behavior, update the relevant files under **`docs/`** (for example `docs/embedders.md`) and any user-facing sections of **`README.md`** when appropriate.

## Submitting changes

1. **Open an issue** first if you are planning a large or ambiguous change, so we can agree on direction.
2. **Keep pull requests focused**: one logical change per PR is easier to review.
3. **Describe** what you changed and why in the PR body. Link related issues when applicable.
4. **Commit messages**: use clear, conventional-style prefixes when it helps readers, for example `feat:`, `fix:`, `docs:`, `test:`, optionally with a short scope.

All contributions are licensed under the same terms as the project (**Apache-2.0**). See `LICENSE`.

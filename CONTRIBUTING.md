# Contributing

Use Python 3.11 and create a virtual environment before installing
`requirements-dev.txt`. Keep changes focused and add tests for synchronization,
queue, parser, or retrieval behavior.

```bash
python -m pytest -q
python -m ruff check app tests scripts
```

Never commit real Feishu documents, tenant tokens, API keys, evaluation outputs,
or local vector-database volumes. Integration tests requiring real Feishu
credentials must remain opt-in and must not run in pull requests from forks.

This is a community project and is not an official Feishu/Lark product.

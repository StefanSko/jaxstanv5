# jaxstanv5

Python project scaffold using the Astral stack:

- [`uv`](https://docs.astral.sh/uv/) for packaging, dependency management, and command execution
- [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting
- [`ty`](https://docs.astral.sh/ty/) for type checking

## Quick start

```bash
uv sync
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest
```

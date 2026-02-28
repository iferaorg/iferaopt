# AGENTS.md

## Project Overview

**iferaopt** is a GPU-accelerated walk-forward optimization system for a 0DTE SPX
put credit spread strategy using Opening Range Breakout signals. The codebase is
Python (3.10+) with PyTorch/CUDA, scikit-learn, and Parquet/Zarr data storage.

## Repository Layout

```
iferaopt/          # Main source package
tests/             # Pytest test suite
data/              # Local data (raw + processed), not committed to git
planning_discussion.txt  # Full design discussion (read for context)
TODO.md            # Project roadmap
```

## Development Workflow

### Running Tests

```bash
pytest                  # run all tests
pytest tests/test_foo.py  # run a specific test file
pytest -x               # stop on first failure
```

### Code Quality

**Format** code with Black before committing:
```bash
black iferaopt/ tests/
```

**Lint / type-check** with pylint and pyright:
```bash
pylint iferaopt/
pyright iferaopt/
```

Existing config files: `.pylintrc`, `.flake8`, `pyproject.toml`.

### Adding New Features

1. Create or modify modules under `iferaopt/`.
2. **Always** add corresponding tests in `tests/` (e.g., `tests/test_<module>.py`).
3. Run the new tests and any related existing tests to confirm nothing is broken.
4. Format with `black` and check with `pylint`/`pyright` before committing.

### Fixing Bugs

1. Write a failing test that reproduces the bug.
2. Apply the minimal fix.
3. Run the new test plus all related existing tests.
4. Format and lint as above.

## Key Guidelines

- Keep functions and classes well-typed (use type annotations).
- Use `pyproject.toml` for project and tool configuration.
- Data files under `data/` are git-ignored; do not commit large data.
- Refer to `TODO.md` for the project roadmap and `planning_discussion.txt` for
  detailed design rationale.

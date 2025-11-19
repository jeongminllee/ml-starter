# ML Starter

A template repository for Machine Learning projects, designed to help you get started quickly with a robust development environment.

## Features
- **Python 3.12** environment management with `uv` (or `poetry`).
- **Code Quality**: `ruff` (linting), `black` (formatting), `pre-commit` hooks.
- **Testing**: `pytest` framework.
- **Experiment Tracking**: `wandb` integration ready.
- **CI/CD**: GitHub Actions for automated testing and linting.

## Getting Started

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or Poetry

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ml-starter
    ```

2.  **Set up the environment:**

    Using `uv` (Recommended):
    ```bash
    # Create virtual environment and install dependencies
    uv sync
    ```

    Using `pip`:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -e ".[dev]"
    ```

3.  **Install pre-commit hooks:**
    ```bash
    pre-commit install
    ```

## Development Workflow

- **Run Tests:**
    ```bash
    pytest
    ```

- **Lint & Format:**
    ```bash
    ruff check . --fix
    black .
    ```

- **Verify Environment:**
    ```bash
    python scripts/verify_env.py
    ```

## Project Structure
```
ml-starter/
├── .github/            # GitHub Actions & Templates
├── src/
│   └── ml_starter/     # Source code
├── tests/              # Tests
├── scripts/            # Utility scripts
├── notebooks/          # Jupyter notebooks
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

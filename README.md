## Overview
This is the machine learning model for the EMS-Sense project.

## How to use
This model is built using [sklearn](https://scikit-learn.org/stable). To use it, simply follow these steps:
1. Setup uv on your machine [uv](https://github.com/astral-sh/uv).
2. Run `uv venv` to create your virtual environment.
3. Run `uv pip install -r pyproject.toml` to install the dependencies.
4. Run `uv run model.py` to train the model.
5. Run `fastapi dev api.py` to serve the api for the model.

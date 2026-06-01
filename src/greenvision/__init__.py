"""GreenVision — plant-disease image classifier.

A fine-tuned EfficientNet-B0 trained on PlantVillage with a two-phase transfer-learning
strategy (freeze backbone -> unfreeze and fine-tune), tracked with MLflow and served via
FastAPI.

The package is intentionally light at import time: importing :mod:`greenvision` does not
pull in torch. Import the submodules you need (``greenvision.constants``,
``greenvision.data``, ``greenvision.model``, ``greenvision.engine``) explicitly.
"""

__version__ = "0.1.0"

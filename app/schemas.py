"""Pydantic response schemas for the GreenVision API.

The ``/predict`` contract is fixed by guide §7: ``{"class", "confidence", "class_index"}``.
Because ``class`` is a reserved word in Python, the field is named ``class_name`` and
exposed under the JSON key ``class`` via a Pydantic alias. FastAPI serializes responses
with ``by_alias=True``, so clients see ``class``; ``populate_by_name=True`` lets us build
the models with the Python field name.

🔒 Changing this schema later is a breaking change and requires confirmation (agent.md).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Prediction(BaseModel):
    """One class prediction: the label, its softmax confidence, and its index."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(
        alias="class",
        description="Predicted class label (PlantVillage `Crop___Disease` convention).",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Softmax probability for this class, in [0, 1]."
    )
    class_index: int = Field(
        ge=0, description="Index into class_names.json (inference ground truth)."
    )


class PredictionResponse(Prediction):
    """The ``/predict`` response: the top prediction plus a ranked ``top_k`` list."""

    top_k: list[Prediction] = Field(
        default_factory=list,
        description="Top-k predictions in descending confidence order (includes the top one).",
    )


class HealthResponse(BaseModel):
    """Liveness/readiness payload for ``/health``."""

    # `model_loaded` starts with the protected `model_` prefix; opt out of the namespace.
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="`ok` when the service is up.")
    model_loaded: bool = Field(description="True once the checkpoint is loaded and ready.")
    device: str | None = Field(
        default=None, description="Device the model is running on (`cpu`/`cuda`), if loaded."
    )


class ClassesResponse(BaseModel):
    """The frozen class list served at ``/classes``."""

    num_classes: int = Field(description="Number of classes (should equal NUM_CLASSES).")
    classes: list[str] = Field(description="Class labels in index order (class_names.json).")

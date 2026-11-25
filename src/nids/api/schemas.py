from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FlowIn(BaseModel):
    """One network flow: feature name -> value (same column names as the training data).

    Extra keys are ignored; missing required features return 422.
    """
    features: dict[str, float] = Field(..., examples=[{"Flow Duration": 1200.0,
                                                       "Flow Packets/s": 48000.0}])


class BatchIn(BaseModel):
    flows: list[FlowIn]


class Contribution(BaseModel):
    feature: str
    value: float
    contribution: float


class Prediction(BaseModel):
    id: int | None = None
    label: str
    confidence: float
    is_attack: bool
    probabilities: dict[str, float]
    explanation: list[Contribution]
    story: str
    severity: str | None = None
    severity_score: float | None = None
    anomaly_score: float | None = None
    incident_id: int | None = None
    model: str


class FeedbackIn(BaseModel):
    verdict: Literal["confirmed", "false_positive"]
    note: str | None = None

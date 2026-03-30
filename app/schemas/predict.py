from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class ServiceVisit(BaseModel):
    date: date
    mile: float = Field(..., ge=0)
    repair_type: str = Field(..., min_length=1)


class RawPredictRequest(BaseModel):
    vin: str = Field(..., min_length=1)
    owner_type: str = Field(default='个人')
    car_mode: str = Field(..., min_length=1)
    car_level: str = Field(..., description='例如 family_1 / family_2 / family_3')
    member_level: str = Field(default='无')
    purchase_date: date
    reference_date: date | None = None
    visits: list[ServiceVisit] = Field(..., min_length=2)


class FeaturePredictRequest(BaseModel):
    VIN: str = Field(..., min_length=1)
    last_mile: float
    last_till_now_days: float
    first_to_purchase_day_diff: float
    first_to_purchase_mile_diff: float
    second_to_first_day_diff: float
    second_to_first_mile_diff: float
    day_diff_median: float
    mile_diff_median: float
    day_speed_median: float
    day_cv: float
    mile_cv: float
    day_speed_cv: float
    all_times: float
    car_age: float
    last_repair_type: str
    all_repair_types: str
    owner_type: str
    car_mode: str
    car_level: str
    member_level: str


class NeighborInfo(BaseModel):
    vin: str
    label: int
    similarity: float


class PredictResponse(BaseModel):
    vin: str
    pred_label: int
    churn_score: float
    positive_neighbors: int
    neighbor_count: int
    top_neighbors: list[NeighborInfo]
    used_features: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_bundle_path: str
    model_load_error: str | None = None
    model_provider: str | None = None

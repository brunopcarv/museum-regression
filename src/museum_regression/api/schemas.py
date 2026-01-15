"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class CitySchema(BaseModel):
    """Schema for city data."""

    id: Optional[int] = None
    name: str
    country: str
    population: int
    population_year: Optional[int] = None
    wikipedia_url: Optional[str] = None

    class Config:
        from_attributes = True


class MuseumSchema(BaseModel):
    """Schema for museum data."""

    id: Optional[int] = None
    name: str
    museum_type: Optional[str] = None
    visitors: int
    visitor_year: Optional[int] = None
    wikipedia_url: Optional[str] = None
    city: Optional[CitySchema] = None

    class Config:
        from_attributes = True


class MuseumListResponse(BaseModel):
    """Response schema for museum list."""

    count: int
    museums: List[MuseumSchema]


class RegressionMetricsSchema(BaseModel):
    """Schema for regression metrics."""

    r2_score: float = Field(..., description="R-squared coefficient of determination")
    mse: float = Field(..., description="Mean squared error")
    rmse: float = Field(..., description="Root mean squared error")
    mae: float = Field(..., description="Mean absolute error")
    coefficient: float = Field(..., description="Regression coefficient (slope)")
    intercept: float = Field(..., description="Regression intercept")
    n_samples: int = Field(..., description="Number of samples used for training")
    cv_mean: Optional[float] = Field(None, description="Cross-validation mean R²")
    cv_std: Optional[float] = Field(None, description="Cross-validation std R²")
    equation: str = Field(..., description="Linear equation as string")


class PredictionRequest(BaseModel):
    """Request schema for prediction."""

    population: int = Field(..., gt=0, description="City population")


class PredictionResponse(BaseModel):
    """Response schema for prediction."""

    population: int
    predicted_visitors: int
    confidence_note: str = "Prediction based on linear regression model"


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    version: str
    timestamp: datetime
    database_initialized: bool
    model_trained: bool

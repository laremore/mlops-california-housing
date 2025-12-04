
from pydantic import BaseModel, Field
from typing import Optional, List

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None

class PredictionRequest(BaseModel):

    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    housing_median_age: float = Field(..., ge=0, le=100, description="Median age of houses")
    total_rooms: float = Field(..., ge=0, description="Total number of rooms")
    total_bedrooms: float = Field(..., ge=0, description="Total number of bedrooms")
    population: float = Field(..., ge=0, description="Total population")
    households: float = Field(..., ge=0, description="Number of households")
    median_income: float = Field(..., ge=0, description="Median income (tens of thousands)")
    

    bedrooms_per_room: Optional[float] = Field(0.2, ge=0, le=1, description="Bedrooms per room ratio")
    population_per_household: Optional[float] = Field(3.0, ge=0, description="Population per household")
    households_per_population: Optional[float] = Field(0.3, ge=0, le=1, description="Households per population")

    ocean__INLAND: Optional[int] = Field(0, ge=0, le=1)
    ocean__NEAR_BAY: Optional[int] = Field(0, ge=0, le=1)
    ocean__NEAR_OCEAN: Optional[int] = Field(0, ge=0, le=1)
    ocean__ISLAND: Optional[int] = Field(0, ge=0, le=1)

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    request_id: str

class BatchPredictionRequest(BaseModel):
    instances: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    total_processed: int
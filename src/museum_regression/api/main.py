"""
FastAPI REST service for museum population regression.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from museum_regression import __version__
from museum_regression.api.schemas import (
    CitySchema,
    HealthResponse,
    MuseumListResponse,
    MuseumSchema,
    PredictionResponse,
    RegressionMetricsSchema,
)
from museum_regression.data.wikipedia_client import WikipediaClient
from museum_regression.db.database import Database
from museum_regression.ml.regression import MuseumRegressionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
db: Optional[Database] = None
model: Optional[MuseumRegressionModel] = None
wiki_client: Optional[WikipediaClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global db, model, wiki_client

    # Get database path from environment or use default
    db_path = os.environ.get("DATABASE_PATH", "/data/museum_data.db")
    min_visitors = int(os.environ.get("MIN_VISITORS", 2_000_000))

    logger.info(f"Initializing database at {db_path}")
    db = Database(db_path)
    db.init_db()

    logger.info(f"Initializing Wikipedia client (min visitors: {min_visitors:,})")
    wiki_client = WikipediaClient(min_visitors=min_visitors)

    logger.info("Initializing regression model")
    model = MuseumRegressionModel()

    # Load model params from database (trained by data-ingestion container)
    regression_result = db.get_latest_regression_result()
    if regression_result:
        logger.info(
            f"Loading model from database (R² = {regression_result.r2_score:.4f})"
        )
        model.load_from_params(
            coefficient=regression_result.coefficient,
            intercept=regression_result.intercept,
            r2_score=regression_result.r2_score,
            mse=regression_result.mse,
            mae=regression_result.mae,
            n_samples=regression_result.n_samples,
        )
    else:
        logger.warning(
            "No trained model found in database. Run data-ingestion container or POST /regression/train"
        )

    yield

    # Cleanup
    logger.info("Shutting down")


# Create FastAPI app
app = FastAPI(
    title="Museum Population Regression API",
    description=(
        "API for correlating museum visitor attendance with city populations. "
        "Fetches data from Wikipedia and provides linear regression analysis."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow(),
        database_initialized=db is not None,
        model_trained=model.is_trained if model else False,
    )


@app.get("/museums", response_model=MuseumListResponse, tags=["Museums"])
async def list_museums():
    """Get all museums in the database."""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    df = db.get_museum_city_data()

    museums = []
    for _, row in df.iterrows():
        museums.append(
            MuseumSchema(
                name=row["museum_name"],
                museum_type=row.get("museum_type"),
                visitors=row["visitors"],
                visitor_year=row.get("visitor_year"),
                city=CitySchema(
                    name=row["city"],
                    country=row["country"],
                    population=row["population"],
                ),
            )
        )

    return MuseumListResponse(count=len(museums), museums=museums)


@app.get("/regression/predict", response_model=PredictionResponse, tags=["Regression"])
async def predict_visitors(
    population: int = Query(..., gt=0, description="City population")
):
    """Predict museum visitors based on city population."""
    if not model or not model.is_trained:
        raise HTTPException(
            status_code=400, detail="Model not trained. Please train the model first."
        )

    predicted = model.predict(population)

    return PredictionResponse(
        population=population,
        predicted_visitors=int(predicted),
        confidence_note=f"Based on linear regression (R² = {model.metrics.r2_score:.3f})",
    )


@app.get(
    "/regression/stats", response_model=RegressionMetricsSchema, tags=["Regression"]
)
async def get_model_stats():
    """Get statistics for the trained regression model."""
    if not model or not model.is_trained:
        raise HTTPException(
            status_code=400, detail="Model not trained. Please train the model first."
        )

    return RegressionMetricsSchema(
        **model.metrics.to_dict(), equation=model.get_equation()
    )


def run_server():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "museum_regression.api.main:app", host="0.0.0.0", port=8000, reload=True
    )


if __name__ == "__main__":
    run_server()

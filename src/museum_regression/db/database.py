"""
Database operations for museum and city data.
"""

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from museum_regression.data.wikipedia_client import City, Museum
from museum_regression.db.models import (
    Base,
    CityModel,
    MuseumModel,
    RegressionResultModel,
)

logger = logging.getLogger(__name__)


class Database:
    """Database manager for museum and city data."""

    def __init__(self, db_path: str = "museum_data.db"):
        """
        Initialize the database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def init_db(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {self.db_path}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def add_city(self, city: City, session: Optional[Session] = None) -> CityModel:
        """
        Add a city to the database.

        Args:
            city: City data object
            session: Optional existing session

        Returns:
            Created or existing CityModel
        """
        should_close = session is None
        session = session or self.get_session()

        try:
            # Check if city already exists
            existing = (
                session.query(CityModel)
                .filter_by(name=city.name, country=city.country)
                .first()
            )

            if existing:
                # Update population if changed
                if existing.population != city.population:
                    existing.population = city.population
                    session.commit()
                return existing

            # Create new city
            city_model = CityModel(
                name=city.name,
                country=city.country,
                population=city.population,
                population_year=city.population_year,
                wikipedia_url=city.wikipedia_url,
            )
            session.add(city_model)
            session.commit()
            logger.info(f"Added city: {city.name}")
            return city_model

        finally:
            if should_close:
                session.close()

    def add_museum(
        self, museum: Museum, city_model: CityModel, session: Optional[Session] = None
    ) -> MuseumModel:
        """
        Add a museum to the database.

        Args:
            museum: Museum data object
            city_model: Associated city model
            session: Optional existing session

        Returns:
            Created or existing MuseumModel
        """
        should_close = session is None
        session = session or self.get_session()

        try:
            # Check if museum already exists
            existing = session.query(MuseumModel).filter_by(name=museum.name).first()

            if existing:
                # Update visitor count if changed
                if existing.visitors != museum.visitors:
                    existing.visitors = museum.visitors
                    existing.visitor_year = museum.year
                    session.commit()
                return existing

            # Create new museum
            museum_model = MuseumModel(
                name=museum.name,
                museum_type=museum.museum_type,
                visitors=museum.visitors,
                visitor_year=museum.year,
                wikipedia_url=museum.wikipedia_url,
                city_id=city_model.id,
            )
            session.add(museum_model)
            session.commit()
            logger.info(f"Added museum: {museum.name}")
            return museum_model

        finally:
            if should_close:
                session.close()

    def populate_from_wikipedia(
        self, museums: list[Museum], cities: dict[str, City]
    ) -> tuple[int, int]:
        """
        Populate the database from Wikipedia data.

        Args:
            museums: List of museums
            cities: Dictionary of city data

        Returns:
            Tuple of (museums_added, cities_added)
        """
        session = self.get_session()
        museums_added = 0
        cities_added = 0

        try:
            for museum in museums:
                # Get or create city
                city_data = cities.get(museum.city)
                if not city_data:
                    logger.warning(
                        f"No city data for {museum.city}, skipping {museum.name}"
                    )
                    continue

                # Check if city exists
                city_model = (
                    session.query(CityModel)
                    .filter_by(name=city_data.name, country=city_data.country)
                    .first()
                )

                if not city_model:
                    city_model = self.add_city(city_data, session)
                    cities_added += 1

                # Add museum
                existing_museum = (
                    session.query(MuseumModel).filter_by(name=museum.name).first()
                )

                if not existing_museum:
                    self.add_museum(museum, city_model, session)
                    museums_added += 1

            session.commit()
            return museums_added, cities_added

        finally:
            session.close()

    def get_all_museums(self) -> list[MuseumModel]:
        """Get all museums from the database."""
        session = self.get_session()
        try:
            return session.query(MuseumModel).all()
        finally:
            session.close()

    def get_all_cities(self) -> list[CityModel]:
        """Get all cities from the database."""
        session = self.get_session()
        try:
            return session.query(CityModel).all()
        finally:
            session.close()

    def get_museum_city_data(self) -> pd.DataFrame:
        """
        Get combined museum and city data as a DataFrame.

        Returns:
            DataFrame with museum and city information
        """
        session = self.get_session()
        try:
            museums = (
                session.query(MuseumModel, CityModel)
                .join(CityModel, MuseumModel.city_id == CityModel.id)
                .all()
            )

            data = []
            for museum, city in museums:
                data.append(
                    {
                        "museum_name": museum.name,
                        "museum_type": museum.museum_type,
                        "visitors": museum.visitors,
                        "visitor_year": museum.visitor_year,
                        "city": city.name,
                        "country": city.country,
                        "population": city.population,
                    }
                )

            return pd.DataFrame(data)

        finally:
            session.close()

    def save_regression_result(
        self,
        r2_score: float,
        mse: float,
        mae: float,
        coefficient: float,
        intercept: float,
        n_samples: int,
        rmse: Optional[float] = None,
    ) -> RegressionResultModel:
        """
        Save regression model results to the database.

        Args:
            r2_score: R-squared score
            mse: Mean squared error
            mae: Mean absolute error
            coefficient: Regression coefficient (slope)
            intercept: Regression intercept
            n_samples: Number of samples used
            rmse: Root mean squared error (computed from mse if not provided)

        Returns:
            Created RegressionResultModel
        """
        import numpy as np

        session = self.get_session()
        try:
            result = RegressionResultModel(
                r2_score=r2_score,
                mse=mse,
                rmse=rmse if rmse is not None else np.sqrt(mse),
                mae=mae,
                coefficient=coefficient,
                intercept=intercept,
                n_samples=n_samples,
            )
            session.add(result)
            session.commit()
            logger.info(f"Saved regression result: R² = {r2_score:.4f}")
            return result
        finally:
            session.close()

    def get_latest_regression_result(self) -> Optional[RegressionResultModel]:
        """Get the most recent regression result."""
        session = self.get_session()
        try:
            return (
                session.query(RegressionResultModel)
                .order_by(RegressionResultModel.created_at.desc())
                .first()
            )
        finally:
            session.close()

    def clear_data(self):
        """Clear all data from the database."""
        session = self.get_session()
        try:
            session.query(MuseumModel).delete()
            session.query(CityModel).delete()
            session.query(RegressionResultModel).delete()
            session.commit()
            logger.info("Database cleared")
        finally:
            session.close()

"""
Database models for museum and city data.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class CityModel(Base):
    """SQLAlchemy model for cities."""

    __tablename__ = "cities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    country = Column(String(255), nullable=False)
    population = Column(Integer, nullable=False)
    population_year = Column(Integer, nullable=True)
    wikipedia_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to museums
    museums = relationship("MuseumModel", back_populates="city")

    def __repr__(self):
        return f"<City(name='{self.name}', country='{self.country}', population={self.population})>"


class MuseumModel(Base):
    """SQLAlchemy model for museums."""

    __tablename__ = "museums"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    museum_type = Column(String(100), nullable=True)
    visitors = Column(Integer, nullable=False)
    visitor_year = Column(Integer, nullable=True)
    wikipedia_url = Column(String(500), nullable=True)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to city
    city = relationship("CityModel", back_populates="museums")

    def __repr__(self):
        return f"<Museum(name='{self.name}', visitors={self.visitors})>"


class RegressionResultModel(Base):
    """SQLAlchemy model for storing regression results."""

    __tablename__ = "regression_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(100), nullable=False, default="linear_regression")
    r2_score = Column(Float, nullable=True)
    mse = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    coefficient = Column(Float, nullable=True)  # Slope
    intercept = Column(Float, nullable=True)
    n_samples = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<RegressionResult(r2={self.r2_score:.4f}, coef={self.coefficient:.4f})>"
        )

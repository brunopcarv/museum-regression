"""Shared fixtures for museum-population-regression tests."""

import os
import tempfile

import pandas as pd
import pytest

from museum_regression.data.wikipedia_client import City, Museum
from museum_regression.db.database import Database


@pytest.fixture
def sample_museums():
    """Sample museum data for testing."""
    return [
        Museum(
            name="Test Museum 1",
            city="Test City 1",
            country="Test Country",
            visitors=5_000_000,
        ),
        Museum(
            name="Test Museum 2",
            city="Test City 2",
            country="Test Country",
            visitors=3_000_000,
        ),
    ]


@pytest.fixture
def sample_cities():
    """Sample city data for testing."""
    return {
        "Test City 1": City(
            name="Test City 1", country="Test Country", population=10_000_000
        ),
        "Test City 2": City(
            name="Test City 2", country="Test Country", population=5_000_000
        ),
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for regression testing."""
    return pd.DataFrame(
        {
            "museum_name": ["Museum A", "Museum B", "Museum C", "Museum D", "Museum E"],
            "city": ["City A", "City B", "City C", "City D", "City E"],
            "country": ["Country A", "Country B", "Country C", "Country D", "Country E"],
            "population": [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000],
            "visitors": [500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000],
        }
    )


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = Database(db_path)
    db.init_db()
    yield db
    os.unlink(db_path)

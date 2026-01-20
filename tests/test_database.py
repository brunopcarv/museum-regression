"""Tests for database operations."""

import pandas as pd

from museum_regression.data.wikipedia_client import City, Museum


class TestDatabase:
    """Tests for database operations."""

    def test_database_initialization(self, temp_db):
        """Test database initialization creates tables."""
        assert temp_db.engine is not None

    def test_add_and_retrieve_city(self, temp_db):
        """Test adding and retrieving a city."""
        city = City(name="Paris", country="France", population=2_100_000)
        temp_db.add_city(city)

        cities = temp_db.get_all_cities()
        assert len(cities) == 1
        assert cities[0].name == "Paris"

    def test_add_and_retrieve_museum(self, temp_db):
        """Test adding and retrieving a museum."""
        city = City(name="Paris", country="France", population=2_100_000)
        temp_db.add_city(city)

        cities = temp_db.get_all_cities()
        city_model = cities[0]

        museum = Museum(name="Louvre", city="Paris", country="France", visitors=7_800_000)
        temp_db.add_museum(museum, city_model)

        museums = temp_db.get_all_museums()
        assert len(museums) == 1

    def test_get_museum_city_data(self, temp_db, sample_museums, sample_cities):
        """Test getting combined museum/city data as DataFrame."""
        temp_db.populate_from_wikipedia(sample_museums, sample_cities)
        df = temp_db.get_museum_city_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "museum_name" in df.columns
        assert "population" in df.columns

    def test_save_and_get_regression_result(self, temp_db):
        """Test saving and retrieving regression results."""
        temp_db.save_regression_result(
            r2_score=0.85, mse=1_000_000, mae=500, coefficient=0.5, intercept=1000, n_samples=10
        )
        result = temp_db.get_latest_regression_result()
        assert result is not None
        assert result.r2_score == 0.85

    def test_clear_data(self, temp_db, sample_museums, sample_cities):
        """Test clearing all data."""
        temp_db.populate_from_wikipedia(sample_museums, sample_cities)
        temp_db.clear_data()
        assert len(temp_db.get_all_museums()) == 0

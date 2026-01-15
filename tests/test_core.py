"""Consolidated tests for the museum-population-regression project.

This file contains ~30 focused tests covering core functionality:
- Wikipedia client (data fetching)
- Database operations
- Regression model
- API endpoints
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from museum_regression.data.wikipedia_client import City, Museum, WikipediaClient
from museum_regression.db.database import Database
from museum_regression.ml.regression import MuseumRegressionModel, RegressionMetrics


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_museums():
    """Sample museum data for testing."""
    return [
        Museum(
            name="Test Museum 1",
            city="Test City 1",
            country="Test Country",
            visitors=5_000_000,
            year=2023,
        ),
        Museum(
            name="Test Museum 2",
            city="Test City 2",
            country="Test Country",
            visitors=3_000_000,
            year=2023,
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


# =============================================================================
# Wikipedia Client Tests
# =============================================================================


class TestWikipediaClient:
    """Tests for Wikipedia data fetching."""

    def test_museum_dataclass_creation(self):
        """Test Museum dataclass creation with defaults."""
        museum = Museum(
            name="Louvre",
            city="Paris",
            country="France",
            visitors=7_800_000,
            year=2023,
        )
        assert museum.name == "Louvre"
        assert museum.museum_type == ""  # Default

    def test_city_dataclass_creation(self):
        """Test City dataclass creation with defaults."""
        city = City(name="Paris", country="France", population=2_100_000)
        assert city.name == "Paris"
        assert city.population_year is None  # Default

    def test_client_initialization(self):
        """Test WikipediaClient initialization."""
        client = WikipediaClient(min_visitors=1_000_000)
        assert client.min_visitors == 1_000_000

    def test_parse_number_formats(self):
        """Test parsing various number formats."""
        client = WikipediaClient()
        assert client._parse_number("1,000,000") == 1_000_000
        assert client._parse_number("5000000") == 5_000_000
        assert client._parse_number("1 000 000") == 1_000_000
        assert client._parse_number("5,000,000[1]") == 5_000_000  # With footnote
        assert client._parse_number("") is None
        assert client._parse_number("no numbers") is None

    def test_parse_museum_table(self):
        """Test parsing museum table HTML."""
        client = WikipediaClient(min_visitors=2_000_000)
        html = """
        <table class="wikitable">
            <tr><th>Museum</th><th>City</th><th>Visitors</th></tr>
            <tr>
                <td><a href="/wiki/Louvre">Louvre</a></td>
                <td>Paris, France</td>
                <td>7,800,000</td>
            </tr>
            <tr>
                <td>Small Museum</td>
                <td>Small Town, Country</td>
                <td>500,000</td>
            </tr>
        </table>
        """
        museums = client.parse_museum_table(html)
        assert len(museums) == 1  # Only Louvre meets threshold
        assert museums[0].name == "Louvre"

    @patch.object(WikipediaClient, "get_museum_list_html")
    @patch.object(WikipediaClient, "parse_museum_table")
    def test_fetch_museums(self, mock_parse, mock_get_html):
        """Test fetching museums from Wikipedia."""
        mock_get_html.return_value = "<html>test</html>"
        mock_parse.return_value = [
            Museum(name="Test", city="City", country="Country", visitors=5_000_000, year=2023)
        ]
        client = WikipediaClient()
        museums = client.fetch_museums()
        assert len(museums) == 1
        mock_get_html.assert_called_once()


# =============================================================================
# Database Tests
# =============================================================================


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

        # Get the city back within a session context
        cities = temp_db.get_all_cities()
        city_model = cities[0]

        museum = Museum(name="Louvre", city="Paris", country="France", visitors=7_800_000, year=2023)
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


# =============================================================================
# Regression Model Tests
# =============================================================================


class TestRegressionModel:
    """Tests for the regression model."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = MuseumRegressionModel()
        assert not model.is_trained

    def test_prepare_data(self, sample_dataframe):
        """Test data preparation."""
        model = MuseumRegressionModel()
        X, y = model.prepare_data(sample_dataframe)
        assert X.shape == (5, 1)
        assert y.shape == (5,)

    def test_train_model(self, sample_dataframe):
        """Test model training."""
        model = MuseumRegressionModel()
        metrics = model.train(sample_dataframe)

        assert model.is_trained
        assert isinstance(metrics, RegressionMetrics)
        assert metrics.n_samples == 5
        assert metrics.r2_score >= 0

    def test_predict_requires_training(self):
        """Test that prediction fails before training."""
        model = MuseumRegressionModel()
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(1_000_000)

    def test_predict_after_training(self, sample_dataframe):
        """Test prediction after training."""
        model = MuseumRegressionModel()
        model.train(sample_dataframe)
        prediction = model.predict(5_000_000)
        assert prediction >= 0

    def test_load_from_params(self):
        """Test loading model from stored parameters."""
        model = MuseumRegressionModel()
        model.load_from_params(
            coefficient=0.5, intercept=1000.0, r2_score=0.85, mse=1_000_000.0, mae=500.0, n_samples=10
        )
        assert model.is_trained
        prediction = model.predict(2_000_000)
        assert prediction == 0.5 * 2_000_000 + 1000.0

    def test_get_equation(self, sample_dataframe):
        """Test equation string generation."""
        model = MuseumRegressionModel()
        assert model.get_equation() == "Model not trained"
        model.train(sample_dataframe)
        equation = model.get_equation()
        assert "visitors" in equation
        assert "population" in equation

    def test_metrics_to_dict(self, sample_dataframe):
        """Test metrics dictionary conversion."""
        model = MuseumRegressionModel()
        metrics = model.train(sample_dataframe)
        metrics_dict = metrics.to_dict()
        assert "r2_score" in metrics_dict
        assert "coefficient" in metrics_dict


# =============================================================================
# API Tests
# =============================================================================


class TestAPI:
    """Tests for API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_env(self, tmp_path):
        """Set up temporary database for each test."""
        db_path = str(tmp_path / "test.db")
        os.environ["DATABASE_PATH"] = db_path
        yield
        # Clean up import cache
        import sys
        for mod in [k for k in sys.modules if k.startswith("museum_regression")]:
            del sys.modules[mod]

    @pytest.fixture
    def client(self):
        """Create test client."""
        from museum_regression.api.main import app
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def client_with_data(self, sample_museums, sample_cities):
        """Create test client with sample data."""
        from museum_regression.api.main import app
        with TestClient(app) as client:
            from museum_regression.api.main import db, model
            if db:
                db.populate_from_wikipedia(sample_museums, sample_cities)
                db.save_regression_result(
                    r2_score=0.85, mse=1_000_000.0, mae=500.0, coefficient=0.5, intercept=1000.0, n_samples=2
                )
                if model:
                    model.load_from_params(
                        coefficient=0.5, intercept=1000.0, r2_score=0.85, mse=1_000_000.0, mae=500.0, n_samples=2
                    )
            yield client

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_list_museums_empty(self, client):
        """Test listing museums when empty."""
        response = client.get("/museums")
        assert response.status_code == 200
        assert response.json()["count"] == 0

    def test_list_museums_with_data(self, client_with_data):
        """Test listing museums with data."""
        response = client_with_data.get("/museums")
        assert response.status_code == 200
        assert response.json()["count"] >= 1

    def test_predict_without_model(self, client):
        """Test prediction fails without model."""
        response = client.get("/regression/predict?population=1000000")
        assert response.status_code == 400

    def test_predict_with_model(self, client_with_data):
        """Test prediction with loaded model."""
        response = client_with_data.get("/regression/predict?population=5000000")
        assert response.status_code == 200
        assert "predicted_visitors" in response.json()

    def test_get_stats(self, client_with_data):
        """Test getting model stats."""
        response = client_with_data.get("/regression/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["r2_score"] == 0.85
        assert "equation" in data

"""Tests for API endpoints."""

import os

import pytest
from fastapi.testclient import TestClient


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

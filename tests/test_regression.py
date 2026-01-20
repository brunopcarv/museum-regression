"""Tests for the regression model."""

import pytest

from museum_regression.ml.regression import MuseumRegressionModel, RegressionMetrics


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

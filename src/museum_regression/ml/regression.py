"""
Linear regression model for correlating city population with museum visitors.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    """Data class for regression model metrics."""

    r2_score: float
    mse: float
    rmse: float
    mae: float
    coefficient: float
    intercept: float
    n_samples: int
    cv_scores: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "r2_score": self.r2_score,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "coefficient": self.coefficient,
            "intercept": self.intercept,
            "n_samples": self.n_samples,
            "cv_mean": (
                float(np.mean(self.cv_scores)) if self.cv_scores is not None else None
            ),
            "cv_std": (
                float(np.std(self.cv_scores)) if self.cv_scores is not None else None
            ),
        }


class MuseumRegressionModel:
    """
    Linear regression model for predicting museum visitors based on city population.
    """

    def __init__(self):
        """Initialize the regression model."""
        self.model: Optional[LinearRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Optional[RegressionMetrics] = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    def load_from_params(
        self,
        coefficient: float,
        intercept: float,
        r2_score: float,
        mse: float,
        mae: float,
        n_samples: int,
    ) -> None:
        """
        Load model from saved parameters (coefficient and intercept).

        This allows the model to make predictions without retraining,
        using parameters stored in the database.

        Args:
            coefficient: The regression coefficient (slope)
            intercept: The regression intercept
            r2_score: R-squared score from training
            mse: Mean squared error from training
            mae: Mean absolute error from training
            n_samples: Number of samples used in training
        """
        # Create a LinearRegression model and set its parameters directly
        self.model = LinearRegression()
        self.model.coef_ = np.array([coefficient])
        self.model.intercept_ = intercept

        # Store metrics
        self.metrics = RegressionMetrics(
            r2_score=r2_score,
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mae,
            coefficient=coefficient,
            intercept=intercept,
            n_samples=n_samples,
            cv_scores=None,
        )

        self._is_trained = True
        logger.info(
            f"Model loaded from params: coef={coefficient:.6f}, intercept={intercept:.2f}"
        )

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            df: DataFrame with 'population' and 'visitors' columns

        Returns:
            Tuple of (X, y) arrays
        """
        # Extract features and target
        X = df[["population"]].values
        y = df["visitors"].values

        return X, y

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = False,
    ) -> RegressionMetrics:
        """
        Train the linear regression model.

        Args:
            df: DataFrame with museum and city data
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            scale_features: Whether to scale features

        Returns:
            RegressionMetrics with model performance
        """
        logger.info(f"Training regression model on {len(df)} samples")

        X, y = self.prepare_data(df)

        # Optional feature scaling
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Split data
        if len(X) > 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            # Not enough data for split, use all data
            X_train, X_test, y_train, y_test = X, X, y, y

        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Cross-validation (if enough samples)
        cv_scores = None
        if len(X) >= 5:
            cv_scores = cross_val_score(
                LinearRegression(), X, y, cv=min(5, len(X)), scoring="r2"
            )

        # Store metrics
        self.metrics = RegressionMetrics(
            r2_score=r2,
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mae,
            coefficient=float(self.model.coef_[0]),
            intercept=float(self.model.intercept_),
            n_samples=len(df),
            cv_scores=cv_scores,
        )

        self._is_trained = True
        logger.info(f"Model trained: R² = {r2:.4f}, RMSE = {np.sqrt(mse):,.0f}")

        return self.metrics

    def predict(self, population: int) -> float:
        """
        Predict museum visitors based on city population.

        Args:
            population: City population

        Returns:
            Predicted annual visitors
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")

        X = np.array([[population]])
        if self.scaler:
            X = self.scaler.transform(X)

        prediction = self.model.predict(X)[0]
        return max(0, prediction)  # Ensure non-negative

    def predict_batch(self, populations: list[int]) -> list[float]:
        """
        Predict visitors for multiple cities.

        Args:
            populations: List of city populations

        Returns:
            List of predicted annual visitors
        """
        return [self.predict(pop) for pop in populations]

    def get_equation(self) -> str:
        """
        Get the regression equation as a string.

        Returns:
            String representation of the linear equation
        """
        if not self._is_trained:
            return "Model not trained"

        coef = self.model.coef_[0]
        intercept = self.model.intercept_

        sign = "+" if intercept >= 0 else "-"
        return f"visitors = {coef:.4f} × population {sign} {abs(intercept):,.0f}"

    def plot_regression(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a scatter plot with regression line.

        Args:
            df: DataFrame with museum data
            figsize: Figure size
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(
            df["population"] / 1_000_000,
            df["visitors"] / 1_000_000,
            alpha=0.7,
            s=100,
            label="Museums",
        )

        # Regression line
        x_range = np.linspace(
            df["population"].min(), df["population"].max(), 100
        ).reshape(-1, 1)

        if self.scaler:
            x_scaled = self.scaler.transform(x_range)
            y_pred = self.model.predict(x_scaled)
        else:
            y_pred = self.model.predict(x_range)

        ax.plot(
            x_range / 1_000_000,
            y_pred / 1_000_000,
            color="red",
            linewidth=2,
            label=f"Regression (R² = {self.metrics.r2_score:.3f})",
        )

        # Labels
        ax.set_xlabel("City Population (millions)", fontsize=12)
        ax.set_ylabel("Annual Museum Visitors (millions)", fontsize=12)
        ax.set_title(
            "Museum Visitors vs City Population\nLinear Regression Analysis",
            fontsize=14,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add equation annotation
        ax.annotate(
            self.get_equation(),
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        return fig

    def plot_residuals(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create residual diagnostic plots.

        Args:
            df: DataFrame with museum data
            figsize: Figure size
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before plotting")

        X, y = self.prepare_data(df)
        if self.scaler:
            X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        residuals = y - y_pred

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Residuals vs Predicted
        axes[0].scatter(y_pred / 1_000_000, residuals / 1_000_000, alpha=0.7)
        axes[0].axhline(y=0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted Visitors (millions)")
        axes[0].set_ylabel("Residuals (millions)")
        axes[0].set_title("Residuals vs Predicted Values")
        axes[0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1].hist(residuals / 1_000_000, bins=15, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Residuals (millions)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Residuals")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Residual plot saved to {save_path}")

        return fig

    def get_feature_importance(self) -> dict:
        """
        Get feature importance (coefficient magnitude for linear regression).

        Returns:
            Dictionary with feature importance
        """
        if not self._is_trained:
            return {}

        return {
            "population": {
                "coefficient": float(self.model.coef_[0]),
                "interpretation": (
                    f"For every 1 person increase in population, "
                    f"museum visitors increase by {self.model.coef_[0]:.4f}"
                ),
            }
        }

    def summary(self) -> str:
        """
        Get a text summary of the model.

        Returns:
            Summary string
        """
        if not self._is_trained:
            return "Model not trained"

        lines = [
            "=" * 50,
            "MUSEUM VISITOR REGRESSION MODEL SUMMARY",
            "=" * 50,
            f"\nSamples: {self.metrics.n_samples}",
            f"\nModel Equation:",
            f"  {self.get_equation()}",
            f"\nPerformance Metrics:",
            f"  R² Score:    {self.metrics.r2_score:.4f}",
            f"  RMSE:        {self.metrics.rmse:,.0f} visitors",
            f"  MAE:         {self.metrics.mae:,.0f} visitors",
        ]

        if self.metrics.cv_scores is not None:
            lines.extend(
                [
                    f"\nCross-Validation (5-fold):",
                    f"  Mean R²:     {np.mean(self.metrics.cv_scores):.4f}",
                    f"  Std R²:      {np.std(self.metrics.cv_scores):.4f}",
                ]
            )

        lines.extend(
            [
                f"\nInterpretation:",
                f"  {self.get_feature_importance()['population']['interpretation']}",
                "=" * 50,
            ]
        )

        return "\n".join(lines)

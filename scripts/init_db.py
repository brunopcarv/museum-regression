#!/usr/bin/env python3
"""
Database initialization script for museum data.

This script is run by the data-ingestion container to:
1. Populate the database with museum and city data from Wikipedia
2. Train the regression model and save parameters to the database

It only runs if the database doesn't already have data.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    db_path = os.environ.get("DATABASE_PATH", "/data/museum_data.db")
    min_visitors = int(os.environ.get("MIN_VISITORS", 2_000_000))

    # Import here to avoid unnecessary imports if skipping
    from museum_regression.db.database import Database

    # Initialize database (creates tables if not exist)
    logger.info(f"Initializing database at {db_path}")
    db = Database(db_path)
    db.init_db()

    # Check if database already has data
    df = db.get_museum_city_data()

    if len(df) > 0:
        # Check if model params exist
        existing_result = db.get_latest_regression_result()
        if existing_result:
            logger.info(
                f"Database already contains {len(df)} records and trained model."
            )
            logger.info(
                f"Model params: coef={existing_result.coefficient:.6f}, R²={existing_result.r2_score:.4f}"
            )
            logger.info("Skipping initialization.")
            return 0
        else:
            logger.info(
                f"Database has {len(df)} records but no model. Training model..."
            )
    else:
        logger.info("Database is empty. Fetching data from Wikipedia...")

        # Fetch data from Wikipedia
        from museum_regression.data.wikipedia_client import WikipediaClient

        wiki_client = WikipediaClient(min_visitors=min_visitors)

        try:
            museums = wiki_client.fetch_museums()
            if not museums:
                logger.error("No museums found from Wikipedia")
                return 1

            logger.info(f"Found {len(museums)} museums")

            logger.info("Fetching city populations...")
            cities = wiki_client.fetch_city_populations(museums)
            logger.info(f"Found population data for {len(cities)} cities")

            logger.info("Populating database...")
            museums_added, cities_added = db.populate_from_wikipedia(museums, cities)
            logger.info(
                f"Successfully added {museums_added} museums and {cities_added} cities"
            )

            # Reload data
            df = db.get_museum_city_data()
            logger.info(f"Database now contains {len(df)} records")

        except Exception as e:
            logger.error(f"Failed to fetch data from Wikipedia: {e}")
            return 1

    # Train regression model
    if len(df) >= 2:
        logger.info("Training regression model...")
        from museum_regression.ml.regression import MuseumRegressionModel

        model = MuseumRegressionModel()
        metrics = model.train(df)

        logger.info(
            f"Model trained: R² = {metrics.r2_score:.4f}, RMSE = {metrics.rmse:,.0f}"
        )
        logger.info(f"Equation: {model.get_equation()}")

        # Save model params to database
        logger.info("Saving model parameters to database...")
        db.save_regression_result(
            r2_score=metrics.r2_score,
            mse=metrics.mse,
            mae=metrics.mae,
            coefficient=metrics.coefficient,
            intercept=metrics.intercept,
            n_samples=metrics.n_samples,
        )
        logger.info("Model parameters saved successfully.")
    else:
        logger.warning(f"Not enough data to train model (need >= 2, have {len(df)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

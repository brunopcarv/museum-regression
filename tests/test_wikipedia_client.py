"""Integration tests for Wikipedia web scraping."""

import pytest

from museum_regression.data.wikipedia_client import WikipediaClient


@pytest.mark.integration
class TestWikipediaClientIntegration:
    """Integration tests that verify web scraping against Wikipedia pages."""

    def test_fetch_all_museums_from_wikipedia(self):
        """Test fetching all museum data from the actual Wikipedia page."""
        client = WikipediaClient(min_visitors=0)
        museums = client.fetch_museums()

        # Should find exactly 63 museums in the table
        # Based on: https://en.wikipedia.org/wiki/List_of_most-visited_museums
        assert len(museums) == 63, (
            f"Expected 63 museums in the table, found {len(museums)}"
        )

        # Verify each museum has required fields populated
        for museum in museums:
            assert museum.name
            assert museum.city
            assert museum.visitors > 0

        # Check if Louvre is in the list
        museum_names = [m.name.lower() for m in museums]
        assert any(
            "louvre" in name for name in museum_names
        )

    def test_fetch_museums_with_2m_plus_visitors(self):
        """Test fetching museums with >= 2M annual visitors."""
        client = WikipediaClient(min_visitors=2_000_000)
        museums = client.fetch_museums()

        # Should find exactly 42 museums with >= 2M visitors
        # Based on: https://en.wikipedia.org/wiki/List_of_most-visited_museums
        assert len(museums) == 42

        # Verify all museums meet the threshold
        for museum in museums:
            assert museum.visitors >= 2_000_000

    def test_fetch_city_population_from_wikipedia(self):
        """Test fetching city population data from actual Wikipedia pages."""
        client = WikipediaClient()

        city = client.get_city_population("Paris", "France")
        assert city is not None
        assert city.name == "Paris"
        assert city.country == "France"
        assert city.population == 2_048_472 # data from https://en.wikipedia.org/wiki/Paris

    def test_fetch_all_city_populations_from_wikipedia(self):
        """Test fetching population data for all cities hosting museums."""
        client = WikipediaClient(min_visitors=0)
        museums = client.fetch_museums()

        # Should have 63 museums
        assert len(museums) == 63

        # Get unique (city, country) pairs from museums
        unique_city_country_pairs = {(m.city, m.country) for m in museums}

        # Fetch population data for all cities
        cities = client.fetch_city_populations(museums)

        # Assert unique (city, country) pairs is 38
        assert len(unique_city_country_pairs) == 38

        # Assert cities with population data matches unique (city, country) pairs
        assert len(cities) == len(unique_city_country_pairs)

        # Verify each city has valid population data
        for city in cities.values():
            assert city.name
            assert city.population > 0

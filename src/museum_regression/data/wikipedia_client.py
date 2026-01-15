"""
Wikipedia API client for fetching museum and city population data.

Uses the Wikipedia API to retrieve information about the most visited museums
and population data for their host cities.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
MUSEUM_LIST_PAGE = "List_of_most-visited_museums"


@dataclass
class Museum:
    """Data class representing a museum."""

    name: str
    city: str
    country: str
    visitors: int  # Annual visitors
    year: int  # Year of visitor count
    museum_type: str = ""
    wikipedia_url: str = ""


@dataclass
class City:
    """Data class representing a city with population data."""

    name: str
    country: str
    population: int
    population_year: Optional[int] = None
    wikipedia_url: str = ""


class WikipediaClient:
    """Client for fetching museum and city data from Wikipedia API."""

    def __init__(self, min_visitors: int = 2_000_000):
        """
        Initialize the Wikipedia client.

        Args:
            min_visitors: Minimum annual visitors threshold for museums
        """
        self.min_visitors = min_visitors
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "MuseumPopulationRegression/0.1 (research project)"}
        )

    def _api_request(self, params: dict) -> dict:
        """Make a request to Wikipedia API."""
        params["format"] = "json"
        response = self.session.get(WIKIPEDIA_API_URL, params=params)
        response.raise_for_status()
        return response.json()

    def _parse_number(self, text: str) -> Optional[int]:
        """Parse a number from text, handling commas and various formats."""
        if not text:
            return None
        # Remove footnote references like [1], [2], etc.
        text = re.sub(r"\[.*?\]", "", text)
        # Remove commas and spaces
        text = text.replace(",", "").replace(" ", "").strip()
        # Extract the first number found
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def get_museum_list_html(self) -> str:
        """Fetch the HTML content of the museum list page."""
        params = {
            "action": "parse",
            "page": MUSEUM_LIST_PAGE,
            "prop": "text",
        }
        data = self._api_request(params)
        return data["parse"]["text"]["*"]

    def parse_museum_table(self, html: str) -> list[Museum]:
        """
        Parse the museum table from Wikipedia HTML.

        Args:
            html: HTML content of the museum list page

        Returns:
            List of Museum objects meeting the visitor threshold
        """
        soup = BeautifulSoup(html, "html.parser")
        museums = []

        # Find all tables with class 'wikitable'
        tables = soup.find_all("table", class_="wikitable")

        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue

            # Check if this is the right table by looking at headers
            header_row = rows[0]
            headers = [
                th.get_text(strip=True).lower()
                for th in header_row.find_all(["th", "td"])
            ]

            # Look for museum-related headers
            if not any("museum" in h or "name" in h for h in headers):
                continue

            # Find column indices
            name_idx = next(
                (i for i, h in enumerate(headers) if "museum" in h or "name" in h), 0
            )
            city_idx = next(
                (i for i, h in enumerate(headers) if "city" in h or "location" in h), 1
            )
            visitors_idx = next(
                (i for i, h in enumerate(headers) if "visitor" in h), -1
            )

            if visitors_idx == -1:
                continue

            # Parse data rows
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) <= max(name_idx, city_idx, visitors_idx):
                    continue

                try:
                    # Extract museum name
                    name_cell = cells[name_idx]
                    name_link = name_cell.find("a")
                    name = (
                        name_link.get_text(strip=True)
                        if name_link
                        else name_cell.get_text(strip=True)
                    )
                    wiki_url = (
                        f"https://en.wikipedia.org{name_link['href']}"
                        if name_link and name_link.get("href")
                        else ""
                    )

                    # Extract city and country
                    location_cell = cells[city_idx]
                    location_text = location_cell.get_text(strip=True)
                    # Parse city, country (usually comma-separated)
                    location_parts = location_text.split(",")
                    city = (
                        location_parts[0].strip() if location_parts else location_text
                    )
                    country = (
                        location_parts[-1].strip() if len(location_parts) > 1 else ""
                    )

                    # Extract visitor count
                    visitors_text = cells[visitors_idx].get_text(strip=True)
                    visitors = self._parse_number(visitors_text)

                    if visitors and visitors >= self.min_visitors:
                        museum = Museum(
                            name=name,
                            city=city,
                            country=country,
                            visitors=visitors,
                            year=2023,  # Default, can be parsed from table if available
                            wikipedia_url=wiki_url,
                        )
                        museums.append(museum)
                        logger.info(
                            f"Found museum: {name} in {city} with {visitors:,} visitors"
                        )

                except (IndexError, ValueError) as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue

        return museums

    def fetch_museums(self) -> list[Museum]:
        """
        Fetch the list of most visited museums from Wikipedia.

        Returns:
            List of Museum objects with >2M annual visitors
        """
        logger.info(f"Fetching museums with >= {self.min_visitors:,} annual visitors")
        html = self.get_museum_list_html()
        museums = self.parse_museum_table(html)
        logger.info(f"Found {len(museums)} museums meeting criteria")
        return museums

    def get_city_population(self, city: str, country: str) -> Optional[City]:
        """
        Fetch population data for a city from Wikipedia.

        Args:
            city: City name
            country: Country name

        Returns:
            City object with population data, or None if not found
        """
        # Try different search variations
        search_queries = [
            f"{city}",
            f"{city} ({country})",
            f"{city}, {country}",
        ]

        for query in search_queries:
            try:
                # Search for the city page
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 1,
                }
                data = self._api_request(params)

                if not data["query"]["search"]:
                    continue

                page_title = data["query"]["search"][0]["title"]

                # Get the page content
                params = {
                    "action": "parse",
                    "page": page_title,
                    "prop": "text",
                }
                data = self._api_request(params)
                html = data["parse"]["text"]["*"]

                # Parse population from infobox
                population = self._extract_population_from_html(html)

                if population:
                    return City(
                        name=city,
                        country=country,
                        population=population,
                        wikipedia_url=f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                    )

            except Exception as e:
                logger.warning(f"Error fetching population for {city}: {e}")
                continue

        return None

    def _extract_population_from_html(self, html: str) -> Optional[int]:
        """Extract population from Wikipedia city page HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Look for population in infobox
        infobox = soup.find("table", class_="infobox")
        if not infobox:
            return None

        # Search for rows containing "population"
        for row in infobox.find_all("tr"):
            header = row.find("th")
            if header and "population" in header.get_text().lower():
                # Get the next row or the value in current row
                data = row.find("td")
                if data:
                    pop = self._parse_number(data.get_text())
                    if pop and pop > 10000:  # Sanity check
                        return pop

                # Check next sibling row for the actual number
                next_row = row.find_next_sibling("tr")
                if next_row:
                    data = next_row.find("td")
                    if data:
                        pop = self._parse_number(data.get_text())
                        if pop and pop > 10000:
                            return pop

        # Fallback: search for any large number near "population" text
        text = soup.get_text()
        pop_match = re.search(
            r"population[^\d]*(\d{1,3}(?:,\d{3})+|\d+)", text, re.IGNORECASE
        )
        if pop_match:
            pop = self._parse_number(pop_match.group(1))
            if pop and pop > 10000:
                return pop

        return None

    def fetch_city_populations(self, museums: list[Museum]) -> dict[str, City]:
        """
        Fetch population data for all cities hosting museums.

        Args:
            museums: List of museums

        Returns:
            Dictionary mapping city names to City objects
        """
        cities = {}
        unique_locations = {(m.city, m.country) for m in museums}

        for city_name, country in unique_locations:
            if city_name in cities:
                continue

            logger.info(f"Fetching population for {city_name}, {country}")
            city = self.get_city_population(city_name, country)

            if city:
                cities[city_name] = city
                logger.info(f"  Population: {city.population:,}")
            else:
                logger.warning(f"  Could not find population data")

        return cities

"""Tests for Wikipedia data fetching."""

from unittest.mock import patch

from museum_regression.data.wikipedia_client import City, Museum, WikipediaClient


class TestWikipediaClient:
    """Tests for Wikipedia data fetching."""

    def test_museum_dataclass_creation(self):
        """Test Museum dataclass creation with defaults."""
        museum = Museum(
            name="Louvre",
            city="Paris",
            country="France",
            visitors=7_800_000,
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
            Museum(name="Test", city="City", country="Country", visitors=5_000_000)
        ]
        client = WikipediaClient()
        museums = client.fetch_museums()
        assert len(museums) == 1
        mock_get_html.assert_called_once()

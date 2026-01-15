# Museum Population Regression

Correlates museum visitor attendance with city populations using Wikipedia data and linear regression.

## Quick Start

```bash
docker compose up --build -d
```

**Access:**
- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Jupyter: http://localhost:8888 (token: `museum_regression`)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/museums` | GET | List all museums |
| `/regression/predict` | GET | Predict visitors for a population |
| `/regression/stats` | GET | Model statistics |

## Development

```bash
uv sync --all-extras
uv run pytest tests/ -v
```

## Data Source

[Wikipedia - List of most visited museums](https://en.wikipedia.org/wiki/List_of_most_visited_museums)

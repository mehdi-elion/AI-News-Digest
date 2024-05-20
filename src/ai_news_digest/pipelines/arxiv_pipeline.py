import time
from typing import Any  # noqa: N999

import dlt
import feedparser
from dlt.sources.helpers import requests
from loguru import logger

BASE_ARXIV_URL = "http://export.arxiv.org/api"

QUERIES = [
    "ti:deep AND ti:learning",
    "ti:transformer",
    "abs:explainability AND abs:neural",
    "abs:interpretability AND abs:neural",
]


def fetch_arxiv_articles(query: str, endpoint: str = "query"):
    url = f"{BASE_ARXIV_URL}/{endpoint}"
    params_query = {"search_query": query, "start": 0, "max_results": 100}

    try:
        response = requests.get(url=url, params=params_query)
        response.raise_for_status()
        d = feedparser.parse(response.text)
        serializable_d_entries = [{k: v for k, v in e.items() if not k.endswith("parsed")} for e in d.entries]

        time.sleep(3)

        yield from serializable_d_entries

    except requests.HTTPError as e:
        logger.exception(e)


@dlt.source
def arxiv_source() -> Any:
    for query in QUERIES:
        yield dlt.resource(
            fetch_arxiv_articles(query=query, endpoint="query"),
            name=f"arxiv_articles_{query}",
            write_disposition="merge",
            primary_key="id",
        )


def load_arxiv() -> None:
    pipeline = dlt.pipeline(
        pipeline_name="arxiv_api",
        destination="duckdb",
        dataset_name="arxiv_api_data",
    )

    load_info = pipeline.run(arxiv_source())
    print(load_info)


if __name__ == "__main__":
    load_arxiv()

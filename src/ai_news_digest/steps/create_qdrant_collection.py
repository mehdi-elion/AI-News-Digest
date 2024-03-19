import json
from pathlib import Path
from typing import Any

import polars as pl
from ai_news_digest.utils import get_root_project_path
from loguru import logger
from qdrant_client import QdrantClient

if __name__ == "__main__":
    # Retrieve root project path
    root_path = get_root_project_path()
    if root_path is None:
        logger.error("Could not find root project path.")
        raise SystemExit(1)
    root_path = Path(root_path)

    # Read arxiv data
    data_path = root_path / "data" / "03_primary"
    arxiv_results: dict[str, Any] = {}
    for file in data_path.iterdir():
        if file.suffix == ".json" and file.name.startswith("arxiv_dict"):
            with open(file, "r") as f:
                info_dict = json.load(f)["results"]
                arxiv_results |= info_dict
    logger.info(f"{len(arxiv_results)} documents found in arxiv data.")
    df = (
        pl.from_dict(arxiv_results)
        .transpose(include_header=True, header_name="url")
        .unnest("column_0")
        .select(
            pl.exclude("date").exclude("category"),
            pl.col("category").cast(pl.Categorical),
            pl.col("date").str.to_datetime(format="%Y-%m-%d %H:%M:%S%z"),
        )
    )

    # Connect to vector database
    client = QdrantClient(location="localhost", port=6333)
    client.close()

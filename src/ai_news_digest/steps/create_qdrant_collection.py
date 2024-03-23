import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from ai_news_digest.utils import get_root_project_path
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich import print
from sentence_transformers import SentenceTransformer

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
    # Retrieve encoder + setup
    encoder = SentenceTransformer("BAAI/bge-small-en")
    encoder_dim = encoder.get_sentence_embedding_dimension()

    # Connect to vector database
    qdrant = QdrantClient(location="localhost", port=6333)
    # we use recreate_collection so that we can iterate on the script
    qdrant.recreate_collection(
        collection_name="arxiv_collection", vectors_config=VectorParams(size=encoder_dim, distance=Distance.COSINE)
    )
    vectors = np.array([encoder.encode(text) for text in df["abstract"].to_numpy()])
    payload = df.to_dicts()
    ids = np.arange(len(vectors))
    logger.info(f"Starting upload of {len(vectors)} vectors to Qdrant.")
    qdrant.upload_collection(collection_name="arxiv_collection", vectors=vectors, payload=payload)
    logger.info("Upload complete.")

    # query database with similarity search
    query = "Find papers around the topic of VIT or commonly known as Vision Transformers."
    hits = qdrant.search(collection_name="arxiv_collection", query_vector=encoder.encode(query), limit=5)
    width, _ = shutil.get_terminal_size()
    for idx, hit in enumerate(hits):
        print(f"Hit {idx + 1}:")
        print("=" * width)
        print(f"Cosine similarity: {hit.score}")
        print(hit.payload)
        print("")

    # cleanup
    qdrant.close()

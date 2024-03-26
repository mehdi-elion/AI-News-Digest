import datetime  # noqa: D100
import json
from pathlib import Path
from typing import Any

import arxiv
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

# search params
QUERIES = [
    "ti:deep AND ti:learning",
    "ti:transformer",
    "abs:explainability AND abs:neural",
    "abs:interpretability AND abs:neural",
]
MAX_RESULTS = 100
OUTPUT_FOLDER = "data/03_primary/"


# TODO: make it a CLI
if __name__ == "__main__":
    # get today's date
    now = str(datetime.datetime.now()).split(".")[0]
    now = now.replace(" ", "_").replace(":", "-")

    # create output file name
    output_filename = Path(OUTPUT_FOLDER) / f"arxiv_dict_{now}.json"
    logger.info(f"Will run arxiv search and output results in {output_filename}...")

    # init res dist
    res_dict = {}

    # iterate over queries
    logger.info("Proceeding to individual queries...")

    progress = Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        SpinnerColumn(),
    )

    n_queries = len(QUERIES)
    with progress:
        retrieve_data_from_arxiv = progress.add_task("Running queries on Arxiv...", total=n_queries)
        for i, query in enumerate(QUERIES):
            progress.console.print(f"Running query '{query}' ({i+1}/{n_queries})...")
            search = arxiv.Search(
                query=query,
                max_results=MAX_RESULTS,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            progress.update(retrieve_data_from_arxiv, advance=1)

            # parse results
            for result in search.results():
                res_dict[result.entry_id] = {
                    "title": result.title,
                    "abstract": result.summary,
                    "date": str(result.published),
                    "category": result.primary_category,
                }

    # store results
    over_dict: dict[str, Any] = {}
    over_dict["queries"] = QUERIES
    over_dict["results"] = res_dict
    with open(output_filename, "w", encoding="utf8") as f:
        json.dump(over_dict, f, ensure_ascii=False)
    logger.success(f"Successfully saved results to '{output_filename}' !")

import datetime
import json
from pathlib import Path
from typing import Any, Dict

import arxiv
from loguru import logger
from tqdm import tqdm

# search params
QUERIES = [
    "ti:deep AND ti:learning",
    "ti:transformer",
    "abs:explainability AND abs:neural",
    "abs:interpretability AND abs:neural",
]
MAX_RESULTS = 4
OUTPUT_FOLDER = "data/03_primary/"


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
    for i, query in enumerate(tqdm(QUERIES)):
        # run query via arxiv API
        logger.info(f"Running query '{query}' ({i+1}/{len(QUERIES)})...")
        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        # parse results
        for result in search.results():
            res_dict[result.entry_id] = {
                "title": result.title,
                "abstract": result.summary,
                "date": str(result.published),
                "category": result.primary_category,
            }

    # store results
    over_dict: Dict[str, Any] = {}
    over_dict["queries"] = QUERIES
    over_dict["results"] = res_dict
    with open(output_filename, "w") as f:
        json.dump(over_dict, f)
    logger.success(f"Successfully saved results to '{output_filename}' !")

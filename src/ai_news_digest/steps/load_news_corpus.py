import json
from datetime import datetime, timedelta
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import newspaper
import pandas as pd
import yaml
from gnews import GNews
from loguru import logger
from newsapi import NewsApiClient

OUTPUT_FOLDER = "data/03_primary/"


def load_credentials(path_to_creds: str = "conf/local/credentials.yml") -> Dict[str, Any]:
    """Load credentials from specified path.

    Parameters
    ----------
    path_to_creds : str, optional
        Path to credential file (git untracked), by default "../conf/local/credentials.yml"

    Returns
    -------
    credentials: dict
        Dictionary containing various credentials

    """
    with open(path_to_creds, "r") as file:
        credentials: Dict[str, Any] = yaml.safe_load(file)

    return credentials


# TODO: combine with `newspaper3k` to load full article texts
def load_news_gnews(
    keywords: str,
    language: str = "en",
    country: Optional[str] = None,
    period: Optional[str] = None,  # e.g. '30d'
    start_date: Optional[Tuple[int, int, int]] = None,
    end_date: Optional[Tuple[int, int, int]] = None,
    exclude_websites: List[str] = [],
    max_results: Optional[int] = None,
    override_content: bool = False,
    use_logs: bool = False,
    standardize: bool = False,
) -> List[Dict[str, Any]]:
    """Load a set of news using the GNews library.

    More details about the arguments can be found at https://pypi.org/project/gnews/.

    Parameters
    ----------
    keywords : str
        Query to use for the search
    language : str, optional
        Language of the article, by default "en"
    country : Optional[str], optional
        Country related to the article, by default None
    period : Optional[str], optional
        Length of the time-wise sliding window used to filter articles;
        must be of the form '30d' (30 days), by default None
    start_date : Tuple[int, int, int] in (yyyy, mm, dd) format, optional
        Starting date of the time window, by default None
    end_date : Tuple[int, int, int], optional
        End date of the time window in (yyyy, mm, dd) format, by default None
    exclude_websites : str, optional
        List of websites to exclude, by default []
    max_results : int, optional
        Maximum number of results, by default None
    override_content : bool
        Whether to use newspaper3k to download the full article text and store
        it as article "content" in output, by default False.
    use_logs : bool
        Whether to use log messages, by default False.
    standardize : bool
        Whether to post-process results to have the match a given standard,
        by default False.

    Returns
    -------
    found_news: List[dict]
        List of found articles.

    """
    # set filters
    google_news = GNews(
        language=language,
        country=country,
        period=period,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
        exclude_websites=exclude_websites,
        proxy=None,
    )

    # run search
    found_news: List[Dict[str, Any]] = google_news.get_news(keywords)

    # fetch full text if available
    if override_content:
        if use_logs:
            logger.info("Fetching articles' full texts using newspaper3k...")
        for i, dico in enumerate(found_news):
            try:
                article = newspaper.Article(url=dico["url"])
                article.download()
                article.parse()
                found_news[i]["full_text"] = article.text
            except Exception as e:
                logger.warning(f"newspaper3k failed with error: {e}")

    # standardize
    if standardize:
        found_news = [
            {
                "url": d["url"],
                "title": d["title"],
                "content": d["full_text"] if "full_text" in d.keys() else None,
                "metadata": {
                    "query_engine": "gnews",
                    "top_image": article.top_image,
                    "description": d["description"],
                    "published_date": pd.to_datetime(d["published date"]),
                    "publisher": d["publisher"],
                },
            }
            for d in found_news
        ]

    # return found news
    return found_news


# TODO: combine with `newspaper3k` to load full article texts
# TODO: iterate over sources & domains to curb the 100-article limit (free-tier)
def load_news_newsapi(
    credentials: Dict[str, Any],
    query: Optional[str] = None,
    sources: List[str] = ["bbc-news", "the-verge"],
    domains: List[str] = ["apnews.com"],
    language: str = "en",
    period: Optional[str] = "30d",
    from_date: Optional[str] = None,  # YYYY-MM-DD format
    to_date: Optional[str] = None,  # YYYY-MM-DD format
    sort_by: str = "relevancy",
    override_content: bool = False,
    use_logs: bool = False,
    standardize: bool = False,
) -> List[Dict[str, Any]]:
    """Load a set of news using the NewsAPI library.

    Parameters
    ----------
    credentials : dict
        Local credentials used for NewsAPI authentication.
    query : str, optional
        Query used for the news search, by default None
    sources : List[str]
        List (possibly empty) of sources, by default ["bbc-news", "the-verge"]
    domains : List[str]
        List (possibly empty) of domains, by default ["apnews.com"]
    language : str, optional
        Language of the articles, by default "en"
    period : str, optional
        Length of the time-wise sliding window used to filter articles;
        must be of the form '30d' (30 days), by default '30d'.
        This argument overrides `from_date` and `to_date` when it is specified.
    from_date : str, optional
        Start date of the time window in '%Y-%m-%d' format, by default None
    to_date : str, optional
        End date of the time window in '%Y-%m-%d' format, by default None
    sort_by : str
        Sorting criteria, by default "relevancy". Options are : 'relevancy',
        'popularity' and 'publishedAt'
    override_content : bool
        Whether to use newspaper3k to download the full article text and store
        it as article "content" in output, by default False.
    use_logs : bool
        Whether to use log messages, by default False.
    standardize : bool
        Whether to post-process results to have the match a given standard,
        by default False.

    Returns
    -------
    results: List[dict]
        List of found articles

    Raises
    ------
    ValueError
        `period` doesn't follow the right format

    """
    # init client
    newsapi = NewsApiClient(api_key=credentials["news_api"]["key"])

    # 'period' mode: override if valid
    if period is not None:
        if not period.endswith("d"):
            raise ValueError("Invalid format for `period` argument. Must be of type `xxd`")

        else:
            days = int(period[:-1])
            now = datetime.now()
            from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = now.strftime("%Y-%m-%d")
            logger.info(
                "Detected valid value for `period` argument, "
                "`from_date` and `to_date` will be overwritten with "
                f"'{from_date}' and '{to_date}' respectively."
            )

    # get number of results
    total_results = None
    results = []
    page_idx = 1
    page_size = 20
    run_loop = True
    max_runtime_sec = 120.0
    max_results = 100
    t0 = time()
    if use_logs:
        logger.info("Fetching articles...")
    while run_loop and time() - t0 < max_runtime_sec:
        # run query for current
        all_articles = newsapi.get_everything(
            q=query,
            sources=",".join(sources),
            domains=",".join(domains),
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by=sort_by,
            page=page_idx,
            page_size=20,
        )

        # get number of total results
        total_results = all_articles["totalResults"]

        # store current results
        results += all_articles["articles"]

        # update indexes and booleans
        page_idx += 1
        if len(results) >= total_results or page_idx * page_size > max_results:
            run_loop = False

    if override_content:
        if use_logs:
            logger.info("Fetching articles' full texts using newspaper3k...")
        for i, dico in enumerate(results):
            try:
                article = newspaper.Article(url=dico["url"])
                article.download()
                article.parse()
                results[i]["full_text"] = article.text
            except Exception as e:
                logger.warning(f"newspaper3k failed with error: {e}")

    if standardize:
        results = [
            {
                "url": d["url"],
                "title": d["title"],
                "content": d["full_text"] if "full_text" in d.keys() else d["content"],
                "metadata": {
                    "query_engine": "newsapi",
                    "top_image": d["urlToImage"],
                    "description": d["description"],
                    "published_date": pd.to_datetime(d["publishedAt"]),
                    "source": d["source"],
                    "authors": [d["author"]],
                },
            }
            for d in results
        ]

    return results


# TODO: make it a CLI
if __name__ == "__main__":
    # get today's date
    now = str(datetime.now()).split(".")[0]
    now = now.replace(" ", "_").replace(":", "-")

    # create output file name
    output_filename = Path(OUTPUT_FOLDER) / f"NewsAPI_dict_{now}.json"
    logger.info(f"Will run news search and output results in {output_filename}...")

    # load credentials from local folder
    logger.info("Loading credentials...")
    credentials = load_credentials()

    # get news
    logger.info("Loading news using NewsApi")
    params = dict(
        query=None,
        sources=["bbc-news", "the-verge"],
        domains=["apnews.com", "reuters.com", "techcrunch.com"],
        language="en",
        period="30d",
        from_date=None,
        to_date=None,
        sort_by="relevancy",
        override_content=True,
        use_logs=True,
    )
    news_list = load_news_newsapi(
        credentials=credentials,
        **params,  # type: ignore
    )
    logger.info(f"Successfully found {len(news_list)} news articles !")

    # store results
    over_dict = {}
    over_dict["params"] = params
    over_dict["results"] = {dico["url"]: dico for dico in news_list}  # type: ignore
    with open(output_filename, "w", encoding="utf8") as f:
        json.dump(over_dict, f, ensure_ascii=False)
    logger.success(f"Successfully saved results to '{output_filename}' !")

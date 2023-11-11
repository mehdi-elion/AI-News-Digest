# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # API Trials

# %% [markdown]
# ## Setup

# %%
import json
import urllib
import urllib.request

import arxiv
import pandas as pd
import xmltodict
from rich import print

# %% [markdown]
# ## Arxiv API with `urllib`

# %%
# url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=3'
url = "http://export.arxiv.org/api/query?search_query=ti:deep+AND+ti:learning&start=0&max_results=3"
data = urllib.request.urlopen(url)
xml_res = data.read().decode("utf-8")
print(xml_res)

# %%
d = xmltodict.parse(xml_res)
print(d["feed"]["entry"])

# %%
print(d["feed"]["entry"][2]["summary"])

# %% [markdown]
# ## Arxiv API with `arxiv`

# %%
search = arxiv.Search(
    # query = "ti:quantum",
    # query = "abs:graph",
    # query = "ti:quantum+OR+abs:graph",
    query="ti:deep AND ti:learning",
    max_results=3,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

for result in search.results():
    print(f"--- {result.title} [{result.published}] ---")
    print(result.summary)

# %%
with open("../data/03_primary/arxiv_dict_2023-09-04_01-09-05.json", "r") as f:
    info_dict = json.load(f)["results"]

# %%
pd.DataFrame(info_dict).transpose().head()

# %%

# %% [markdown]
# ## Semantics Scholar API with `urllib`

# %%

# %%

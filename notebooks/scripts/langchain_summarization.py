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
# # Lanchain : Summarization

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Libraries

# %%
from typing import List, Optional, Union

import arxiv
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
import yaml
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
from rich import print
from sklearn.cluster import DBSCAN
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BigBirdPegasusForConditionalGeneration,
    PegasusTokenizerFast,
)

# %% [markdown]
# ### Check GPU availability

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cuda:0":
    print(torch.cuda.get_device_properties(device))
else:
    print(f"No cuda device found; running on {device}")

# %% [markdown]
# ## Load secrets

# %%
with open("../conf/local/hf_secrets.yml", "r") as f:
    hf_secrets = yaml.load(f, Loader=yaml.SafeLoader)

# %% [markdown]
# ## Load data

# %%
# run search query on arxiv
search = arxiv.Search(
    query="ti:LLM OR (ti:LARGE AND ti:LANGUAGE AND ti:MODEL)",
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

# display some results
for result in list(search.results())[:2]:
    print(f"--- {result.title} [{result.published}] ---")
    print(result.summary)

# %%

# %% [markdown]
# ## Cluster Articles

# %% [markdown]
# Interesting models to try:
# * [google/bigbird-pegasus-large-arxiv](https://huggingface.co/google/bigbird-pegasus-large-arxiv)
# * [allenai/led-large-16384-arxiv](https://huggingface.co/allenai/led-large-16384-arxiv)
# * [google/pegasus-arxiv](https://huggingface.co/google/pegasus-arxiv)
# * [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) (very interesting summarization results from that one)

# %% [markdown]
# Interesting links for embeddings & langchain:
# * [HuggingFaceBgeEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceBgeEmbeddings.html?highlight=device)
# * [HuggingFaceEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceEmbeddings.html?highlight=device)
# * [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html?highlight=device)

# %%
# Load model
model_name = "google/bigbird-pegasus-large-arxiv"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# %%
# get abstract
result = next(search.results())
summary = result.summary

# feed-forward pass
inputs = tokenizer([summary], max_length=4096, return_tensors="pt", truncation=True)
outputs = model(**inputs, return_dict=True)

# Generate Summary
# summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=15)
# model_summary = tokenizer.batch_decode(summary_ids,
# skip_special_tokens=True,
# clean_up_tokenization_spaces=False)[0]
# print(f"model_summary: '{model_summary}'")

# Retrieve embedding
cls_token_embed = outputs.encoder_last_hidden_state[:, 0, :]


# %%
def embed_pegasus(
    input_text: Union[str, List[str]],
    model: BigBirdPegasusForConditionalGeneration,
    tokenizer: PegasusTokenizerFast,
    batch_size: Optional[int] = None,
    device: str = "cpu",
) -> Union[np.ndarray, torch.Tensor]:
    model = model.to(device)

    dim = model.config.hidden_size

    if isinstance(input_text, str):
        inputs = tokenizer(
            [input_text],
            max_length=4096,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        outputs = model(**inputs.to(device), return_dict=True)
        cls_token_embeds = outputs.encoder_last_hidden_state[:, 0, :]

        del outputs
        del inputs
        torch.cuda.empty_cache()

    else:
        if batch_size is not None:
            indices = list(range(0, len(input_text), batch_size)) + [None]
            cls_token_embeds = torch.empty((0, dim), device=device)

            for i in range(len(indices) - 1):
                input_text_i = input_text[indices[i] : indices[i + 1]]
                inputs_i = tokenizer(
                    input_text_i,
                    max_length=4096,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                )
                outputs_i = model(**inputs_i.to(device), return_dict=True)
                cls_token_embeds_i = outputs_i.encoder_last_hidden_state[:, 0, :]
                cls_token_embeds = torch.cat((cls_token_embeds, cls_token_embeds_i), dim=0)

                del outputs_i
                del inputs_i
                torch.cuda.empty_cache()

        else:
            inputs = tokenizer(input_text, max_length=4096, return_tensors="pt", truncation=True)
            outputs = model(**inputs.to(device), return_dict=True)
            cls_token_embeds = outputs.encoder_last_hidden_state[:, 0, :]

            del outputs
            del inputs
            torch.cuda.empty_cache()

    return cls_token_embeds


# %%
# retrieve abstracts & metadata
abstracts = [result.summary for result in search.results()]
titles = [result.title for result in search.results()]

# embed abstracts
with torch.no_grad():
    embeddings = embed_pegasus(
        input_text=abstracts,
        model=model,
        tokenizer=tokenizer,
        batch_size=20,
        device=device,
    )

# convert to numpy
if device == "cpu":
    embeddings = embeddings.numpy()
else:
    embeddings = embeddings.detach().cpu().numpy()

# reduce dimension
reducer = umap.UMAP(n_components=2, n_neighbors=5, metric="cosine", min_dist=0.1)
umap_coords = reducer.fit_transform(embeddings)

# %%
# store results in a dataframe
df = pd.DataFrame(data=umap_coords, columns=[f"umap_{i}" for i in range(umap_coords.shape[1])])
df["title"] = titles

# display
df.head()

# %%
# viz
fig = px.scatter(df, x="umap_0", y="umap_1", hover_data="title")
fig.show()

# %%
# clustering with (H)DBSCAN

clusterer = DBSCAN(eps=0.8)
clusterer.fit(df[[col for col in df.columns if "umap" in col]])
df["cluster"] = clusterer.labels_

# viz
fig = px.scatter(df, x="umap_0", y="umap_1", hover_data="title", color="cluster")
fig.show()

# %% [markdown]
# ## Summarize clusters of articles

# %% [markdown]
# Interesting links:
# * [https://python.langchain.com/docs/integrations/retrievers/arxiv](https://python.langchain.com/docs/integrations/retrievers/arxiv)
# * [https://python.langchain.com/docs/additional_resources/tutorials](https://python.langchain.com/docs/additional_resources/tutorials)
# * [https://python.langchain.com/docs/use_cases/summarization](https://python.langchain.com/docs/use_cases/summarization)
# * [https://huggingface.co/facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
# * [https://python.langchain.com/docs/integrations/llms/openllm](https://python.langchain.com/docs/integrations/llms/openllm)

# %% [markdown]
# ### Load model

# %%
# load model
# See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
repo_id = "facebook/bart-large-cnn"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={
        "temperature": 0.15,
        "max_length": 100,
        # "device": device,
    },
    huggingfacehub_api_token=hf_secrets["hf_hub_token"],
)


# question = "Who won the FIFA World Cup in the year 1994? "
# template = """
# Question: {question}

# Answer:
# """

# prompt = PromptTemplate(template=template, input_variables=["question"])

# llm_chain = LLMChain(prompt=prompt, llm=llm)
# print(llm_chain.run(question))

# %% [markdown]
# ### Option 1: Stuff

# %% [markdown]
# ### Option 2: Map reduce

# %%
# build docs

cluster_idx = 0
docs = [
    Document(
        page_content=abstracts[i],
        metadata={"title": titles[i]},
    )
    for i in range(len(titles))
    if df.loc[df["title"] == titles[i], "cluster"].values[0] == cluster_idx
]

# %%
print(docs[:2])

# %%
# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# %%
# Reduce
reduce_template = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary of the main themes.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

# %%
# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# %%
print(map_reduce_chain.run(split_docs))

# %% [markdown]
# ### Option 3: Refine

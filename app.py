import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger
from rich import print
import plotly.express as px
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain_community.llms import VLLMOpenAI
import torch, random
from src.ai_news_digest.utils import check_gpu_availability, create_run_folder
from src.ai_news_digest.steps.clustering_utils import (
    clustering_pipeline,
    entropy,
    grid_search_umap_projection,
    grid_search_clustering,
)
import src.ai_news_digest.steps.load_news_corpus as lnc
import src.ai_news_digest.steps.load_arxiv_corpus as lac
import src.ai_news_digest.steps.llm_utils as llmu

# check GPU availablility
device = check_gpu_availability()

# set default values
MODEL_ID = "BAAI/bge-small-en"
MODEL_KWARGS = {"device": device}
ENCODE_KWARGS = {
    "normalize_embeddings": True,
    "batch_size": 16,
    "output_value": "sentence_embedding",
    "convert_to_numpy": True,
    "show_progress_bar": True,
}
UMAP_KWARGS = {
    'n_neighbors': 4, 
    'min_dist': 0.1, 
    'n_components': 2, 
    'metric': 'cosine', 
    'init': 'random'
}
CLUSTERING_KWARGS = {
    'min_cluster_size': 3,
    'min_samples': 3, 
    'max_cluster_size': None, 
    'cluster_selection_epsilon': 0.1
}

# define summarization prompt
summary_prompt_template = """<s>[INST]You will be provided with a list of news articles that you must summarize.
Write a concise summary focusing on the key pieces of information mentioned in the articles.
Make sure the summary covers all the issues tackled in the group of articles. Use less than 200 words.
Here is the list of articles:
"{text}"
CONCISE SUMMARY (less than 180 words): [/INST]"""

# define keyword-generation prompt
keyword_prompt_template = """<s>[INST]You will be provided with a summary made from a list of news articles.
Extract a list of keywords that best covers all pieces of information tackled in this articles.
Don't use brackets or any special characters. Use only commas to separate keywords.
Here is the summary of articles:
"{}"
CONCISE COMMA-SEPARATED LIST OF KEYWORDS: [/INST]"""

# define title-generation prompt
title_prompt_template = """<s>[INST]You will be provided with a summary made from a list of news articles.
Write a short title that best covers all pieces of information tackled in this articles.
Don't use brackets or any special characters. Don't provide any explanation or alternative title. 
Write only one unique title in less than 15 words.
Here is the summary of articles:
"{}"
UNIQUE TITLE (< 15 words): [/INST]"""

# set random seeds
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)

# resource caching
@st.cache_resource
def load_hf_embed_model(
    model_name: str, 
    model_kwargs: dict, 
    encode_kwargs: dict
):
    hf = HuggingFaceEmbeddings(
        model_name=MODEL_ID,
        model_kwargs={"device": device},
        encode_kwargs=ENCODE_KWARGS,
    )
    return hf

#------------------------------ APP ------------------------------

st.set_page_config(layout="wide")
st.title('AI-Powered News')

with st.sidebar:
    # selectors for news query
    st.header(
        "What's your topic of interest ?", 
        anchor=None, 
        help=None,
        divider=False
    )
    st.write(
        "Type keywords decribing your topic of intrest. Those keywords will be " \
        "used to search relevant news articles :"
    )
    keywords = st.text_area(
        label="Query", 
        value="Artificial Intelligence, Hardware, Large Language Models", 
        max_chars=None,
        key=None,
        # type="default",
        help=None,
        # autocomplete=None,
        on_change=None,
        args=None,
        kwargs=None,
        placeholder=None,
        disabled=False,
        label_visibility="visible"
    )

    st.write("Set the maximum article age to a certain number of days:")
    days = st.slider('Maximum article age (in days)', 0, 90, 45)

    st.write("Set the maximum number of articles resulting from the query:")
    max_results = st.slider('Maximum number of results', 50, 300, 150)

    generate_button = st.button("Generate News Digest", type="primary")

print(f"keywords = '{keywords}'")
print(f"days = {days}")
print(f"max_results = {max_results}")
print(f"generate_button = {generate_button}")


if generate_button:
    
    # fetch news articles
    with st.spinner('Fetching news articles...'):
        res_gnews = lnc.load_news_gnews_parallel(
            keywords=keywords,
            language="en",
            period=f"{days}d",
            start_date=None,
            end_date=None,
            exclude_websites=[],
            max_results=max_results,
            override_content=True,
            use_logs=False,
            standardize=True,
            n_jobs=5,
        )

    # store in dataframe
    with st.spinner('Cleaning data...'):
        df_data = pd.DataFrame({d["url"]: d for d in res_gnews}).transpose()
        df_data = df_data.dropna(subset=["content"], axis=0)
        st.dataframe(
            df_data, 
            hide_index=True, 
            column_order=("title", "content"),
            use_container_width=True,
        )
        print(df_data.shape)
        print(df_data.head())

    # embed data
    with st.spinner('Embedding articles...'):
        
        # load model with caching
        hf = load_hf_embed_model(
            model_name=MODEL_ID,
            model_kwargs={"device": device},
            encode_kwargs=ENCODE_KWARGS,
        )
        
        # compute embeddings & store in dataframe
        embeddings = np.array(hf.embed_documents(df_data["content"]))
        df_embed = pd.DataFrame(
            data=embeddings, 
            columns=[f"embed_{i}" for i in range(embeddings.shape[1])],
            index=df_data.index
        )
        print("df_embed: ", df_embed.reset_index(drop=True).head())

    # Clustering articles
    with st.spinner('Clustering articles...'):
        df_umap, clustering = clustering_pipeline(
            df_embed,
            UMAP_KWARGS,
            CLUSTERING_KWARGS,
            random_state=123,
            df_data=df_data,
        )
        X_cluster = df_umap[[c for c in df_umap.columns if "umap_" in c]]
    
    # viz
    df_umap["cluster"] = [str(elt) for elt in clustering.labels_]
    df_umap["noise"] = [int(elt==-1) for elt in clustering.labels_]
    fig = px.scatter(
        df_umap,
        x="umap_0",
        y="umap_1",
        hover_data=[
            "title",
            # "ID",
        ],
        color="cluster",
        symbol="noise",
        category_orders={"cluster": list(np.sort(pd.unique(clustering.labels_)).astype(str))},
    )
    st.plotly_chart(fig, use_container_width=True)

    # summarizing articles
    with st.spinner('Summarizing articles...'):

        # Connect to deployed vLLM API
        llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            # model_kwargs={"stop": ["."]},
        )
        
        # retrieve list of clusters
        clusters_list = sorted(df_umap["cluster"].unique().tolist())
        num_clusters = len(clusters_list)
        print(f"num_clusters = {num_clusters}")

        # init dataframe to store results
        df_analysis = pd.DataFrame(columns=["cluster_idx", "title", "keywords", "summary"])

        # iterate over clusters
        my_bar = st.progress(0, text="Summarizing clusters of articles")
        for k, idx in enumerate(clusters_list):
            my_bar.progress((k+1)/num_clusters, text=f"cluster n°{k}...")

            # query docs from current cluster
            df_cluster = df_umap.query("cluster==@idx").copy()

            # build langchain docs
            docs = [
                Document(
                    page_content=f"Article n°{i} -- {row['content']}",
                    metadata=row.to_dict(),
                )
                for i, row in df_cluster.iterrows()
            ]

            # TODO: pre-summarize each article if too long

            # summarize
            summary = llmu.summarize_recurse(llm, summary_prompt_template, docs)

            # generate a title
            title = llm.predict(title_prompt_template.format(summary))

            # generate keywords
            keywords = llm.predict(keyword_prompt_template.format(summary))
            keywords = [s.strip() for s in keywords.strip().split(",")]

            # store results
            df_analysis.loc[k, :] = np.array([[idx, title, keywords, summary]], dtype=object)
    
    # display summarized clusters
    st.dataframe(
        df_analysis, 
        hide_index=True, 
        use_container_width=True,
    )



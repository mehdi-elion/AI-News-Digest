import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger
from rich import print
from copy import deepcopy
import requests
import librosa
import plotly.express as px
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain_community.llms import VLLMOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain_core.prompts import format_document
from operator import itemgetter
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
from src.ai_news_digest.steps.llm_utils import (
    SUMMARY_PROMPT_TEMPLATE,
    KEYWORD_PROMPT_TEMPLATE,
    TITLE_PROMPT_TEMPLATE,
    RAG_TEMPLATE,
    DEFAULT_DOCUMENT_PROMPT,
)

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
TTS_URL = f"http://localhost:5002/api/tts"
TTS_PARAMS = {
    'speaker_id': "p241",
    'out_path': "/root/tts-output/hello.wav",
    # 'language_id': "en",
}

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

@st.cache_resource
def load_llm(
    openai_api_key: str="EMPTY",
    openai_api_base: str="http://localhost:8000/v1",
    model_name: str="mistralai/Mistral-7B-Instruct-v0.2",
) -> VLLMOpenAI:
    llm = VLLMOpenAI(
        openai_api_key=openai_api_key, 
        openai_api_base=openai_api_base, 
        model_name=model_name,
    )
    return llm


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
    max_results = st.slider('Maximum number of results', 20, 300, 150)

    generate_button = st.button(
        "Generate News Digest", 
        type="primary",
    )

clustering_is_done = 'df_data' in st.session_state
clustering_is_done = clustering_is_done and 'df_umap' in st.session_state
clustering_is_done = clustering_is_done and 'df_analysis' in st.session_state

if clustering_is_done:
    run_clustering = generate_button
else:
    run_clustering = generate_button

if run_clustering:
        
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
        st.session_state['df_data'] = df_data

if "df_data" in st.session_state:
    expander_news_data = st.expander("See fetched news articles")
    expander_news_data.dataframe(
        st.session_state["df_data"], 
        hide_index=True, 
        column_order=("title", "content"),
        use_container_width=True,
    )
    
if run_clustering:
        
    # embed data
    with st.spinner('Embedding articles...'):
        
        # load model with caching
        hf = load_hf_embed_model(
            model_name=MODEL_ID,
            model_kwargs={"device": device},
            encode_kwargs=ENCODE_KWARGS,
        )
        st.session_state["hf"] = hf
        
        # compute embeddings & store in dataframe
        embeddings = np.array(hf.embed_documents(df_data["content"]))
        df_embed = pd.DataFrame(
            data=embeddings, 
            columns=[f"embed_{i}" for i in range(embeddings.shape[1])],
            index=df_data.index
        )

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
    st.session_state["df_umap"] = df_umap
    st.session_state["fig"] = fig

if "fig" in st.session_state:
    st.plotly_chart(st.session_state["fig"], use_container_width=True)

if run_clustering:
    
    # summarizing articles
    with st.spinner('Summarizing clusters of articles...'):

        # Connect to deployed vLLM API
        llm = load_llm()
        st.session_state["llm"] = llm
        
        # retrieve list of clusters
        clusters_list = sorted(df_umap["cluster"].unique().tolist())
        num_clusters = len(clusters_list)

        # init dataframe to store results
        df_analysis = pd.DataFrame(columns=["cluster_idx", "title", "keywords", "summary"])

        # iterate over clusters
        my_bar = st.progress(0, text="Summarizing clusters of articles")
        for k, idx in enumerate(clusters_list):
            my_bar.progress((k)/(num_clusters-1), text=f"cluster n°{k}/{num_clusters-1}...")

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
            summary = llmu.summarize_recurse(llm, SUMMARY_PROMPT_TEMPLATE, docs)

            # generate a title
            title = llm.predict(TITLE_PROMPT_TEMPLATE.format(summary))

            # generate keywords
            keywords = llm.predict(KEYWORD_PROMPT_TEMPLATE.format(summary))
            keywords = [s.strip() for s in keywords.strip().split(",")]

            # store results
            df_analysis.loc[k, :] = np.array([[idx, title, keywords, summary]], dtype=object)
        
        # remove progress bar
        my_bar.empty()
    
    # store restult in session
    st.session_state['df_analysis'] = df_analysis

if "df_analysis" in st.session_state:
    
    # display summarized clusters
    expander_clusters_data = st.expander("See obtained clusters")
    expander_clusters_data.dataframe(
        st.session_state["df_analysis"], 
        hide_index=True, 
        use_container_width=True,
    )


if 'df_analysis' in st.session_state and "df_umap" in st.session_state:

    # retrieve dfs & hf
    df_analysis = st.session_state["df_analysis"]
    df_umap = st.session_state["df_umap"]
    hf = st.session_state["hf"]
    
    # choose cluster for RAG & TTS
    cluster_choice = st.selectbox(
        'Select a cluster for RAG & TTS:',
        (row["title"] for i, row in df_analysis.iterrows()),
        index=0,
    )
    cluster_idx = list(df_analysis.title.values.flatten()).index(cluster_choice)
    cluster_idx0 = df_analysis.index[cluster_idx]   # re-indexing if "-1" cluster
    cluster_idx = df_analysis.loc[cluster_idx0, "cluster_idx"]
    if 'cluster_choice' not in st.session_state:
        st.session_state["cluster_choice"] = cluster_choice
    else:
        if st.session_state["cluster_choice"] != cluster_choice:
            st.session_state["cluster_choice"] = cluster_choice
            if "chat_history" in st.session_state:
                del st.session_state["chat_history"]

    # create separate tabs for RAG and TTS
    tab1, tab2 = st.tabs(["💬 RAG", "🔊 TTS"])

    # RAG tab
    tab1.subheader("Chat with your News !")
    messages = tab1.container(height=500)

    if cluster_choice:

        # retrieve cluster docs
        df_cluster = df_umap[df_umap["cluster"]==cluster_idx]
        docs = [
            Document(
                page_content=f"Article n°{i} -- {row['content']}",
                metadata=row.to_dict(),
            )
            for i, row in df_cluster.iterrows()
        ]
        
        # chunk & embed docs
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        split_docs_embeddings = hf.embed_documents([d.page_content for d in split_docs])

        # create and feed vectorestore
        qdrant = Qdrant.from_documents(
            split_docs,
            hf,
            collection_name="trial_rag_gnews",
            #---- IN-MEMORY, FLUSHED WHEN FINISHED ----
            location=":memory:",  # Local mode with in-memory storage only
            # #---- ON-DISK PERSISTING ----
            # path="/tmp/local_qdrant",
            # #---- QDRANT SERVER ----
            # url="http://localhost:6333/",
            # prefer_grpc=True,
        )

        # define langchain retriever
        retriever = qdrant.as_retriever(search_kwargs=dict(k=5))
        
        # define ptompt template
        prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

        # define RAG chain
        chain = (
            {
                "context": itemgetter("question") 
                    | retriever 
                    | llmu._combine_documents,
                "question": itemgetter("question"),
            }
            | prompt
            | st.session_state["llm"]
            | StrOutputParser()
        )

        # set question  
        question = tab1.chat_input("Ask anything about the chosen cluster:")
        messages.chat_message("assistant").write("Hello, what would you like to know ?")

        if "chat_history" in st.session_state:
            print("st.session_state['chat_history']", st.session_state["chat_history"])
            for msgs in st.session_state["chat_history"]:
                for msg_idx, msg in enumerate(msgs):
                    if msg_idx%2==0:
                        messages.chat_message("user").write(msg)
                        print("user: ", msg)
                    else:
                        messages.chat_message("assistant").write(msg)
                        print("assistant: ", msg)


        if question:
            messages.chat_message("user").write(question)
                        
            # apply RAG to given question
            answer = chain.invoke({
                "question": question, 
                "language": "English"
            })

            # write answer
            messages.chat_message("assistant").write(f"{answer}") 

            # store in chat history
            if not "chat_history" in st.session_state:
                st.session_state["chat_history"] = []
            st.session_state["chat_history"].append([question, answer])
            
    
    # TTS tab
    tab2.subheader("Let an AI read your News !")
    
    if st.session_state['cluster_choice'] is not None:
        tab2.write(f"You chose cluster n°{cluster_idx}: {cluster_choice}")
        tts_button = tab2.button("Generate Speech", type="primary")
        if 'tts_button' not in st.session_state:
            st.session_state['tts_button'] = tts_button
        
        if tts_button:

            # text to synthesize as speech
            text = df_analysis.loc[cluster_idx0, "title"] 
            text = text + "\n\n" + df_analysis.loc[cluster_idx0, "summary"]
        
            # build request for tts server
            params = deepcopy(TTS_PARAMS)
            params['text'] = text

            # sending get request and saving the response as response object
            r = requests.get(url=TTS_URL, params=params)

            # write response as .wav file
            with open("data/07_model_output/test.wav", 'wb') as f:
                f.write(r.content)
            
            # display audio
            x, sr = librosa.load("data/07_model_output/test.wav")
            tab2.audio(x, sample_rate=sr)

            # display text as well for better UX
            tab2.markdown(text + "\n" + df_analysis.loc[cluster_idx0, "summary"])






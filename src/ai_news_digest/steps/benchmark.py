import datetime
import json
import random
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
import yaml
from langchain.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from loguru import logger
from rich import print
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from transformers import AutoTokenizer, BertModel, BertTokenizer, BertTokenizerFast

from ai_news_digest.utils import check_gpu_availability, create_run_folder

# define types
LANGCHAIN_TYPE_NAME = Literal[
    "HuggingFaceInstructEmbeddings",
    "HuggingFaceBgeEmbeddings",
    "HuggingFaceEmbeddings",
]
LANGCHAIN_TYPE = Union[
    HuggingFaceInstructEmbeddings,
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
]

# define default argument values
PATH_INFO_DICT = "data/03_primary/arxiv_dict_2023-09-04_01-09-05.json"
# PATH_INFO_DICT = "data/03_primary/arxiv_dict_2023-09-04_01-21-10.json"
MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {
    "normalize_embeddings": True,
    "batch_size": 16,
    "output_value": "sentence_embedding",
    "convert_to_numpy": True,
    "show_progress_bar": True,
}
UMAP_KWARGS = {
    "n_neighbors": 4,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "cosine",
}


# define helpers
def compute_bert_embeddings(
    input_text: Union[str, List[str]],
    tokenizer: Union[BertTokenizerFast, AutoTokenizer, BertTokenizer],
    model: BertModel,
    tokenizer_kwargs: dict = {},
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute embeddings output by a BERT model (huggingface implementation).

    Parameters
    ----------
    input_text : Union[str, List[str]]
        Input text(s) to compute embeddings from
    tokenizer : Union[BertTokenizerFast, AutoTokenizer, BertTokenizer]
        Text tokenizer associated to provided BERT model
    model : BertModel
        BERT model (huggingface implementation)
    tokenizer_kwargs : dict, optional
        Keyword arguments passed to the model, by default {}
    batch_size : Optional[int], optional
        Number of input samples to feed to the model at once, by default None.
        If None, all inputs samples are passed to the model at once.

    Returns
    -------
    embeddings : torch.Tensor
        BERT-computed embeddings

    """
    # no batches
    if batch_size is None or type(input_text) is str:
        # tokenize inputs
        inputs = tokenizer(input_text, return_tensors="pt", **tokenizer_kwargs)

        # feed-forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # retrieve embeddings
        # DOC: https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertModel.forward # noqa
        embeddings = outputs.pooler_output

    # several batches
    else:
        # initialize results
        embeddings = torch.empty((0, model.config.hidden_size))

        # iterate over batches of inputs
        indices = list(range(0, len(input_text), batch_size)) + [None]
        for idx, i in enumerate(indices[:-1]):
            # tokenize batch of inputs
            input_text_i = input_text[indices[idx] : indices[idx + 1]]
            inputs_i = tokenizer(input_text_i, return_tensors="pt", **tokenizer_kwargs)

            # feed-forward pass
            with torch.no_grad():
                outputs_i = model(**inputs_i)

            # retrieve and store batch embeddings
            embeddings_i = outputs_i.pooler_output
            embeddings = torch.cat((embeddings, embeddings_i), dim=0)

        # clear memory
        del outputs_i
        del inputs_i
        del embeddings_i
        torch.cuda.empty_cache()

    return embeddings


def load_langchain_model(
    model_name: str,
    model_type: LANGCHAIN_TYPE_NAME,
    model_kwargs: Dict[str, Any] = MODEL_KWARGS,
    encode_kwargs: Dict[str, Any] = ENCODE_KWARGS,
    embed_instruction: str = "Represent the document for retrieval: ",
) -> LANGCHAIN_TYPE:
    """Load a language model via langchain's interface.

    Parameters
    ----------
    model_name : str
        Name of the model passed to langchain's loader (huggignface `repo_id`).
    model_type : LANGCHAIN_TYPE_NAME
        Name of the lancgchain loader to be used.
    model_kwargs : Dict[str, Any]
        Model keyword arguments passed to langchain's loader
    encode_kwargs : Dict[str, Any], optional
        Keyword arguments passed to the model `encode` method, by default ENCODE_KWARGS
    embed_instruction : _type_, optional
        Instruction prefix passed to `Instructor` models, ignored otherwise,
        by default "Represent the document for retrieval: "

    Returns
    -------
    model: LANGCHAIN_TYPE
        Model instance as return by langchain's wrappers

    Raises
    ------
    RuntimeError
        If provided model type is not supported

    """
    if model_type == "HuggingFaceBgeEmbeddings":
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    elif model_type == "HuggingFaceEmbeddings":
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    elif model_type == "HuggingFaceInstructEmbeddings":
        return HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            embed_instruction=embed_instruction,
        )

    else:
        raise RuntimeError(f"model_type '{model_type}' is not supported.")


def entropy(x: np.ndarray) -> float:
    """Compute the joint entropy in a multivariate (low_dim) tabular dataset.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, n_features)
        Input dataset to compute the joint entropy from

    Returns
    -------
    float
        Joint entropy of the dataset

    """
    counts = np.histogramdd(x)[0]
    freqs = counts / np.sum(counts)
    logs = np.log2(np.where(freqs > 0, freqs, 1))
    return -np.sum(freqs * logs)


def run_benchmark(
    path_to_info_dict: str = PATH_INFO_DICT,
    path_to_model_config: str = "conf/base/cluster_bench_models.yml",
    umap_kwargs: Dict[str, Any] = UMAP_KWARGS,
    n_clusters: int = 5,
    random_state: int = 123,
) -> None:
    """Run embedding and clustering benchmark.

    Parameters
    ----------
    path_to_info_dict : str, optional
        Path to yaml with parsed pdf info, by default PATH_INFO_DICT
    path_to_model_config : str, optional
        Path to list of model configs (`yml` file),
        by default "conf/base/cluster_bench_models.yml"
    umap_kwargs : Dict[str, Any], optional
        Keyword arguments passed to umap, by default UMAP_KWARGS
    n_clusters : int, optional
        Number of KMeans clusters, by default 5
    random_state : int, optional
        Random seed, by default 123

    Returns
    -------
    None

    """
    # compute paths
    info_path = Path(path_to_info_dict)

    # create subfolder for artifacts
    run_path = create_run_folder()
    logger.info(f"Successfully created run folder : '{run_path}'")

    # set random seeds
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)

    # check GPU availability
    device = check_gpu_availability()

    # load data (original & translated)
    with open(info_path, "r") as f:
        info_dict = json.load(f)["results"]
    logger.info(f"Successfully loaded prepared data from : {info_path}")

    # retrieve abstracts, titles, dates & paper IDs
    df_data = pd.DataFrame(info_dict).transpose()

    # retrieve list of models
    with open(path_to_model_config, "r") as f:
        models_config = yaml.safe_load(f)
    logger.info(f"Successfully loaded models config from '{path_to_model_config}'")

    # initialize structures to store results
    df_bench = pd.DataFrame(index=list(models_config.keys()))

    # iterate over models
    for i, (model_nickname, model_config) in enumerate(models_config.items()):
        logger.info(f"Running model '{model_nickname} ({i+1}/{len(models_config)})'")

        # load model
        hf = load_langchain_model(
            model_name=model_config["model_name"],
            model_type=model_config["type"],
            model_kwargs={"device": device},
            encode_kwargs=ENCODE_KWARGS,
            embed_instruction=model_config["embed_instruction"],
        )

        # compute embeddings
        embed_t0 = time()
        embeddings = hf.embed_documents(df_data["abstract"])
        embed_t1 = time()

        # convert to array
        embeddings = np.array(embeddings)
        print(f"embeddings.shape: {embeddings.shape}")

        # instanciate umap projector
        reducer = umap.UMAP(random_state=random_state, **umap_kwargs)

        # project data
        umap_proj = reducer.fit_transform(embeddings)

        # store in a dataframe with metadata
        df = pd.DataFrame(
            columns=[f"umap_{i}" for i in range(umap_proj.shape[1])], data=umap_proj
        )
        df = pd.concat((df, df_data.reset_index(names=["ID"])), axis=1)

        # clustering
        X_cluster = df[[col for col in df.columns if "umap" in col]]
        clustering = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            init="k-means++",
            n_init=1,
        )
        clustering.fit(X_cluster)
        df["cluster"] = [str(elt) for elt in clustering.labels_]

        # viz
        fig = px.scatter(
            df,
            x="umap_0",
            y="umap_1",
            hover_data=[
                "title",
                "ID",
            ],
            color="cluster",
            color_continuous_scale=px.colors.qualitative.D3,
        )
        fig.write_html(run_path / f"umap_clust_{model_nickname}.html")

        # compute and store scores
        scores = {}
        scores["Calinski-Harabasz ↑"] = calinski_harabasz_score(
            X_cluster, df["cluster"]
        )
        scores["Silhouette ↑"] = silhouette_score(X_cluster, df["cluster"])
        scores["Davies-Bouldin ↓"] = davies_bouldin_score(X_cluster, df["cluster"])
        scores["Entropy ↓"] = entropy(X_cluster.values)
        scores["embed_dim"] = embeddings.shape[1]
        scores["runtime"] = datetime.timedelta(seconds=embed_t1 - embed_t0)
        for score_name, score_val in scores.items():
            df_bench.loc[model_nickname, score_name] = score_val

        # save model specific artifacts
        df.to_csv(run_path / f"df_{model_nickname}.csv")

    # save global artifacts
    df_bench.to_csv(run_path / "df_bench.csv")

    return None


if __name__ == "__main__":
    logger.info("I'm in the main !!!")
    run_benchmark()

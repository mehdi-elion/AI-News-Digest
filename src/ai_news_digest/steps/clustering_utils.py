from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import umap
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from loguru import logger
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    silhouette_score,
)
from tqdm.contrib.itertools import product
from transformers import AutoTokenizer, BertModel, BertTokenizer, BertTokenizerFast

# define types
LANGCHAIN_TYPE_NAME = Literal[
    "HuggingFaceInstructEmbeddings",
    "HuggingFaceEmbeddings",
]
LANGCHAIN_TYPE = Union[
    HuggingFaceInstructEmbeddings,
    HuggingFaceEmbeddings,
]

# define default argument values
PATH_INFO_DICT = "data/03_primary/arxiv_dict_2023-11-06_00-22-42.json"
MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {
    "normalize_embeddings": True,
    "batch_size": 16,
    "output_value": "sentence_embedding",
    "convert_to_numpy": True,
    "show_progress_bar": True,
}
CLUSTERING_KWARGS = {
    "min_cluster_size": 10,
    "min_samples": 3,
    "max_cluster_size": None,
    "cluster_selection_epsilon": 0.05,
}
UMAP_KWARGS = {"n_neighbors": 4, "min_dist": 0.001, "n_components": 2, "metric": "cosine", "init": "random"}


# define helpers
def compute_bert_embeddings(
    input_text: Union[str, List[str]],
    tokenizer: Union[BertTokenizerFast, AutoTokenizer, BertTokenizer],
    model: BertModel,
    tokenizer_kwargs: dict[str, Any] | None = None,
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
    tokenizer_kwargs = tokenizer_kwargs or {}
    # no batches
    if batch_size is None or isinstance(input_text, str):
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

    return embeddings  # type: ignore


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
    if model_type in ["HuggingFaceBgeEmbeddings", "HuggingFaceEmbeddings"]:
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
        raise NotImplementedError(f"model_type '{model_type}' is not supported.")


def entropy(x: npt.NDArray[np.float_]) -> float:
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
    entropy: float = -np.sum(freqs * logs)
    return entropy


# TODO: add on option / argument to choose between HDBSCAB, KMeans, etc...
def clustering_pipeline(
    df_embed: pd.DataFrame,
    umap_kwargs: Dict[str, Any],
    clustering_kwargs: Optional[Dict[str, Any]] = None,
    random_state: int = 123,
    df_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Perform UMAP projection of embeddings followed by a clustering stage.

    Parameters
    ----------
    df_embed : pd.DataFrame
        Dataframe containing document embeddings; one row per document, one
        column per embedding coordinate (f'embed_{i}')
    umap_kwargs : dict
        UMAP parameters
    clustering_kwargs : Optional[dict], optional
        Cluestering parameters, by default None. If None, the clu
    random_state : int, optional
        Random seed used for the UMAP projection, by default 123
    df_data : Optional[pd.DataFrame], optional
        Dataframe containing document metadata, by default None. If not None,
        will be concatenated to `df_umap`, i.e. the output of the
        `clustering_pipeline()` function

    Returns
    -------
    df_umap : pd.DataFrame
        Dataframe containing projected coordinates of document embeddings.
    clustering : Any
        Scikit-learn clustering object
        (e.g. `sklearn.cluster._hdbscan.hdbscan.HDBSCAN`)

    """
    # --- dimensionality reduction ---

    # instanciate umap projector
    reducer = umap.UMAP(random_state=random_state, **umap_kwargs)

    # project data
    umap_proj = reducer.fit_transform(df_embed)

    # normalize umap coords
    umap_proj = (umap_proj - umap_proj.min(axis=0)) / (umap_proj.max(axis=0) - umap_proj.min(axis=0))

    # store in a dataframe
    df_umap = pd.DataFrame(columns=[f"umap_{i}" for i in range(umap_proj.shape[1])], data=umap_proj)

    # add metadata if available
    if df_data is not None:
        df_umap = pd.concat((df_umap, df_data.reset_index(names=["ID"])), axis=1)

    # --- clustering ---
    if clustering_kwargs is not None:
        X_cluster = df_umap[[col for col in df_umap.columns if "umap" in col]]
        clustering = HDBSCAN(**clustering_kwargs)
        clustering.fit(X_cluster)
        df_umap["cluster"] = [str(elt) for elt in clustering.labels_]
        df_umap["noise"] = [int(elt == -1) for elt in clustering.labels_]

    else:
        clustering = None

    # --- result ---
    return df_umap, clustering


def grid_search_umap_projection(
    umap_grid: Dict[str, List[Any]],
    df_embed: pd.DataFrame,
    random_state: int = 123,
    df_data: Optional[pd.DataFrame] = None,
) -> Tuple[float, Optional[Dict[str, Any]], pd.DataFrame]:
    """Run grid search over UMAP parameters to optimize entropy.

    Parameters
    ----------
    umap_grid : Dict[str, List[Any]]
        Grid of UMAP parameters.
    df_embed : pd.DataFrame
        Dataframe containing document embeddings; one row per document, one
        column per embedding coordinate (f'embed_{i}')
    random_state : int, optional
        Random seed used for the UMAP projection, by default 123
    df_data : Optional[pd.DataFrame], optional
        Dataframe containing document metadata, by default None. If not None,
        will be concatenated to `df_umap`, i.e. the output of the
        `clustering_pipeline()` function

    Returns
    -------
    best_entropy : float
        Best (lowest) entropy found across the grid of clustering parameters.
    best_umap_kwargs : Dict[str, Any]
        UMAP parameters corresponding to the best entropy
    df_umap_search : pd.DataFrame
        Dataframe containing all tested parameters combinations and associated
        scores

    """
    best_entropy = np.inf
    best_umap_kwargs = None
    df_umap_search = pd.DataFrame(columns=list(umap_grid.keys()) + ["Entropy ↓"])

    for umap_vals in product(*tuple(umap_grid.values())):
        # retrieve current umap kwargs
        umap_kwargs = {}
        for i, key in enumerate(umap_grid.keys()):
            umap_kwargs[key] = umap_vals[i]

        try:
            df_umap, clustering = clustering_pipeline(
                df_embed,
                umap_kwargs,
                clustering_kwargs=None,
                random_state=random_state,
                df_data=df_data,
            )
            X_cluster = df_umap[[c for c in df_umap.columns if "umap_" in c]]
            curr_entropy = entropy(X_cluster.values)
            df_umap_search = pd.concat(
                (
                    df_umap_search,
                    pd.DataFrame(data=np.array([umap_vals + (curr_entropy,)]), columns=df_umap_search.columns),
                ),
                axis=0,
                ignore_index=True,
            )

            if curr_entropy < best_entropy:
                best_entropy = curr_entropy
                best_umap_kwargs = umap_kwargs

        except Exception as e:
            logger.warning(f" {umap_kwargs} failed with following error: {e}; " f"will proceed to next iteration")

    logger.success(f"Best entropy: {best_entropy}")
    logger.success(f"Best umap_kwargs: {best_umap_kwargs}")

    return best_entropy, best_umap_kwargs, df_umap_search


def grid_search_clustering(
    clustering_grid: Dict[str, List[Any]],
    df_embed: pd.DataFrame,
    umap_kwargs: Dict[str, Any] = UMAP_KWARGS,
    random_state: int = 123,
    df_data: Optional[pd.DataFrame] = None,
) -> Tuple[float, Optional[Dict[str, Any]], pd.DataFrame]:
    """Run grid search over clustering parameters to optimize Silhouette Score.

    Parameters
    ----------
    clustering_grid : Dict[str, List[Any]]
        Grid of clustering parameters.
    df_embed : pd.DataFrame
        Dataframe containing document embeddings; one row per document, one
        column per embedding coordinate (f'embed_{i}')
    umap_kwargs : Dict[str, Any], optional
        UMAP parameters to use to project embeddings into a 1-dimensional space,
        by default UMAP_KWARGS
    random_state : int, optional
        Random seed used for the UMAP projection, by default 123
    df_data : Optional[pd.DataFrame], optional
        Dataframe containing document metadata, by default None. If not None,
        will be concatenated to `df_umap`, i.e. the output of the
        `clustering_pipeline()` function

    Returns
    -------
    best_silhouette : float
        Best (highest) silhouette score found across the grid of clustering
        parameters.
    best_clustering_kwargs : Dict[str, Any]
        Clustering parameters corresponding to the best silhouette score
    df_clustering_search : pd.DataFrame
        Dataframe containing all tested parameters combinations and associated
        scores

    """
    best_silhouette = -np.inf
    best_clustering_kwargs = None
    df_clustering_search = pd.DataFrame(columns=list(clustering_grid.keys()) + ["Silhouette ↑"])

    for clustering_vals in product(*tuple(clustering_grid.values())):
        # retrieve current clustering kwargs
        clustering_kwargs = {}
        for i, key in enumerate(clustering_grid.keys()):
            clustering_kwargs[key] = clustering_vals[i]

        try:
            df_umap, clustering = clustering_pipeline(
                df_embed,
                umap_kwargs,
                clustering_kwargs=clustering_kwargs,
                random_state=random_state,
                df_data=df_data,
            )
            X_cluster = df_umap[[c for c in df_umap.columns if "umap_" in c]]
            curr_silhouette = silhouette_score(X_cluster.values, df_umap["cluster"])
            df_clustering_search = pd.concat(
                (
                    df_clustering_search,
                    pd.DataFrame(
                        data=np.array([clustering_vals + (curr_silhouette,)]), columns=df_clustering_search.columns
                    ),
                ),
                axis=0,
                ignore_index=True,
            )

            if curr_silhouette > best_silhouette:
                best_silhouette = curr_silhouette
                best_clustering_kwargs = clustering_kwargs

        except Exception as e:
            logger.warning(f" {umap_kwargs} failed with following error: {e}; " f"will proceed to next iteration")

    logger.info(f"Best silhouette: {best_silhouette}")
    logger.info(f"Best clustering_kwargs: {best_clustering_kwargs}")

    return best_silhouette, best_clustering_kwargs, df_clustering_search


if __name__ == "__main__":
    logger.info("I'm in the main !!!")

import json

import lancedb
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import LanceDB
from loguru import logger
from rich import print
from tqdm import tqdm

# constants & config
DB_NAME = "sample-lancedb"
TABLE_NAME = "arxiv_table"
PATH_TO_ARXIV_DATA = "data/03_primary/arxiv_dict_2023-11-06_00-22-42.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # or "BAAI/bge-base-en"
MODEL_KWARGS = {
    # "device": "cpu",
    "device": 0,
}
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

# read arxiv data
with open(PATH_TO_ARXIV_DATA, "r") as f:
    info_dict = json.load(f)["results"]
df = pd.DataFrame(info_dict).transpose()
logger.success(f"Read data from {PATH_TO_ARXIV_DATA}: ")
print(df.head())

# init database
uri = f"data/04_feature/{DB_NAME}"
db = lancedb.connect(uri)
logger.success(f"Created vector database with uri: '{uri}'")

# load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=MODEL_KWARGS,
    encode_kwargs=ENCODE_KWARGS,
)
logger.success(f"Loaded embedding model from huggingface: '{EMBEDDING_MODEL_NAME}'")

# create a table
# this doc page shows an example with doc chunking (in case we need it):
# --> https://python.langchain.com/docs/modules/data_connection/vectorstores/
logger.info("Creating LanceDB table...")
table_data = []
for key, value in tqdm(info_dict.items()):
    table_data += [
        {
            "id": key,
            "url": key,
            "title": value["title"],
            "text": value["abstract"],
            "date": value["date"],
            "category": value["category"],
            "vector": embedding_model.embed_query(value["abstract"]),
        }
    ]
table = db.create_table(
    TABLE_NAME,
    data=table_data,
    mode="overwrite",
)
logger.success(f"LanceDB table '{TABLE_NAME}' created !")

# create langchain vector store from LanceDB table
vectorstore = LanceDB(table, embedding_model)
logger.success(f"Created Langchain vectorstore from LanceDB table '{TABLE_NAME}'!")

# query database with similarity search
query = "Find papers adressing model interpretability or explainability"
docs = vectorstore.similarity_search(query)
print(docs[0].dict())

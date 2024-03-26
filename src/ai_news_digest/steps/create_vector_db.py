import json  # noqa: D100

import lancedb
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB
from loguru import logger
from rich import print
from tqdm import tqdm

# constants & config
CHUNKING = True
DB_NAME = "sample-lancedb"
TABLE_NAME = "arxiv_table"
PATH_TO_ARXIV_DATA = "data/03_primary/arxiv_dict_2023-11-06_00-22-42.json"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"  # or "sentence-transformers/all-mpnet-base-v2"
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

# instanciate text splitter
# TODO: explore usage of token counts with `RecursiveCharacterTextSplitter.from_huggingface_tokenizer`
if CHUNKING:
    logger.info("Will split documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
else:
    logger.info("Will embed documents without splitting...")


# create a table
# this doc page shows an example with doc chunking (in case we need it):
# --> https://python.langchain.com/docs/modules/data_connection/vectorstores/
logger.info("Creating LanceDB table...")
table_data = []
for key, value in tqdm(info_dict.items()):
    if not CHUNKING:
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
    else:
        # TODO: avoid iteratiev approach, use better option:
        #  -> `text_splitter.split_documents(docs)`
        #  -> from json directly ? (implies refacto of news json)
        splits = text_splitter.split_text(value["abstract"])
        for idx, split in enumerate(splits):
            table_data += [
                {
                    "id": f"{key}_split-{idx}",
                    "url": key,
                    "title": value["title"],
                    "text": split,
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
for i in range(4):
    print(f"doc title n°{i}: ")
    print(f"\t---> id: {docs[i].dict()['metadata']['id']}")
    print(f"\t---> title: {docs[i].dict()['metadata']['title']}")
    print(f"\t---> distance: {docs[i].dict()['metadata']['_distance']}")
# print("\n", docs[0].dict())

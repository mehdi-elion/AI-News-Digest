from typing import List, Optional, Union, Any, Tuple, Union
import sys
from loguru import logger
from pathlib import Path 

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
import yaml, json
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline, VLLMOpenAI

from rich import print
from sklearn.cluster import KMeans, HDBSCAN, DBSCAN

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)



def summarize_recurse(
    llm: Union[HuggingFacePipeline, VLLMOpenAI],
    prompt_template: str,
    docs: List[Document],
) -> str:
    
    # define Stuff chain
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    # TODO: sort documents based on their distance to O (origin)


    # get max sequence length OR force a specific value
    # TODO: check this value is valid and functional 
    # --> https://github.com/huggingface/transformers/blob/f1cc6157210abd9ea953da097fb704a1dd778638/src/transformers/models/mistral/configuration_mistral.py#L63 
    Lt_max = 6000

    # compute number of promt tokens
    Lt_prompt = llm.get_num_tokens(prompt_template)

    # compute nmber of tokens for docs
    Lt_docs_list = [llm.get_num_tokens(doc.page_content) for doc in docs]
    Lt_docs = np.sum(Lt_docs_list)
    
    # end case
    # TODO: check if this context_length is valid
    # if Lt_docs + Lt_prompt <= Lt_max:
    if stuff_chain.prompt_length(docs=docs) <= Lt_max:
        return stuff_chain.run(docs)
    
    else:

        news_docs = []
        j1, j2 = 0, 1

        while j1 < len(docs) and j2 < len(docs):

            if stuff_chain.prompt_length(docs=docs[j1:j2+1]) > Lt_max:
                news_docs.append(Document(page_content=stuff_chain.run(docs[j1:j2])))
                j1 = j2
                j2 = j2 + 1
            else:
                j2 = j2 + 1
        
        if j2 >= len(docs):
            news_docs.append(Document(page_content=stuff_chain.run(docs[j1:])))

        return summarize_recurse(llm, prompt_template, news_docs)
    
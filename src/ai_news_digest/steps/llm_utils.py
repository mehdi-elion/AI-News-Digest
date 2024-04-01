from typing import List, Union

from langchain import LLMChain, PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.document import Document
from langchain_community.llms import HuggingFacePipeline, VLLMOpenAI
from langchain_core.prompts import format_document


# summarization prompt template
SUMMARY_PROMPT_TEMPLATE = """<s>[INST]You will be provided with a list of news articles that you must summarize.
Write a concise summary focusing on the key pieces of information mentioned in the articles.
Make sure the summary covers all the issues tackled in the group of articles. Use less than 200 words.
Here is the list of articles:
"{text}"
CONCISE SUMMARY (less than 180 words): [/INST]"""

# keyword-generation prompt template
KEYWORD_PROMPT_TEMPLATE = """<s>[INST]You will be provided with a summary made from a list of news articles.
Extract a list of keywords that best covers all pieces of information tackled in this articles.
Don't use brackets or any special characters. Use only commas to separate keywords.
Here is the summary of articles:
"{}"
CONCISE COMMA-SEPARATED LIST OF KEYWORDS: [/INST]"""

# title-generation prompt template
TITLE_PROMPT_TEMPLATE = """<s>[INST]You will be provided with a summary made from a list of news articles.
Write a short title that best covers all pieces of information tackled in this articles.
Don't use brackets or any special characters. Don't provide any explanation or alternative title. 
Write only one unique title in less than 15 words.
Here is the summary of articles:
"{}"
UNIQUE TITLE (< 15 words): [/INST]"""

# RAG prompt template
RAG_TEMPLATE = """<s>[INST]Concisely answer the question (in less than 180 words) based only on the following context:
{context}

Question: {question}

Answer: [/INST]
"""

# prompt template to combine context for RAG
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def summarize_recurse(
    llm: Union[HuggingFacePipeline, VLLMOpenAI],
    prompt_template: str,
    docs: List[Document],
) -> str:
    """Generate a summary by appling a recursive stuffing strategy.

    Parameters
    ----------
    llm : Union[HuggingFacePipeline, VLLMOpenAI]
        LLM wraped using one of langchain's APIs
    prompt_template : str
        Prompt template use for summarization (in which docs to summarize
        will be injected)
    docs : List[Document]
        List of lanchchain documents to summarize

    Returns
    -------
    summary : str
        Summary obtained after applying

    """
    # define Stuff chain
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    # TODO: sort documents based on their distance to O (origin)

    # get max sequence length OR force a specific value
    # TODO: check this value is valid and functional
    # --> https://github.com/huggingface/transformers/blob/f1cc6157210abd9ea953da097fb704a1dd778638/src/transformers/models/mistral/configuration_mistral.py#L63
    Lt_max = 6000

    # end case
    # TODO: check if this context_length is valid
    # if Lt_docs + Lt_prompt <= Lt_max:
    if stuff_chain.prompt_length(docs=docs) <= Lt_max:
        return stuff_chain.run(docs)

    else:
        news_docs = []
        j1, j2 = 0, 1

        while j1 < len(docs) and j2 < len(docs):
            if stuff_chain.prompt_length(docs=docs[j1 : j2 + 1]) > Lt_max:  # noqa
                news_docs.append(Document(page_content=stuff_chain.run(docs[j1:j2])))
                j1 = j2
                j2 = j2 + 1
            else:
                j2 = j2 + 1

        if j2 >= len(docs):
            news_docs.append(Document(page_content=stuff_chain.run(docs[j1:])))

        return summarize_recurse(llm, prompt_template, news_docs)

def _combine_documents(
    docs: Document, 
    document_prompt: str=DEFAULT_DOCUMENT_PROMPT, 
    document_separator: str="\n\n",
) -> str:
    """Combine contents from various documents in one single string for RAG.

    Parameters
    ----------
    docs : langchain.schema.document.Document
        Documents
    document_prompt : str, optional
        Prompt template in which input documents will be combined, 
        by default DEFAULT_DOCUMENT_PROMPT
    document_separator : str, optional
        String used to seperate documents from one another within the prompt, 
        by default "\n\n"

    Returns
    -------
    prompt : str
        Prompt resulting from the insertion of document contents within the 
        specified prompt template.

    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
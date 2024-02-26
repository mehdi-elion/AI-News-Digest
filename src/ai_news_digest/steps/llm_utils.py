from typing import List, Union

from langchain import LLMChain, PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.document import Document
from langchain_community.llms import HuggingFacePipeline, VLLMOpenAI


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

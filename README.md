# AI News Digest

## Overview

This is your new Data project, which was generated using cookiecutter.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

or

```console
conda env create -f src/environment.yml --force
conda activate ai_news
```

## How to run your project


## How to test your project


## Project dependencies


## TODO List
Quick and non-exhaustive list of tasks:
- [ ] iterate over newsapi sources & domains to curb the 100-article limit (free-tier)
- [ ] enable gnews and/or newsapi backends for news corpus retrieval
- [ ] debug text-embedding-inference API usage
- [ ] use text-generation-inference API with small models
- [ ] plug RAG pipeline with :
    - [ ] text-gen/embed-inference APIs
    - [ ] small Models
    - [ ] clustering & viz pipelines
    - [ ] summarization pipeline
- [ ] add cunking feature to RAG pipeline
- [ ] improve clusetring stage with:
    - [ ] HDBscan
    - [ ] normalization of preojected coordinates (umap) for easier setup of clustering thresholds
    - [ ] try Deep TDA (topological Data Analysis) as a replacement for UMAP
        - [medium article](https://medium.com/@juanc.olamendy/deep-tda-a-new-dimensionality-reduction-algorithm-2d04fa6ed2eb)
        - [giotto-tda](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html)
        - [giotto-deep](https://github.com/giotto-ai/giotto-deep)
        - [linkedin article](https://www.linkedin.com/pulse/deep-tda-new-dimensionality-reduction-algorithm-olamendy-turruellas/)
        - [scikit-tda](https://scikit-tda.org/)
    - [ ] use the elbow method

# Unified embeddings
Serve different embedding models using a unified http api


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
Because there are many implementation details in embedding models it is hard to try different types of models in your application. 
The sentence Transformers library does a good job at providing an interface for many tasks, but it has so many options that it too has become confusing.

This library abstracts the implementation details away and only provides endpoints for the available options with the model.



## Features
- Web server for various embeddings
- interface for 
  - symmetric embeddings (document similarity)
  - assymetric embeddings (retrieval, QA)
  - classification embeddings (to be used in downstream classification tasks)
  - clustering embeddings
  - reranking
- A default interface to load models for only symmetric embeddings
 

## Installation
First make sure you have [UV](https://docs.astral.sh/uv/) available in your enviuronment

```bash
# Clone the repository
git clone https://github.com/vriesdemichael/unified-embeddings.git

# Navigate to the project directory
cd unified-embeddings

# Install dependencies
uv sync
```

## Usage
You can start the server using 
`uv run python -m unified_embeddings` or `uv run unified-embeddings`

or you can use 
`uvicorn unified_embeddings.server:app_factory --factory --port 8000 --host "0.0.0.0"


If you want to use your other models than the ones provided here please add them in the [models dir](./src/unified_embeddings/models)

Implement the functions from [ServeableEmbedder](./src/unified_embeddings/servable_embedder.py) to set up how your embedding model deals with specific types of embeddings.

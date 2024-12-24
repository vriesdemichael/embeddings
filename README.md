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
Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/yourusername/yourproject.git

# Navigate to the project directory
cd yourproject

# Install dependencies
pip install -r requirements.txt
```

## Usage
Examples and instructions on how to use the project.

```python
import yourmodule

# Example usage
yourmodule.yourfunction()
```

## Contributing
Guidelines for contributing to the project.

## License
Information about the project's license.

## Acknowledgements
Credits and acknowledgements for those who contributed to the project.
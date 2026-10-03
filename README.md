# Building-a-Chatbot-with-RAG-and-Graph-Based-Reasoning
# KIT719 Project 2

## Overview

This project implements a local Graph-Enhanced Retrieval-Augmented Generation (RAG) chatbot. The system combines the BBC News dataset with 10 locally stored text documents, applies NLTK preprocessing, creates BERT embeddings for semantic retrieval, builds an RDF knowledge graph, queries the graph with SPARQL, trains a TransE graph-embedding model, and uses Qwen2.5-1.5B-Instruct to choose a retrieval route and generate grounded answers.

## Main Components

- BBC News dataset (2,410 records)
- 10 local documents
- NLTK preprocessing: sentence splitting, tokenisation, stop-word removal, lemmatisation, POS tagging and NER
- Three-sentence document chunking
- bert embeddings
- Cosine-similarity document retrieval
- RDF knowledge graph using RDFLib
- SPARQL graph search
- TransE graph embeddings
- Qwen2.5-1.5B-Instruct for retrieval routing and answer generation
- Gradio web interface for the final chatbot
- Baseline and failure-case evaluation


## Required Files and Folder Structure

- BBCNews.csv

- document/
    d-1.txt
    d-2.txt
    d-3.txt
    d-4.txt
    d-5.txt
    d-6.txt
    d-7.txt
    d-8.txt
    d-9.txt
    d-10.txt


## Installation

Run the installation cell at the beginning of the notebook:


!pip -q install rank-bm25 rdflib transformers gradio networkx matplotlib


For a fresh local Python environment, the following broader installation is recommended because the notebook also imports these packages:

! pip install pandas numpy nltk torch scikit-learn tqdm rank-bm25 rdflib transformers accelerate gradio networkx matplotlib

## How to Run the Application

1. Open  project in Google Colab or Jupyter Notebook.
2. Enable a GPU if available.
3. Place `BBCNews.csv` in the BBC dataset folder.
4. Place the ten local text files `d-1.txt` to `d-10.txt` in the `/document` folder.
5. Run the package-installation cell.
6. Run the import and configuration cells.
7. Run the NLTK download cell.
8. Run the data-loading cells and confirm that the BBC and custom documents load successfully.
9. Run the preprocessing and document-chunking cells.
10. Run the BERT embedding cell. This is one of the slowest preprocessing stages and should normally be completed once before asking questions.
11. Run the RDF knowledge-graph and SPARQL cells.
12. Run the TransE training cell.
13. Run the Qwen model-loading cell.
14. Run the retrieval-routing, context-fusion and `hybrid_rag()` cells.
15. Run the evaluation cells if evaluation results are required.
16. Run the final Gradio interface cell and open the URL displayed by Gradio.


## Gradio Output

The final Gradio interface should display:

- Question input
- Generated answer
- Retrieval method
- Document citations / sources
- Graph / SPARQL evidence


## Troubleshooting

**BBC dataset not found**

Check that dataset `BBCNews.csv` 

**NLTK resource error**

Rerun the NLTK download cell before running preprocessing or NER.

# RAG System with LangChain + HuggingFace + Pinecone

> **Lab Note:** This lab originally specifies using OpenAI for embeddings and LLM. Since OpenAI's API requires paid credits, this implementation uses free, open-source alternatives (HuggingFace models) that run locally. The RAG architecture, LangChain pipeline, and Pinecone integration are identical to what you would build with OpenAI.

---

## Table of Contents

- [What is a RAG?](#what-is-a-rag)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Components Explained](#components-explained)
- [Screenshots](#screenshots)
- [Model Choices](#model-choices)

---

## What is a RAG?

A **Retrieval-Augmented Generation (RAG)** system is an AI architecture that enhances a language model's responses by first retrieving relevant information from a knowledge base, then using that information as context to generate a grounded, accurate answer.

Instead of relying solely on the model's training data (which can be outdated or hallucinated), RAG grounds every answer in real documents you provide.

---

## Architecture

```
User Question
      │
      ▼
[Embedding Model]
Converts the question into a
numerical vector (384 dimensions)
      │
      ▼
[Pinecone Vector Database]
Searches for the most similar
document chunks using cosine similarity
      │
      ▼
[Retrieved Context]
The top 3 most relevant chunks
are returned
      │
      ▼
[Prompt Template]
Context + Question are combined
into a structured prompt
      │
      ▼
[Language Model (LLM)]
Reads the context and generates
a grounded answer
      │
      ▼
Final Answer
```

<!-- 
  📸 IMAGE PLACEHOLDER
  Name: architecture_diagram.png
  How to get it: Take a screenshot of the diagram above rendered as an image,
  or draw it manually and export as PNG.
  Replace this comment with: ![RAG Architecture](screenshots/architecture_diagram.png)
-->

---

## Project Structure

```
rag-project/
│
├── rag_free_v2.ipynb        # Main notebook with full RAG implementation
├── .env                     # API keys (never commit this to GitHub)
├── .gitignore               # Ensures .env is not pushed to GitHub
├── requirements.txt         # Python dependencies
├── data/
│   └── sample.txt           # Knowledge base document
└── screenshots/             # Folder for README screenshots
    ├── step_install.png
    ├── step_pinecone_key.png
    ├── step_index_created.png
    ├── step_embeddings.png
    ├── step_rag_answer.png
    └── architecture_diagram.png
```

---

## Prerequisites

Before running this project you need:

- Python 3.9 or higher installed
- A free [Pinecone account](https://app.pinecone.io) (no credit card required)
- Jupyter Notebook or JupyterLab installed

You do **not** need an OpenAI account or any paid service.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies

Run this command in your terminal, or execute the first cell of the notebook:

```bash
pip install langchain langchain-pinecone langchain-huggingface langchain-community pinecone-client sentence-transformers python-dotenv transformers torch
```

Or if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

<!-- 
  📸 IMAGE PLACEHOLDER
  Name: step_install.png
  How to get it: Run the pip install command in your terminal or notebook,
  take a screenshot showing the packages being installed successfully.
  Replace this comment with: ![Installing dependencies](screenshots/step_install.png)
-->

---

## Configuration

### 1. Get your Pinecone API Key

1. Go to [app.pinecone.io](https://app.pinecone.io) and create a free account
2. In the left menu, click **API Keys**
3. Copy the key shown

<!-- 
  📸 IMAGE PLACEHOLDER
  Name: step_pinecone_key.png
  How to get it: Take a screenshot of the Pinecone dashboard showing the API Keys section.
  Blur or hide the actual key value for security.
  Replace this comment with: ![Pinecone API Key](screenshots/step_pinecone_key.png)
-->

### 2. Create the `.env` file

Create a file named `.env` in the root of the project with the following content:

```
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=rag-demo
```

>  **Never push your `.env` file to GitHub.** Make sure your `.gitignore` contains `.env`.

### 3. Create a `.gitignore` file

Create a file named `.gitignore` with at minimum:

```
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## How to Run

### 1. Open the notebook

```bash
jupyter notebook rag_free_v2.ipynb
```

### 2. Run cells in order

Execute each cell from top to bottom. The notebook is divided into clear steps:

| Step | What it does |
|------|-------------|
| Step 1 | Installs all dependencies |
| Step 2 | Explains `.env` configuration |
| Step 3 | Imports libraries and loads API keys |
| Step 4 | Creates the sample knowledge base document |
| Step 5 | Splits the document into chunks |
| Step 6 | Loads the embedding model (HuggingFace, free) |
| Step 7 | Creates the Pinecone index and stores embeddings |
| Step 8 | Configures the retriever |
| Step 9 | Loads the LLM (flan-t5-base, free) |
| Step 10 | Builds the full RAG pipeline |
| Step 11 | Runs sample queries |
| Step 12 | Inspects retrieved documents |

<!-- 
   IMAGE PLACEHOLDER
  Name: step_index_created.png
  How to get it: Take a screenshot of the notebook output after Step 7 runs,
  showing "Index created successfully" and "Documents successfully indexed in Pinecone".
  Replace this comment with: ![Pinecone Index Created](screenshots/step_index_created.png)
-->

<!-- 
   IMAGE PLACEHOLDER
  Name: step_embeddings.png
  How to get it: Take a screenshot of the notebook output after Step 6 runs,
  showing "Embedding model loaded successfully (384 dimensions, free, runs locally)".
  Replace this comment with: ![Embeddings Loaded](screenshots/step_embeddings.png)
-->

<!-- 
   IMAGE PLACEHOLDER
  Name: step_rag_answer.png
  How to get it: Take a screenshot of the notebook output after Step 11 runs,
  showing the questions and their generated answers.
  Replace this comment with: ![RAG Answers](screenshots/step_rag_answer.png)
-->

---

## Components Explained

### Embedding Model — `sentence-transformers/all-MiniLM-L6-v2`

Converts text into a 384-dimensional numerical vector. Similar texts produce vectors that are geometrically close to each other. This enables semantic search — finding documents that are *meaningfully* similar, not just keyword matches.

**Why this model?** It is free, open-source, and runs entirely on your local machine with no API calls required.

**OpenAI equivalent (paid):** `text-embedding-ada-002` — produces 1536-dimension vectors and requires a paid API key.

### Vector Database — Pinecone

Stores all the document chunk embeddings and allows fast similarity search. When a user asks a question, Pinecone finds the chunks whose vectors are closest to the question's vector.

**Free tier:** Pinecone's free tier supports 1 index with up to 100,000 vectors — more than enough for this lab.

### Language Model — `google/flan-t5-base`

Reads the retrieved context and the user's question, then generates a coherent answer. Flan-T5 is a model from Google fine-tuned specifically for instruction following and question answering.

**Why flan-t5 and not gpt2?** Despite having "GPT" in its name, GPT-2 is a 2019 model not designed for Q&A — it tends to repeat text or generate incoherent answers. Flan-T5 is specifically trained to follow instructions and answer questions based on a given context.

**OpenAI equivalent (paid):** `gpt-3.5-turbo` via `ChatOpenAI` — gives significantly better responses but requires a paid API key (~$5 minimum).

### LangChain

The framework that connects all components together. It provides:
- `TextLoader` — loads documents from files
- `RecursiveCharacterTextSplitter` — splits documents into chunks
- `PineconeVectorStore` — interface between LangChain and Pinecone
- `create_retrieval_chain` — builds the full RAG pipeline
- `ChatPromptTemplate` — structures the prompt sent to the LLM

---

## Screenshots

> Replace each placeholder below with actual screenshots after running the notebook.

### Pinecone API Keys page

![images/Pinecone API Keys page.png](<images/Pinecone API Keys page.png>)

### Dependency installation

![alt text](<images/Dependency installation.png>)


### Index creation and document indexing

![alt text](<images/Index creation and document indexing.png>)

### Embedding model loaded

![images/Embedding model loaded.png](<images/Embedding model loaded.png>)

### RAG system answering questions

![images/RAG system answering questions.png](<images/RAG system answering questions.png>)
---

## Model Choices

| Component | This Project (Free) | Lab Specification (Paid) |
|-----------|-------------------|--------------------------|
| Embeddings | `all-MiniLM-L6-v2` — 384 dims, local | `text-embedding-ada-002` — 1536 dims, OpenAI API |
| LLM | `google/flan-t5-base` — local | `gpt-3.5-turbo` — OpenAI API |
| Vector DB | Pinecone (free tier) | Pinecone (free tier) |
| Framework | LangChain | LangChain |

The RAG pipeline logic, LangChain chains, Pinecone integration, and overall architecture are **identical** in both versions. Only the model provider differs.

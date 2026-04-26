# 🚀 Build Your First RAG Pipeline: From Concept to Code

Welcome to the **Retrieval-Augmented Generation (RAG) Tutorial Repository**.

This project was originally presented at **IEEE Pharos University** and provides a **hands-on guide to building a functional RAG pipeline** using modern AI orchestration tools.

The repository walks through the **full lifecycle of a RAG system**, from document ingestion to generating grounded answers with an LLM.

---

## 🧠 What is RAG?

Traditional **Large Language Models (LLMs)** have two major limitations:

* **Knowledge Cutoff** – They only know information from their training data.
* **Hallucinations** – When unsure, they may generate incorrect answers confidently.

### RAG solves this problem by giving the LLM an **external knowledge source**.

Think of it like an **open-book exam**.

### The RAG workflow consists of three main stages:

### 1️⃣ Ingestion

Documents are split into **semantic chunks** and converted into **vector embeddings**, then stored in a vector database.

### 2️⃣ Retrieval

When a user asks a question, the system **searches the vector database** to find the most relevant pieces of context.

### 3️⃣ Generation

The retrieved context is injected into the **LLM prompt**, allowing the model to generate **accurate, grounded answers**.

---

## 🛠️ Tech Stack

| Component        | Technology                       |
| ---------------- | -------------------------------- |
| Orchestration    | LangChain                        |
| Vector Database  | ChromaDB                         |
| Embeddings Model | HuggingFace (`all-MiniLM-L6-v2`) |
| Inference Engine | Groq (Llama-3-8b)                |
| Environment      | Google Colab                     |

---

## 🚀 Getting Started

### Prerequisites

To run the tutorial notebook you will need:

* A **Groq API Key**

You can create one for free at:

```
https://console.groq.com
```

---

### Installation

Install the required Python libraries:

```bash
pip install langchain langchain-community langchain-huggingface langchain-groq chromadb pypdf sentence-transformers
```

---

## ⚙️ Pipeline Steps

The notebook walks through the following pipeline:

### 📄 1. Load Data

Documents are ingested using **PyPDFLoader**.

Default example:

> "Attention Is All You Need" research paper

---

### ✂️ 2. Chunking

The text is split using:

`RecursiveCharacterTextSplitter`

Configuration:

* **Chunk Size:** 1000 characters
* **Chunk Overlap:** 100 characters

This helps preserve semantic context across chunks.

---

### 🔢 3. Vectorization

Each chunk is converted into **high-dimensional embeddings** using:

`sentence-transformers`

These embeddings allow **semantic similarity search**.

---

### 🔎 4. Retrieval

A **vector similarity search** retrieves the **Top-3 most relevant chunks** related to the user query.

---

### 🧩 5. Chain Assembly

The retrieved context is injected into a **custom system prompt**, which guides the **Llama-3 model** to generate a grounded response.

---

## 🏗️ Beyond the Basics: Production RAG

Building a tutorial RAG system is easy.

Building a **production-ready RAG system** requires additional engineering techniques.

This repository briefly introduces:

### 🔍 Hybrid Search

Combining:

* Semantic Vector Search
* Keyword-based **BM25 Search**

---

### 📊 Reranking

Using **Cross-Encoders** (such as Cohere models) to reorder retrieved results for maximum relevance.

---

### 📈 Evaluation

Instead of manually checking answers, modern RAG systems use **evaluation frameworks** such as:

**RAGAS**

to measure:

* Faithfulness
* Answer relevance
* Context precision

---

## 📂 Repository Contents

```
.
├── RAG_Pipeline_Tutorial.ipynb   # Complete implementation of the RAG pipeline
├── RAG Session.pptx              # Theory and architecture presentation
├── cloudnest-support/            # End-to-end RAG + agent demo app (see below)
└── README.md                     # Project documentation
```

---

## 🧪 Bonus Project: `cloudnest-support/`

A full **end-to-end customer-support agent** that builds on the concepts from the
notebook and takes them to a real, deployable application.

It combines:

* **RAG** over a markdown knowledge base (Chroma + `all-MiniLM-L6-v2`)
* A **LangGraph ReAct agent** with two tools (`search_kb`, `create_ticket`)
* A **Streamlit** chat UI, deployable to Streamlit Community Cloud in minutes

```
User → Streamlit chat UI
         │
         ▼
   LangGraph ReAct Agent  (Llama-3.1-8b on Groq)
         │           │
         ▼           ▼
   search_kb    create_ticket
         │
         ▼
   Chroma + MiniLM embeddings
```

See [`cloudnest-support/README.md`](./cloudnest-support/README.md) for setup,
deployment instructions, and a full workshop guide under
[`cloudnest-support/docs/`](./cloudnest-support/docs).

---

## 👤 Author

**Youssef Bastawisy**


Code and slides were originally shared for educational purposes during a session at **IEEE Pharos University**.

---

⭐ If you found this tutorial helpful, consider **starring the repository**.

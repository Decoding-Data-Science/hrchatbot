# HR Chatbot

This repository contains a retrieval‑augmented generation (RAG) chatbot built with
[Pinecone](https://www.pinecone.io/) and [OpenAI](https://openai.com/).
The content of the HR handbook (under folders like `benefits/`, `guides/` etc.)
is indexed into a Pinecone vector database. Users can ask questions via a
[Gradio](https://gradio.app/) interface and receive answers based on the indexed
material.

## Setup

1. Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root and set the following variables:

```bash
OPENAI_API_KEY=YOUR_OPENAI_KEY
PINECONE_API_KEY=YOUR_PINECONE_KEY
PINECONE_ENV=YOUR_PINECONE_ENVIRONMENT
# optional: PINECONE_INDEX=hr-handbook
```

3. Run the application:

```bash
python app.py
```

The first run will index all `.txt` files into Pinecone. You can skip this step
on subsequent runs by setting the environment variable `SKIP_INDEXING=1`.

## Usage

The Gradio interface now displays category buttons and a list of recommended questions.
Pick a category, choose a suggested question or type your own, and the model retrieves
relevant documents to answer using OpenAI's Chat API. The layout uses two columns with the title centered.

## Folder Structure

```
benefits/               HR benefits information
communities-of-practice/ Community related docs
company/                Company overview
guides/                 Guides and policies
roles/                  Role descriptions
```

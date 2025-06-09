import os
import glob
from pathlib import Path
from typing import List

import openai
import pinecone
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "hr-handbook")

openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
def init_pinecone(index_name: str):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536, metric="cosine")
    return pinecone.Index(index_name)

# Load documents from folders
def load_documents(root_dir: str) -> List[dict]:
    docs = []
    for path in Path(root_dir).rglob("*.txt"):
        category = path.parts[1] if len(path.parts) > 1 else "general"
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            docs.append({"id": str(path), "text": content, "category": category})
    return docs

# Build embeddings and upsert into Pinecone
def index_documents(index: pinecone.Index, docs: List[dict]):
    for batch_start in range(0, len(docs), 100):
        batch = docs[batch_start:batch_start + 100]
        ids = [doc["id"] for doc in batch]
        texts = [doc["text"] for doc in batch]
        embeds = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        vectors = [(id_, embed, {"category": doc["category"]})
                   for id_, embed, doc in zip(ids, embeds["data"], batch)]
        index.upsert(vectors)

# Retrieve docs from Pinecone
def retrieve(query: str, index: pinecone.Index, category: str = None, k: int = 5) -> List[str]:
    embed = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    kwargs = {"top_k": k, "include_metadata": True}
    if category:
        kwargs["filter"] = {"category": {"$eq": category}}
    res = index.query(vector=embed, **kwargs)
    return [m["metadata"]["text"] for m in res["matches"]]

# Generate answer with OpenAI ChatCompletion
def generate_answer(query: str, docs: List[str]) -> str:
    system_prompt = (
        "You are a helpful HR assistant. Use the provided context to answer the question.\n"
        "If the answer is not contained in the context, reply that you don't know."
    )
    context = "\n\n".join(docs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content.strip()

# Gradio interface
def answer_question(query: str, category: str):
    docs = retrieve(query, pinecone_index, category)
    return generate_answer(query, docs)

if __name__ == "__main__":
    pinecone_index = init_pinecone(INDEX_NAME)
    if not int(os.getenv("SKIP_INDEXING", "0")):
        documents = load_documents(".")
        index_documents(pinecone_index, documents)

    categories = sorted({Path(p).parts[0] for p in glob.glob('*/*.txt')})
    # simple recommended questions per category
    recommended = {
        "all": [
            "Where can I find benefits information?",
            "How do I request time off?"
        ],
        "benefits": [
            "What does private medical insurance cover?",
            "How do I join the pension scheme?"
        ],
        "company": [
            "What is the company's mission?",
            "Where is the office located?"
        ],
        "guides": [
            "How do I submit an expense?",
            "Where is the hiring policy?"
        ],
        "roles": [
            "What does a data scientist do?",
            "How do career levels work?"
        ],
        "communities-of-practice": [
            "How can I join a community of practice?",
            "When do CoPs meet?"
        ],
    }

    with gr.Blocks() as demo:
        gr.Markdown("""<h1 style='text-align: center;'>HR Chatbot</h1>""")

        selected_category = gr.State("all")

        with gr.Row():
            category_buttons = []
            for cat in ["all"] + categories:
                btn = gr.Button(cat)
                category_buttons.append(btn)

        with gr.Row():
            with gr.Column():
                examples = gr.Radio(
                    choices=recommended["all"],
                    label="Recommended questions",
                    value=recommended["all"][0]
                )
            with gr.Column():
                query = gr.Textbox(label="Ask a question")
                submit = gr.Button("Submit")
                answer = gr.Textbox(label="Answer")

        def set_category(cat):
            return cat, gr.update(choices=recommended.get(cat, recommended["all"]), value=recommended.get(cat, recommended["all"])[0])

        for btn in category_buttons:
            btn.click(lambda _, cat=btn.value: set_category(cat), inputs=None, outputs=[selected_category, examples])

        def fill_example(q):
            return q

        examples.change(fill_example, inputs=examples, outputs=query)

        def _submit(q, cat):
            c = None if cat == "all" else cat
            return answer_question(q, c)

        submit.click(_submit, inputs=[query, selected_category], outputs=answer)

    demo.launch()

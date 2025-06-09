import os
import glob
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from pinecone import Pinecone, ServerlessSpec

# Load .env from the script's directory
env_path = Path(__file__).resolve().parent / '.env'
print("Loading .env from:", env_path)
load_dotenv(dotenv_path=env_path)

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX", "hr-handbook")

print("Loaded PINECONE_API_KEY:", PINECONE_API_KEY[:6] + "..." if PINECONE_API_KEY else "NOT FOUND")
print("Loaded OPENAI_API_KEY:", OPENAI_API_KEY[:6] + "..." if OPENAI_API_KEY else "NOT FOUND")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone (us-east-1)
def init_pinecone(index_name: str):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # ✅ using us-east-1 region
            )
        )
    return pc.Index(index_name)

# Load text files
def load_documents(root_dir: str) -> List[dict]:
    docs = []
    for path in Path(root_dir).rglob("*.txt"):
        category = path.parts[1] if len(path.parts) > 1 else "general"
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            docs.append({"id": str(path), "text": content, "category": category})
    return docs

# Embed and upsert to Pinecone
def index_documents(index, docs: List[dict]):
    for batch_start in range(0, len(docs), 100):
        batch = docs[batch_start:batch_start + 100]
        ids = [doc["id"] for doc in batch]
        texts = [doc["text"] for doc in batch]
        embeddings = client.embeddings.create(input=texts, model="text-embedding-ada-002")
        vectors = [
            (id_, emb.embedding, {"category": doc["category"]})
            for id_, emb, doc in zip(ids, embeddings.data, batch)
        ]
        index.upsert(vectors)

# Query Pinecone
def retrieve(query: str, index, category: str = None, k: int = 5) -> List[str]:
    embed = client.embeddings.create(input=[query], model="text-embedding-ada-002").data[0].embedding
    kwargs = {"top_k": k, "include_metadata": True}
    if category:
        kwargs["filter"] = {"category": {"$eq": category}}
    res = index.query(vector=embed, **kwargs)
    return [m["metadata"]["text"] for m in res["matches"] if "text" in m["metadata"]]

# Generate final answer
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
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content.strip()

# Gradio logic
def answer_question(query: str, category: str):
    docs = retrieve(query, pinecone_index, category)
    return generate_answer(query, docs)

# Main logic
if __name__ == "__main__":
    pinecone_index = init_pinecone(INDEX_NAME)

    if not int(os.getenv("SKIP_INDEXING", "0")):
        documents = load_documents(".")
        index_documents(pinecone_index, documents)

    categories = sorted({Path(p).parts[0] for p in glob.glob('*/*.txt')})
    recommended = {
        "all": ["Where can I find benefits information?", "How do I request time off?"],
        "benefits": ["What does private medical insurance cover?", "How do I join the pension scheme?"],
        "company": ["What is the company's mission?", "Where is the office located?"],
        "guides": ["How do I submit an expense?", "Where is the hiring policy?"],
        "roles": ["What does a data scientist do?", "How do career levels work?"],
        "communities-of-practice": ["How can I join a community of practice?", "When do CoPs meet?"],
    }

    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>HR Chatbot</h1>")
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

        examples.change(lambda q: q, inputs=examples, outputs=query)

        submit.click(lambda q, cat: answer_question(q, None if cat == "all" else cat),
                     inputs=[query, selected_category], outputs=answer)

    demo.launch()

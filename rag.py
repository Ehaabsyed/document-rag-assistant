import os
import chromadb
from fastembed import TextEmbedding
from dotenv import load_dotenv
from google import genai

load_dotenv()

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="documents"
)

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def ask_rag(question):

    query_embedding = list(
        embedding_model.embed([question])
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the information provided
in the context below.

If the answer cannot be found in the context,
say that you don't know based on the provided documents.

Context:
Respond in plain text only. Do not use markdown formatting
such as bullet points, headers, or bold text — write in
complete sentences and paragraphs.

{context}

Question:

{question}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text
import os
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from dotenv import load_dotenv
from google import genai

load_dotenv()


# Prevent chromadb from trying to load telemetry/gRPC modules
os.environ["ANONYMIZED_TELEMETRY"] = "False"


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return list(self.model.embed([text]))[0].tolist()


persistent_directory = "db/chroma_db"

embedding_model = FastEmbedEmbeddings("BAAI/bge-small-en-v1.5")

db=chroma = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_name="documents",
)

#ask query
# query = input("Ask a question: ")

# retrieval_pipeline = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"score_threshold": 0.5}    
# )

# relevant_docs = retrieval_pipeline.invoke(query)

# print(f"User Query: {query}")
# Display results
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")

# print("--- End of Context ---")
# print(relevant_docs)


def build_prompt(question, context):
    return f"""You are a helpful assistant that answers questions using ONLY the context provided below.

Rules:
- Answer strictly based on the context. Do not use outside knowledge.
- If the answer is not present in the context, respond exactly with: "I don't know based on the provided documents."
- Write your answer in plain text only. Do not use markdown formatting such as bullet points, headers, or bold text.
- Answer in complete sentences and paragraphs.

Context:
{context}

Question:
{question}

Answer:"""


def ask_rag(question):
    # Retrieve relevant documents from the vector store
    retrieval_pipeline = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5}
    )

    relevant_docs = retrieval_pipeline.invoke(question)

    # Combine the content of the retrieved documents into a single context string
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # Build the prompt for the language model
    prompt = build_prompt(question, context)

    # Generate a response using the Gemini API
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text

    

if __name__ == "__main__":
    query = input("Ask a question: ")
    answer = ask_rag(query)
    print("\n--- ANSWER ---")
    print(answer)

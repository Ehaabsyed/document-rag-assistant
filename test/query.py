import chromadb
from fastembed import TextEmbedding

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="documents"
)


def search(query, top_k=5):

    query_embedding = list(
        embedding_model.embed([query])
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


query = input("Ask a question: ")

results = search(query)

for i, document in enumerate(results["documents"][0]):
    print(f"\n--- RESULT {i + 1} ---")
    print(document)
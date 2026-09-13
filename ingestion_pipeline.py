import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()


# --- Wrapper so fastembed works with LangChain's Chroma integration ---
class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return list(self.model.embed([text]))[0].tolist()


def load_documents(data_folder):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                documents.append({
                    "filename": filename,
                    "text": text
                })
    return documents


def load_chunks(documents, chunk_size=500):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=50
    )
    chunks = []
    for document in documents:
        document_chunks = text_splitter.split_text(document["text"])
        for chunk in document_chunks:
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"filename": document["filename"]}
                )
            )
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = FastEmbedEmbeddings("BAAI/bge-small-en-v1.5")

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="documents",
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Starting the ingestion pipeline...")
    documents = load_documents("data")
    print(f"Loaded {len(documents)} documents.")

    chunks = load_chunks(documents)
    for chunk in chunks[:3]:
        print("\n--- CHUNK ---")
        print(chunk.page_content)

    vectorstore = create_vector_store(chunks)
    print(f"Stored {vectorstore._collection.count()} chunks in ChromaDB")


if __name__ == "__main__":
    main()
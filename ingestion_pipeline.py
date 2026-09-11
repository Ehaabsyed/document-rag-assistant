import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from google import genai
from fastembed import TextEmbedding
import chromadb
load_dotenv()
# client =genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")


def load_documents(data_folder):
    documents=[]
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            filepath=os.path.join(data_folder,filename)
            with open (filepath,"r",encoding="utf-8") as file:
                text=file.read()
                documents.append({
                "filename": filename,
                "text": text
            })

    return documents

def load_chunks(documents, chunk_size=500):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=50)
    chunks=[]
    for document in documents:
        document_chunks=text_splitter.split_text(document["text"])
        for chunk in document_chunks:
            chunks.append({
                "filename": document["filename"],
                "text": chunk
            })
    return chunks           

def create_embeddings(chunks):
    embeddings = []
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        print(f"Embedding {i + 1} to {i + len(batch)}...")

        texts = [chunk["text"] for chunk in batch]

        vectors = list(embedding_model.embed(texts))

        for chunk, vector in zip(batch, vectors):
            embeddings.append({
                "filename": chunk["filename"],
                "text": chunk["text"],
                "embedding": vector.tolist()
            })

    return embeddings
            

def main():
    print("Starting the ingestion pipeline...")
    documents=load_documents("data")
    print(f"Loaded {len(documents)} documents.")
    chunks=load_chunks(documents)
    for chunk in chunks[:3]:
        print("\n--- CHUNK ---")
        print(chunk["text"])
        
    embeddings = create_embeddings(chunks)

    print(f"Created embeddings for {len(embeddings)} chunks")
    print(embeddings[0]["embedding"][:10])
    
    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name="documents"
    )
    
    collection.add(
    ids=[str(i) for i in range(len(embeddings))],
    embeddings=[item["embedding"] for item in embeddings],
    documents=[item["text"] for item in embeddings],
    metadatas=[
        {"filename": item["filename"]}
        for item in embeddings
    ]
    )
    print(f"Stored {collection.count()} chunks in ChromaDB")    
    
    
if __name__ == "__main__":
    main()    
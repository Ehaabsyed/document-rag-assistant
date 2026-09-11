from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")

texts = [
    "Google is a technology company.",
    "Google was founded by Larry Page and Sergey Brin."
]

embeddings = list(model.embed(texts))

print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))
print("First 10 values:", embeddings[0][:10])
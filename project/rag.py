import chromadb
import requests

# ChromaDB client
client = chromadb.PersistentClient(path="../chroma_db")

# Collection for meeting documents
collection = client.get_or_create_collection(
    name="meeting_documents"
)

# Ollama embedding function
def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]


def add_document(text, document_id):
    embedding = get_embedding(text)

    collection.add(
        ids=[document_id],
        documents=[text],
        embeddings=[embedding]
    )


def search_documents(query, n_results=3):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results
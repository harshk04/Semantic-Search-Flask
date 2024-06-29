from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

# Initialize the embeddings model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize the Qdrant client
url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

# Initialize the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="documents")

# Perform a semantic search
query = "password manager"
docs = db.similarity_search_with_score(query=query, k=5)

# print(docs)
# Sort documents by score in descending order
sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)

# Print the top 2 documents with the highest scores
top_docs = sorted_docs[:2]

for doc, score in top_docs:
    print(f"Score: {score:.6f}")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 80)

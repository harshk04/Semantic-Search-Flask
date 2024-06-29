from flask import Flask, request, render_template
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

app = Flask(__name__)

# Initialize the embeddings model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize the Qdrant clients
url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

# Initialize the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="documents")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        docs = db.similarity_search_with_score(query=query, k=5)
        sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:2]
        results = []
        for doc, score in top_docs:
            results.append({
                'score': score,
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        return render_template('form.html', query=query, results=results)
    return render_template('form.html', query=None, results=None)

if __name__ == '__main__':
    app.run(debug=True)

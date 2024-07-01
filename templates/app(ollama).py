# APP USING OLLAMA

from flask import Flask, request, render_template
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import ollama

app = Flask(__name__)

# Initialize the embeddings model and Qdrant sclients (unchanged)
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="documents")

# Define function to query LLM (ollama)
def query_llm(prompt, context):
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': f"Using the following context, write about {prompt}:\n{context}",
        },
    ])
    return response['message']['content']

# Flask route for handling semantic search and displaying LLM response
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        docs = db.similarity_search_with_score(query=query, k=5)
        sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)
        
        # Select the top result with the highest score
        top_doc, top_score = sorted_docs[0]

        # Use top result for LLM prompt and context
        prompt = query  # Use query as prompt
        context = top_doc.page_content  # Use top document content as context

        # Query LLM with prompt and context
        llm_response = query_llm(prompt, context)
        
        return render_template('response.html', llm_response=llm_response)
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)

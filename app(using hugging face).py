from flask import Flask, request, render_template
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from gradio_client import Client

app = Flask(__name__)

# Initialize the embeddings model and Qdrant clients (unchanged)
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

# Define function to query LLM (Gradio client)
def query_llm(prompt, context):
    client = Client("harshk04/TextGeneration")
    result = client.predict(
        message=f"Generate response for the query '{prompt}' using the following context: {context}",
        system_message="You are a friendly chatbot",
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        api_name="/chat"
    )
    return result

# Flask route for handling semantic search and displaying LLM response
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        docs = db.similarity_search_with_score(query=query, k=5)
        sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)
        
        # Select the top result with the highest score
        top_doc, top_score = sorted_docs[0]

        prompt = query  # Use query as prompt
        context = top_doc.page_content  # Use top document content as context

        # Query LLM with prompt and context
        llm_response = query_llm(prompt, context)
        
        return render_template('response.html', llm_response=llm_response)
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)

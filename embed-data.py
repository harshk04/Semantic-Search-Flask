import nltk
import ssl
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# URL data loading
urls = [
    "https://brainlox.com/courses/category/technical"
]
url_loader = UnstructuredURLLoader(urls=urls)
url_data = url_loader.load()


# Combine the documents from URL and PDF
documents = url_data

# Split the combined documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load data into Qdrant
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="documents"
)

print("Documents successfully created and loaded into Qdrant!")


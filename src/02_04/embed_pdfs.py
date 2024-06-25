import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Define the path to the PDFs to be loaded
pdf_paths = [
    "../Data/botanical.pdf",
    "../Data/astronomical.pdf",
    "../Data/biological.pdf",
    "../Data/cosmological.pdf",
    "../Data/culinary.pdf",
    "../Data/pharmaceutical.pdf"
]

# Load PDF documents
loaders = [PyPDFLoader(path) for path in pdf_paths]

# Initialize a list to hold all the pages from the loaded PDFs
pages = []

# Load each PDF and extend the pages list with the content
for loader in loaders:
    pages.extend(loader.load())

# Define a text splitter to chunk the documents into smaller pieces
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

# Split the documents into smaller chunks
docs = text_splitter.split_documents(pages)

# Print the total number of document chunks
print(f"Number of document chunks: {len(docs)}")

# Initialize the OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a FAISS vector store from the document chunks and embeddings model
vectordb = FAISS.from_documents(docs, embeddings_model)

# Print the total number of vectors in the FAISS index
print(f"Total vectors in FAISS index: {vectordb.index.ntotal}")

# Save the FAISS index to a local directory
vectordb.save_local("faiss_index")

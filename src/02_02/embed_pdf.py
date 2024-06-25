from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the PDF loader and load the PDF document
loader = PyPDFLoader('michelle_obama_speech.pdf')
pages = loader.load()

# Initialize the text splitter with specific chunk size and overlap
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Split the loaded pages into smaller chunks of text
documents = text_splitter.split_documents(pages)

# Initialize the vector store with the documents and embed them using the
# specified model
vectordb = FAISS.from_documents(
    documents, OpenAIEmbeddings(model="text-embedding-3-small"))

# Print the total number of vectors in the index
print(vectordb.index.ntotal)

# Save the vector store to a local file
vectordb.save_local("faiss2_index")

# Initialize the embeddings model again
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the vector store from the local file with the embeddings model
# allow_dangerous_deserialization=True is used here with caution
new_db = FAISS.load_local("faiss2_index", embeddings_model,
                          allow_dangerous_deserialization=True)

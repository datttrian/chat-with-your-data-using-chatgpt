from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Load the FAISS index from a local directory
db = FAISS.load_local(
    "../faiss_index",  # Path to the local FAISS index
    # Specify the embedding model
    OpenAIEmbeddings(model="text-embedding-3-small"),
    # Allow deserialization (use with caution)
    allow_dangerous_deserialization=True,
)

# Perform a similarity search on the FAISS index
docs = db.similarity_search("What is Aetherfloris Ventus")

# Print the number of search results returned
print(len(docs))


# Define a function to print the search results
def print_output(docs):
    for doc in docs:
        print(
            "The output is: {}. \n\nThe metadata is {} \n\n".format(
                doc.page_content, doc.metadata
            )
        )


# Print the initial search results
print_output(docs)

# Perform a similarity search in the culinary section
docs = db.similarity_search("Can you recommend a herbal brew?")

# Print the number of search results returned
print(len(docs))

# Print the search results for the culinary section
print_output(docs)

# Perform a similarity search in the astronomical section
docs = db.similarity_search("What is the orbit of the sun-like figure?")

# Print the number of search results returned
print(len(docs))

# Print the search results for the astronomical section
print_output(docs)

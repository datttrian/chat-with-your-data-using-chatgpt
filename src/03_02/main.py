import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def print_output(docs):
    for doc in docs:
        print(
            "The output is: {}. \n\nThe metadata is {} \n\n".format(
                doc.page_content, doc.metadata
            )
        )


_ = load_dotenv(find_dotenv())  # read local .env file

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


db = FAISS.load_local(
    "../faiss_index",
    OpenAIEmbeddings(model="text-embedding-3-small"),
    allow_dangerous_deserialization=True,
)

docs = db.similarity_search("What is Aetherfloris Ventus")

# length of search results returned
len(docs)

print_output(docs)

# Search the culinary section
docs = db.similarity_search("Can you recommend a herbal brew?")

# length of search results returned
len(docs)

print_output(docs)

# Search the astronomical section
docs = db.similarity_search("What is the orbit of the sun-like figure?")

# length of search results returned
len(docs)

print_output(docs)

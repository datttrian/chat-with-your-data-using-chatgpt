from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Prompt the model with no additional knowledge of the Voynich manuscript beyond pretraining
llm.invoke("What are the medicinal insights from the Voynich manuscript?")
llm.invoke("What is Aetherfloris Ventus?")

# Load the FAISS index from a local directory
db = FAISS.load_local(
    "../faiss_index",  # Path to the local FAISS index
    # Specify the embedding model
    OpenAIEmbeddings(model="text-embedding-3-small"),
    # Allow deserialization (use with caution)
    allow_dangerous_deserialization=True
)

# Create a retriever from the FAISS index
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Pull the prompt from the hub
prompt = hub.pull("rlm/rag-prompt")

# Print the pulled prompt
print(prompt)


# Define a function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Combine multiple steps in a single chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()  # Convert the chat message to a string
)

# Stream the results for the specified queries
for chunk in rag_chain.stream("What are the medicinal insights from the Voynich manuscript?"):
    print(chunk, end="", flush=True)

for chunk in rag_chain.stream("What is Aetherfloris Ventus?"):
    print(chunk, end="", flush=True)

for chunk in rag_chain.stream("What's the most important part of the Voynich manuscript?"):
    print(chunk, end="", flush=True)

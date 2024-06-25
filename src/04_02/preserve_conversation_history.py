from dotenv import load_dotenv
from langchain import hub
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load OpenAI API Key from a .env file
load_dotenv()

# Initialize the LLM we'll use - OpenAI GPT 3.5 Turbo
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load the FAISS index from a local directory
db = FAISS.load_local(
    "../faiss_index",  # Path to the local FAISS index
    # Specify the embedding model
    OpenAIEmbeddings(model="text-embedding-3-small"),
    # Allow deserialization (use with caution)
    allow_dangerous_deserialization=True
)

# Configure retriever from the FAISS index
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Pull the prompt template from the hub
prompt = hub.pull("rlm/rag-prompt")

# Define a function to format documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Implement a chain that combines multiple steps
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()  # Convert the chat message to a string
)

# Stream and print the LLM's response for specified queries
for chunk in rag_chain.stream("What are the medicinal insights from the Voynich manuscript?"):
    print(chunk, end="", flush=True)

for chunk in rag_chain.stream("What is Aetherfloris Ventus?"):
    print(chunk, end="", flush=True)

for chunk in rag_chain.stream("What's the most important part of the Voynich manuscript?"):
    print(chunk, end="", flush=True)

# Preserve Conversation History
system_prompt = """Given the chat history and a recent user question \
generate a new standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed or otherwise return it as is."""

# Create a chat prompt template with a system prompt and placeholders for chat history and input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
retriever_with_history = create_history_aware_retriever(llm, retriever, prompt)

# Print the history-aware retriever configuration
print(retriever_with_history)

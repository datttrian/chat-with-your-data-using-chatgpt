from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load OpenAI API Key
load_dotenv()

# Initialize the LLM we'll use - OpenAI GPT 3.5 Turbo
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Prompt the model with no additional knowledge of the Voynich manuscript beyond pretraining
response_1 = llm.invoke(
    "What are the medicinal insights from the Voynich manuscript?")
print(response_1.content)

response_2 = llm.invoke("What is Aetherfloris Ventus?")
print(response_2.content)

# Load vector database from disk
db = FAISS.load_local(
    "../faiss_index",
    OpenAIEmbeddings(model="text-embedding-3-small"),
    allow_dangerous_deserialization=True,
)

# Configure retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Preserve Conversation History
system_prompt = """Given the chat history and a recent user question \
generate a new standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed or otherwise return it as is."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever_with_history = create_history_aware_retriever(llm, retriever, prompt)

# Perform question answering with chat history
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(
    retriever_with_history, question_answer_chain)

# Perform the conversation
chat_history = []

# First question
question = "What is Aetherfloris Ventus?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])
print(ai_msg_1["answer"])

# Second question
second_question = "What does a single drop of it do?"
ai_msg_2 = rag_chain.invoke(
    {"input": second_question, "chat_history": chat_history})
print(ai_msg_2["answer"])

# Third question
third_question = "How does it compare to Noctis Umbraherba?"
ai_msg_3 = rag_chain.invoke(
    {"input": third_question, "chat_history": chat_history})
print(ai_msg_3["answer"])

# Fourth question
fourth_question = (
    "Do you think the Biological section of the Voynich Manuscript is important?"
)
ai_msg_4 = rag_chain.invoke(
    {"input": fourth_question, "chat_history": chat_history})
print(ai_msg_4["answer"])

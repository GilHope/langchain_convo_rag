# Step 1: Set up environment and load necessary libraries
import os
from dotenv import load_dotenv
import getpass
import bs4 #Step 2: Load Docs

os.environ["USER_AGENT"] = "my-app/1.0"  # Set a custom User-Agent string

# LangChain imports
from langchain_community.document_loaders import WebBaseLoader #Step 2: Load Docs
from langchain_text_splitters import RecursiveCharacterTextSplitter #Step 2: Split Docs into Chunks
from langchain_chroma import Chroma #Step 2: Embed Chunks and store in Vector
from langchain_openai import OpenAIEmbeddings # Step 2: Embed Chunks and store in Vector
from langchain_community.chat_models import ChatOpenAI # Define LLM
from langchain.chains.combine_documents import create_stuff_documents_chain #Step 2: Prompt template for generating answers using retrieved context
from langchain.chains import create_retrieval_chain #Step 2: Prompt template for generating answers using retrieved context
from langchain_core.prompts import ChatPromptTemplate  # Import for creating the prompt template

# Load environment variables
load_dotenv()

# Retrieve API keys from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# LangSmith environment settings
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API Key: ")

# Step 2: Create retriever
# Load documents from the web
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed the chunks and store them in a Chroma vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for the Q&A chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the full retrieval chain that uses the retriever and the question-answering chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Testing the system
response = rag_chain.invoke({"input": "What is Task Decomposition?"})
print(response["answer"])
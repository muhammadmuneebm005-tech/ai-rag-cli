from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("1. Loading document...")
loader = TextLoader("sample.txt")
docs = loader.load()

print("2. Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

print("3. Generating embeddings & storing in ChromaDB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

print("4. Initializing local LLM (llama3.2)...")
llm = OllamaLLM(model="llama3.2")

print("5. Querying the document...")
system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Perform direct document retrieval
retrieved_docs = retriever.invoke("What is this document about?")
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# Format prompt and invoke local LLM
chain_prompt = prompt.format(context=context_text, input="What is this document about?")
response = llm.invoke(chain_prompt)

print("\n================ ANSWER ================")
print(response)
print("========================================")

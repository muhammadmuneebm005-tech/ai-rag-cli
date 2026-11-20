import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Page Config
st.set_page_config(page_title="Local RAG Assistant", page_icon="🔒", layout="wide")
st.title("🔒 Local Document RAG Assistant")
st.caption("Powered by LangChain, ChromaDB, and local Llama 3.2 via Ollama")

# Cache vector store creation so it only runs once
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("realpage_case.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(documents=splits, embedding=embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()
llm = OllamaLLM(model="llama3.2")

# User Input
st.subheader("Query Your Local Document")
user_query = st.text_input("Enter your question:", "How does the YieldStar algorithm use nonpublic data to fix prices according to the DOJ?")

if st.button("Run RAG Query", type="primary"):
    with st.spinner("Retrieving context & generating answer locally..."):
        system_prompt = (
            "You are a legal & AI security analyst. "
            "Answer the user's question using strictly the retrieved context below. "
            "If the answer is not in the context, state that you do not know.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        chain_prompt = prompt.format(context=context_text, input=user_query)
        response = llm.invoke(chain_prompt)
        
        st.markdown("### 🤖 Generated Response")
        st.success(response)
        
        with st.expander("🔍 View Retrieved Document Chunks"):
            for i, doc in enumerate(retrieved_docs):
                st.write(f"**Chunk {i+1}:**")
                st.info(doc.page_content)

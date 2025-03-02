import os
import re
import torch
import warnings
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

warnings.filterwarnings('ignore')
torch.classes.__path__ = []

os.environ["GROQ_API_KEY"] = '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'
llm = ChatGroq(model_name='llama-3.1-8b-instant', max_tokens=1000, max_retries=2)

if "e5_embed_model" not in st.session_state:
    model_name = "intfloat/multilingual-e5-large-instruct"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    st.session_state["e5_embed_model"] = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

e5_embed_model = st.session_state["e5_embed_model"]

chapters = {
    "Chapter 1": "Sexual Reproduction in Flowering Plants",
    "Chapter 2": "Human Reproduction",
    "Chapter 3": "Reproductive Health",
    "Chapter 4": "Principles of Inheritance and Variation",
    "Chapter 5": "Molecular Basis of Inheritance",
    "Chapter 6": "Evolution",
    "Chapter 7": "Human Health and Disease",
    "Chapter 8": "Microbes in Human Welfare",
    "Chapter 9": "Biotechnology : Principles and Processes",
    "Chapter 10": "Biotechnology and Its Applications",
    "Chapter 11": "Organisms and Populations",
    "Chapter 12": "Ecosystem",
    "Chapter 13": "Biodiversity and Conservation",
    "Chapter 14": "Environmental Issues"
}

def preprocess_text(text):
    text = re.sub(r'[^A-Za-z0-9\s\.,;:\'\"\?\!\-]', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\\[ntrb]', '', text)
    return text.strip()

def load_chapter_data(chapter_name):
    try:
        BASE_DIR = Path(__file__).parent
        
        pdf_path = BASE_DIR / "data" / "chapters" / f'{chapter_name}.pdf'
        index_path = BASE_DIR / "data" / "chapters" / f'{chapter_name}'

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ {pdf_path}")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ {index_path}")

        loader = PyPDFLoader(file_path=pdf_path)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=360)

        docs = [doc for doc in text_splitter.split_documents(document) if preprocess_text(doc.page_content)]

        vector_store = FAISS.load_local(index_path, e5_embed_model, allow_dangerous_deserialization=True)
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 1

        return docs, EnsembleRetriever(retrievers=[bm25_retriever, vector_store.as_retriever(search_kwargs={"k": 4}, search_type='mmr')], weights=[0.4, 0.6])
    
    except Exception as e:
        raise RuntimeError(f"⚠️ Error loading chapter data: {str(e)}")

def qa_chain(query, chat_history, ensemble_retriever):
    try:
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, contextualize_q_prompt)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Only use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        rag_chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})

        return {"answer": response['answer']}

    except Exception as e:
        return {"answer": f"⚠️ Error generating response: {str(e)}"}

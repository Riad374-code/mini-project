from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.vectorstores import FAISS
import langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_groq import ChatGroq
import numpy as np
import pandas as pd
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import trim_messages
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')


st.title("Your best friend during exam sessions!")
st.write("You can upload your PDFs' as you wish and get remarkable answers on your need")

llm=ChatOllama(model='llama3.1')
embedding=OllamaEmbeddings(model='mxbai-embed-large')
session_id=st.text_input("Session ID",value="Default ID")

if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files=st.file_uploader("Upload files to get superior answers",type='pdf',accept_multiple_files=True)
if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temp=f"./temp.pdf"
        with open(temp,'wb') as file:
            file.write(uploaded_file.getvalue())
        
        loader=PyPDFLoader(temp)
        docs=loader.load()
        documents.extend(docs)
    
    # Now we can divide into documents
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    final_docs=text_spliter.split_documents(docs)
    db=FAISS.from_documents(final_docs,embedding)
    retriever=db.as_retriever()

    # Input needed prompts
    contextualize_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    
    contextualize_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',contextualize_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ]
    )
       
    historical_ret_chain=create_history_aware_retriever(llm,retriever,contextualize_prompt)


    question_answer_system_prompt=(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know and give answer from your own knowledge. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
    )

    question_answer_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',question_answer_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ]
    )
   

    question_answer_chain=create_stuff_documents_chain(llm,question_answer_prompt)

    rag_chain=create_retrieval_chain(historical_ret_chain,question_answer_chain)

    # Now we create Runnable Historical chain to store our past conversations

    def session_saver(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,session_saver,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
        
    )

    input=st.text_input("Ask what you want about given information:")
    if input:
        session_history=session_saver(session_id)
        response = conversational_rag_chain.invoke(
            {"input": input},
            config={
                "configurable": {"session_id":session_id}
            },
        )
        
        st.write("Assistant:", response['answer'])
        st.write("Chat History:",  session_history.messages)






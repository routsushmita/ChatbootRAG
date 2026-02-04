import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

st.header("My First Streamlit App")

with st.sidebar:
    st.title("your Document ")
    file=st.file_uploader("Upload your PDF document here", type="pdf")

if file is not None:
    with pdfplumber.open(file) as pdf:
        text=""
        for page in pdf.pages:
            text+=page.extract_text() +"\n"
    # st.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    # st.write(chunks)


    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-ada-002",
    #     openai_api_key="OPENAI_API_KEY"
    # )

    # model_name = 'intfloat/e5-base-v2' # Example model, others available
    # model = SentenceTransformer(model_name)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # embeddings = model.encode(chunks, normalize_embeddings=True)
    # embeddings = model.embed_documents(chunks)

    print(type(embeddings), "================")
    # st.write(embeddings)
    # embeddings_array = np.array(embeddings).astype(np.float32)
    # print(type(embeddings_array), "================")
    vector_store= FAISS.from_texts(chunks, embeddings)  

    user_question = st.text_input("Ask your question here")
    # st.write("You entered:", user_question)
    # retriever = vector_store.as_retriever(
    #     search_type="mmr", search_kwargs={"k":3}
    #     )

    prompt = ChatPromptTemplate.from_messages(
        [   
            (
                "system",
                "You are a helpful AI assistant that helps people answer questions based on pdf.\n\n"
                "Guidelines:\n"
                "1. Provide accurate and concise answers based on the context provided.\n "
                "2. If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n"
                "3. summerize long information, ideally in bullet points where needed.\n"
                "context:\n{context}\n\n",
                
            ),
            ("human", "{question}"),
        ]
    )

    # # llm = ChatOpenAI(
    # #     model="gpt-3.5-turbo", 
    # #     temperature=0, 
    # #     openai_api_key="OPENAI_API_KEY"
    # #     max_Tokens=500
    # #     )

    llm = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 500},
    )

    chain = (
        {"context": retriever | format_docs, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | strOutputParser()
    )


    if user_question:
        response = chain.invoke({"question": user_question})
        st.write(response)
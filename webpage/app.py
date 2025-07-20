import time

import streamlit as st

import sqlite3
# Ensure all necessary imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
# from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
import os
import math

groq_api_key = "your api key here"
api_key_openai_2 = "your api key here"



st.set_page_config(page_title="AYP.ai",
                   page_icon="📚",
                   layout="wide"
)



st.markdown("<h1 style='text-align: center;'><span style='color: orange;'>ASK</span> <span style='color: white;'>YOUR</span> <span style='color: green;'>PDF</span></h1>", unsafe_allow_html=True)

page_bg_img = '''
<style>
.stApp {
background-image: url("https://d3nwecxvwq3b5n.cloudfront.net/AcuCustom/Sitename/DAM/044/ai_and_telecomms.jpg");
background-size: cover;
background-position: top center;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Upload your File: ")

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv"])

st.subheader("File summary: ")
if uploaded_file is not None:
    # Display file details
    st.write("File name:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size:", round(uploaded_file.size/1024, 2), "kb")

    with open(os.path.join("temp_pdf.pdf"), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Use PyPDFLoader to load the uploaded PDF file
    loader = PyPDFLoader("temp_pdf.pdf")
    docu = loader.load()

    # Add page number metadata to each page of the document
    for i, page in enumerate(docu):
        page.metadata['page'] = i + 1

    # Display the raw extracted text (optional)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    documents = text_splitter.split_documents(docu)

    ## LLM Model (open-sourced)
    llm1 = ChatGroq(
        model="llama-3.1-8b-instant",  # "mixtral-8x7b-32768" #"llama-3.1-70b-versatile", ## "llama-3.1-8b-instant"
        temperature=0.3,
        groq_api_key=groq_api_key
    )

    ## LLM Model (openai)
    llm2 = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key_openai_2,
        temperature=0
    )

    # Initialize OpenAI embeddings for use in the vector store
    embeddings_openai = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=api_key_openai_2,
        show_progress_bar=True
    )

    ## Delete the existing collection 
    db_path = '/home/bernadettem/bernadettenotebook/bernadettem/Ritwik/ritwik/Ask_Your_Pdf/database/chroma.sqlite3'  
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM collections")
    conn.commit()

    conn.close()

    ## Ollama Embedding (open-sourced)
    embeddings_ollama = OllamaEmbeddings(model = "mxbai-embed-large", show_progress=True)

    batch_size = 100
    num_batches = math.ceil(len(documents) / batch_size)

    # Process documents in batches to avoid exceeding the batch size limit
    vector_db = None  # Initialize as None
    path = "../database"
    with st.spinner('Processing the PDF file...'):
        for i in range(num_batches):
            batch = documents[i * batch_size: (i + 1) * batch_size]

            # Create or add to the Chroma vector store for the batch
            if vector_db is None:
                vector_db = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings_ollama,
                    collection_name="ritwik",
                    persist_directory=path
                )
            else:
                vector_db.add_documents(batch)

            time.sleep(2)

    st.success('PDF processed successfully!!, Now you can ask questions ...... ')
    # st.subheader("Content of your PDF:")
    # st.text_area("PDF Text", docu, height=500)



    # Define the query prompt template
    QUERY_PROMPT = """
        You are an AI language model specializing in information retrieval. Your task is to generate five distinct versions of the given user question to enhance document retrieval from a vector database.

        - **Objective**: Rephrase the user's question in five unique ways while maintaining the same semantic meaning. Focus on different ways a human might naturally ask the same question to improve the diversity of the search results.
        - Ensure each rephrased question is clear and concise. The purpose is to enhance retrieval performance by helping the system capture the nuances in how the question is framed.
        - **Important**: Avoid using synonyms that alter the original meaning. The rephrased questions should stay true to the user's intent, focusing only on rephrasing and restructuring.
        - **Rephrasing Quality**: Make sure each version of the question is unique in its phrasing but identical in its meaning, to help overcome any limitations of distance-based similarity search algorithms.

        At the end, return all five variations of the question in a list.

        **Original question**: {question}

        1.  
        2.  
        3.  
        4.  
        5.

        """

    # Convert the plain string into a PromptTemplate
    query_prompt_template = PromptTemplate(input_variables=["context", "question"], template=QUERY_PROMPT)

    # Use MultiQueryRetriever to retrieve answers based on similarity search and the provided LLM
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        llm=llm1,
        prompt=query_prompt_template  # Ensure this is a PromptTemplate, not a plain string
    )

    # Define the ChatPromptTemplate for extracting structured answers with page numbers
    template = """
    You are an intelligent AI assistant, answer the question based ONLY on the given below context:
    {context}

    For each 'question' mentioned in the context, generate an structured answer that satisfy the user question:
    **Instruction**:

     - Read carefully the user question.
     - Only from the given 'context' you need to answer the question.
     - You need to fetch all the possible answers of the 'question' from the 'context' only.
     - Do not go beyond of the 'context' given.
     - read the 'context' carefully till the end, and from answer the 'question' correctly .
     - At the end of 'answer', you need to mention the 'page numbers', from where you have taken the answer.


    Question: {question}
    """

    # Create the ChatPromptTemplate from the template
    prompt = ChatPromptTemplate.from_template(template)

    # Create the full chain (retriever, prompt, llm)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm1
            | StrOutputParser()
    )

    st.subheader("Your Question: ")

    # Input for the user to ask a question
    ques = st.text_input("Enter your question:")

    butt = st.button("Get Answer")

    if butt:

        with st.spinner('Processing your question...'):
            time.sleep(3)  

            retrieved_context = []
            for doc in retriever.get_relevant_documents(ques):
                retrieved_context.append({
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", "Unknown")  # Fetch the page number from the metadata
                })

            # Pass context and question to the chain
            res = chain.invoke({"context": retrieved_context, "question": ques})



        # Display the answer
        st.success("Answer generated successfully!")

        st.subheader("Retrieved Content:")
        st.text_area("Content", retrieved_context, height=300)


        st.subheader("Answer of Question:")
        st.text_area("Answer", res, height=300)








from dotenv import load_dotenv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAI

def main():
    load_dotenv()
    google_api_key=os.environ.get("GOOGLE_API_KEY")
    st.set_page_config(page_title="Ask your PDF",page_icon=":books:")
    st.header("Ask your PDF")
    
    #uploading the file
    pdf=st.file_uploader("upload your PDF",type="pdf")

    #extract the text 
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #split into chunks 
        text_splitter=CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks=text_splitter.split_text(text)

        #embedding 
        embeddings=HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
        document=FAISS.from_texts(chunks,embeddings)
        
        #show the user input 
        user_ques=st.text_input("Ask a ques about your PDF:")
        if user_ques:
            docs=document.similarity_search(user_ques)

            #model
            llm= GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key)
            chain=load_qa_chain(llm,chain_type="stuff")

            #response
            response=chain.run(input_documents=docs,question=user_ques)
            st.write(response)


if __name__=='__main__':
    main()

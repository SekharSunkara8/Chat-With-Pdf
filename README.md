ChatwithPDF's
This repository provides a Streamlit application that allows you to interact with your PDFs in a conversational manner. Simply upload a PDF document, ask a question about its content, and the app will retrieve relevant passages and generate a response using Google's large language models (LLMs).

Key Features:
The PDF Conversational App efficiently extracts text from uploaded PDF files, breaking it down into manageable chunks for processing. It then employs a pre-trained sentence transformer model, paraphrase-MiniLM-L6-v2, to create semantic representations of these text portions. By using FAISS, the app quickly identifies relevant text segments based on their similarities. User queries are addressed using Google's powerful text-BISON-001 model, which generates responses that succinctly summarize pertinent information from the PDF.Then, Streamlit ensures a smooth and user-friendly interface, facilitating seamless interaction with the application.


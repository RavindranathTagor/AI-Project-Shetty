import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text and associate with PDF filename
def get_pdf_text_with_references(pdf_docs):
    """
    Extracts text from each PDF and associates it with the PDF filename for references.
    """
    pdf_texts = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_texts.append({"text": text, "source": pdf.name})
    return pdf_texts


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to store embeddings in FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Summarization function
def summarize_with_references(pdf_texts):
    """Summarizes PDFs using Gemini Pro and provides references"""
    # Initialize Gemini Pro
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Create summary prompt
    summary_prompt = """
    Provide a comprehensive summary of the following text. Include main topics, key points and important details:
    
    {text}
    """
    
    combined_text = ""
    references = []
    
    # Combine texts with source tracking
    for pdf in pdf_texts:
        combined_text += f"\nContent from {pdf['source']}:\n{pdf['text']}\n"
        references.append(f"Source: {pdf['source']}")

    # Generate summary using Gemini
    summary = model.predict(summary_prompt.format(text=combined_text))
    
    # Format with references
    summary_with_refs = (
        f"Summary:\n{summary}\n\n"
        f"References:\n" + "\n".join(references)
    )
    return summary_with_refs


# Conversational chain for answering questions
def get_conversational_chain():
    """Enhanced QA chain with source attribution"""
    prompt_template = """
    Using the context provided, answer the question comprehensively.
    Include relevant details and provide citations in the following format:
    [Source: filename.pdf]
    
    Context: {context}
    Question: {question}
    
    Answer with citations:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, 
                          input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Function to handle user questions
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, 
                                 allow_dangerous_deserialization=True)

        # Get relevant docs
        docs = new_db.similarity_search(user_question, k=4)
        
        chain = get_conversational_chain()
        response = chain({
            "input_documents": docs, 
            "question": user_question
        }, return_only_outputs=True)

        # Format response
        st.write("Answer:", response["output_text"])
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")


# Streamlit application
def main():
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="💁",
        layout="wide"  # Enable wide layout
    )

    # Custom CSS for floating input box at the middle bottom
    st.markdown(
        """
        <style>
        .floating-input {
            position: fixed;
            bottom: 20px; /* Distance from the bottom */
            left: 50%; /* Center horizontally */
            transform: translateX(-50%); /* Adjust for perfect centering */
            width: 40%; /* Width of the input box */
            z-index: 1000;
            padding: 10px;
        }
        .floating-input input {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header Section
    st.header("Chat with PDF using Gemini")

    # Placeholder for dynamic summary
    summary_placeholder = st.container()

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing all uploaded PDFs..."):
                    # Extract text with references
                    pdf_texts_with_refs = get_pdf_text_with_references(pdf_docs)

                    # Process text chunks and store embeddings
                    raw_text = " ".join([pdf["text"] for pdf in pdf_texts_with_refs])
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)

                    # Summarize and include references
                    combined_summary = summarize_with_references(pdf_texts_with_refs)

                    # Display summary below the header in markdown format
                    with summary_placeholder:
                        st.subheader("Summary of All PDFs with References")
                        st.markdown(
                            f"<div style='color: white;'>{combined_summary}</div>",
                            unsafe_allow_html=True,
                        )

                    st.success("Processing completed! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    # Floating input box for questions
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()

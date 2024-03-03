import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader


os.environ['OPENAI_API_KEY']="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"
#os.environ['GOOGLE_API_KEY'] = 'AIzaSyDejN_hgur_8NyVg7A454fG6-P61HaA1NU'


def get_pdf_text(pdf_docs):
    text = ""
    for uploaded_file in pdf_docs:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile_path = tmpfile.name

        loader = PyPDFLoader(tmpfile_path)
        pages = loader.load_and_split()
        for page in pages:
            text += page.page_content
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    #model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   #temperature=0.3, google_api_key='AIzaSyDejN_hgur_8NyVg7A454fG6-P61HaA1NU')
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):

    embeddings = OpenAIEmbeddings(openai_api_key="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")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Multiple PDF💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
# New streamlit that can read pdf and summarise findings of pdf
# (Before running, install the following)
# pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub tiktoken
# pip install InstructionEmbedding -- very heavy packages that are not necessary if you use openai

# import 
from apikey import apikey
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)         #created pdf object that has pages
        for page in pdf_reader.pages:       #loop through each page to reach the text
            text+=page.extract_text()       #extract all text of the page and concatenate it into the text variable
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    #
    llm = ChatOpenAI(temperature=0.9)
    # initialize an instance of memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']

    #loop along the whole conversation with an index
    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    return

def main():
    #add a page title and a icon for the page, get open ai key
    os.environ['OPENAI_API_KEY'] = apikey
    st.set_page_config(page_title="Ask Questions about your pdf files", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)
    #initialize the session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask questions about your PDFs :books:")
    user_question=st.text_input("Ask a question about your files:")

    if user_question:
        handle_userinput(user_question)

    #To tell streamlit that the css code is safe to display the css vode
    st.write(user_template.replace("{{MSG}}","Hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello human"), unsafe_allow_html=True)

    #add a side bar for user to add documents
    with st.sidebar:
        st.subheader("Your documents")
        # Create a variable that store the files
        pdf_docs = st.file_uploader("Upload your pdf files here and click on Process. File size must be <200MB",  accept_multiple_files=True)
        #Only run (process the content of the files) when a user uploads the files
        if st.button("Process"):
            #Add a spinner to tell the users that the bot is processing, not frozen
            with st.spinner("Processing"):
                #step 1: Get the text of all the pdf files
                #create a variable called raw_text that store all the raw text by calling a function get_pdf_test

                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #Step 2: Get the text chucks
                text_chunks=get_text_chunks(raw_text)
                st.write(text_chunks)

                #Step 3: Create a vector store for the embeddings
                vectorstore = get_vectorstore(text_chunks)

                #Step 4: Create an instance of the conversation chain, session_state makes the conversation consistent, and won't reinitialize the whole conversation
                st.session_state.conversation = get_conversation_chain(vectorstore)

                #embeddings = OpenAIEmbeddings(openai_api_key=apikey)
                #vectorstore = faiss.from_texts(texts=text_chunks, embedding=embeddings)




# only run this file when the file is directly run, not from importing
if __name__ =='__main__':
    main ()

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
# from langchain_community.embeddings import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env
print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
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


# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # return vectorstore
    try:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        print("Embedding model loaded successfully!")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except ImportError as e:
        print("Error:", e)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
    # load_dotenv()
    # st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []

    # st.write(css, unsafe_allow_html=True)

    # # Display chat history
    # for message in st.session_state.chat_history:
    #     role = "You" if message["role"] == "user" else "Reply"
    #     st.markdown(f"**{role}**")
    #     st.markdown(f"{message['content']}")

    # # User input at the bottom
    # # user_question = st.chat_input("Ask a question...")
    # user_question = st.text_input("Ask a question...")
    
    # # if user_question:
    # #     # Store user input
    # #     st.session_state.chat_history.append({"role": "user", "content": user_question})
        
    # #     # Process and generate a response
    # #     if st.session_state.conversation:
    # #         response = st.session_state.conversation({'question': user_question})
    # #         bot_reply = response['chat_history'][-1].content  # Get the latest bot response
    # #         st.session_state.chat_history.append({"role": "bot", "content": bot_reply})
        
    # #     # Refresh the page to display the chat
    # #     # st.experimental_rerun()
    # #     # st.rerun()
    # #     st.markdown(f"**Reply:**")
    # #     st.markdown(f"{bot_reply}")

    # if user_question:
    #     # Store user input
    #     st.session_state.chat_history.append({"role": "user", "content": user_question})

    #     bot_reply = "I couldn't generate a response."  # Default value to avoid errors

    #     # Process and generate a response
    #     if st.session_state.conversation:
    #         response = st.session_state.conversation({'question': user_question})
    #         if response and 'chat_history' in response and response['chat_history']:
    #             bot_reply = response['chat_history'][-1].content  # Get the latest bot response
    #             st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

    #     # Display bot response immediately
    #     st.markdown(f"**Reply:**")
    #     st.markdown(f"{bot_reply}")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:  # Store PDF text persistently
        st.session_state.pdf_text = None

    st.write(css, unsafe_allow_html=True)

    # **PDF Upload Section**
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload your PDFs", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            pdf_text = get_pdf_text(uploaded_files)
            st.session_state.pdf_text = pdf_text  # Save PDF text in session_state
            text_chunks = get_text_chunks(pdf_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("PDFs processed successfully! You can now ask questions.")

    # **Display chat history**
    for message in st.session_state.chat_history:
        role = "You" if message["role"] == "user" else "Reply"
        st.markdown(f"**{role}**")
        st.markdown(f"{message['content']}")

    # **User Input Section**
    user_question = st.text_input("Ask a question...")
    
    if user_question:
        # Store user input
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        bot_reply = "I couldn't generate a response."  # Default fallback

        # **Ensure PDF is loaded before answering**
        if st.session_state.conversation:
            response = st.session_state.conversation({'question': user_question})
            if response and 'chat_history' in response and response['chat_history']:
                bot_reply = response['chat_history'][-1].content
                st.session_state.chat_history.append({"role": "bot", "content": bot_reply})

        # Display bot response
        st.markdown(f"**Reply:**")
        st.markdown(f"{bot_reply}")



if __name__ == '__main__':
    main()

import os
import io
import streamlit as st
from dotenv                  import load_dotenv
from PyPDF2                  import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings    import OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores  import FAISS
from langchain.memory        import ConversationBufferMemory
from langchain.chains        import ConversationalRetrievalChain
from langchain.llms          import HuggingFaceHub
from langchain.chat_models   import ChatOpenAI
from datetime                import datetime

embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')

st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon=":books:"
    )

st.title("Chat with Multiple PDFs" )

# Função para obter a consulta do usuário e substituir a entrada de texto
def get_query():
    #input_text = st.text_input("Ask a question about your documents...")
    input_text = st.chat_input("Ask a question about your documents...")
    return input_text

def save_conversation_history(conversation_text):
    with io.StringIO() as buffer:
        buffer.write(conversation_text)
        st.download_button(
            label="Click to download conversation",
            data=buffer.getvalue(),
            file_name="conversation.txt",
            mime="text/plain"
        )

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
def get_text_chunks(text):
    #text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    #text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=70, length_function=len)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks
def get_vector_store(text_chunks):
    #embeddings=OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #embeddings=HuggingFaceEmbeddings(model_name = embedding_model_name)
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore
def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1000)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.2, max_tokens=1000)
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']

    if len(st.session_state.chat_history) > 0:

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                # st.write(message)
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                # st.write(message)
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
def main():
    load_dotenv()
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    if "last_saved_question" not in st.session_state:
        st.session_state.last_saved_question = None  # Adicionando a variável last_saved_question

    user_question = get_query()
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        #st.header("Chat with PDF 💬")
        #st.title("LLM Chatapp using LangChain")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload the PDF Files here and Click on Process", accept_multiple_files=True)
        st.markdown('''
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM Model
        ''')
        if st.button('Process'):
            with st.spinner("Processing"):
                #Extract Text from PDF
                raw_text = get_pdf_text(pdf_docs)
                #Split the Text into Chunks
                text_chunks = get_text_chunks(raw_text)
                #Create Vector Store
                vectorstore=get_vector_store(text_chunks)
                # Create Conversation Chain
                st.session_state.conversation=get_conversation_chain(vectorstore)
                st.success("Done!")

        if st.button('Download Conversation History'):
            if st.session_state.chat_history is not None:
                conversation_content = ""
                for i in range(0, len(st.session_state.chat_history), 2):
                    question = st.session_state.chat_history[i].content.replace("content=", "")
                    answer = st.session_state.chat_history[i + 1].content.replace("content=", "")
                    conversation_content += f"question: {question}\n"
                    conversation_content += f"answer: {answer}\n\n"
                save_conversation_history(conversation_content)

if __name__ == "__main__":
    main()

# Streamlit으로 RAG 시스템 구축하기
# 
# https://www.youtube.com/watch?v=xYNYNKJVa4E&t=822s
# https://github.com/HarryKane11/langchain/tree/main

# pip install pysqlite3-binary langchain chromadb unstructured sentence-transformers faiss-cpu tiktoken openai pypdf loguru docx2txt langchain-openai python-dotenv



import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


from dotenv import load_dotenv    # # 설치: pip install python-dotenv
load_dotenv()

# *********************************************************************************************
# LangSmith로 디버깅을 위해
import os

os.environ["LANGCHAIN_TRACING_V2"]="true"                              # LangSmith tracing 하기
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="ls__05992a0a44c744a5b3a618b690b31f9b"  # LangSmith(smith.langchain.com) API Key
os.environ["LANGCHAIN_PROJECT"] = "langchain-study"    # 디버깅을 위한 프로젝트명을 기입합니다.
# *********************************************************************************************

#OpenAPI KEY 입력 받기
# openai_key = st.text_input('OPENAI_API_KEY', type="password")
openai_api_key = os.environ["OPENAI_API_KEY"]


def main():
    st.set_page_config(
    page_title="업로드 파일 기반 챗팅",
    page_icon=":mag_right:")

    st.title("_:red[업로드 파일] 기반 챗팅_ :mag_right:")

    container = st.container(border=True)
    container.write("-- 질문 예시 --")
    container.write("내용 요약 해줘")
    container.write("좀더 상세히 설명해줘")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("파일 업로드",type=['pdf','docx'],accept_multiple_files=True)
        # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("시작")

        with st.expander("참고"):
            st.markdown("모두의AI 참고", help = "https://www.youtube.com/watch?v=xYNYNKJVa4E&t=822s")
            st.markdown("LangChain, Chromadb, Faiss, OpenAI, PDF, MSWord, LangSmith")
    
    if process:
        file_count = 0
        for uploaded_file in uploaded_files:
            file_count += 1

        if 0 == file_count:
            st.info("파일을 업로드 해주세요.")
            st.stop()

        if not openai_api_key:
            st.info("OpenAI API key를 입력해 주세요.")
            st.stop()

        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "파일 업로드 -> '시작' 버튼 클릭 -> 궁금한 내용 질문"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        if None == st.session_state.processComplete:
            st.info("시작 버튼을 클릭해 주세요.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain.invoke({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        # model_name = "jhgan/ko-sbert-nli"
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )
    
    # embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
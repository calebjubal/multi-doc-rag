import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from htmlTemplates import css
import os
from langchain_cerebras import ChatCerebras


# ---------------- EMBEDDINGS WRAPPER ---------------- #

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


# ---------------- PDF + TEXT UTILS ---------------- #

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings("BAAI/bge-base-en-v1.5")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# ---------------- LLM + CHAIN ---------------- #

def get_conversation_chain(vectorstore):
    # llm = HuggingFaceEndpoint(
    #     repo_id="deepseek-ai/DeepSeek-R1-0528",
    #     task="text-generation",
    #     max_new_tokens=512,
    #     do_sample=False,
    #     repetition_penalty=1.03,
    #     provider="auto",
    # )

    llm = ChatCerebras(
        model="llama-3.3-70b",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        temperature=0.7,
        max_tokens=1024,
    )

    # chat_model = ChatHuggingFace(llm=llm)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        # llm=chat_model,
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# ---------------- RESPONSE PARSER ---------------- #

def split_thinking_and_answer(text: str):
    thinking = ""
    answer = text

    if "<think>" in text and "</think>" in text:
        start = text.find("<think>") + len("<think>")
        end = text.find("</think>")
        thinking = text[start:end].strip()
        answer = text[end + len("</think>"):].strip()

    note_marker = "*(Note:"
    if note_marker in answer:
        answer = answer.split(note_marker)[0].strip()

    return thinking, answer


# ---------------- CHAT HANDLER ---------------- #

def handle_user_input(user_question):
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        response = st.session_state.conversation.invoke(
            {"question": user_question}
        )

        st.session_state.chat_history = response["chat_history"]

        thinking, answer = split_thinking_and_answer(
            response["chat_history"][-1].content
        )

        if thinking:
            with st.expander("Thinking"):
                st.write(thinking)

        st.write(answer)


# ---------------- MAIN APP ---------------- #

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Docs", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple Documents 📚")

    # -------- CHAT AREA (ALWAYS RENDERED) -------- #
    chat_container = st.container()

    with chat_container:
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    with st.chat_message("user"):
                        st.markdown(message.content)
                else:
                    thinking, answer = split_thinking_and_answer(message.content)
                    with st.chat_message("assistant"):
                        if thinking:
                            with st.expander("Thinking"):
                                st.write(thinking)
                        st.write(answer)
        else:
            # Spacer so input stays at bottom even with no messages
            st.markdown(
                "<div style='height: 60vh'></div>",
                unsafe_allow_html=True
            )

    # -------- INPUT (ALWAYS LAST) -------- #
    user_question = st.chat_input("Ask a question about your documents")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

    # -------- SIDEBAR -------- #
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your documents here",
            accept_multiple_files=True
        )

        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):

                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)
                
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Documents processed successfully!")


if __name__ == "__main__":
    main()
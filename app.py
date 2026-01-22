import streamlit as st

def main():
    st.set_page_config(page_title="My Streamlit App", layout="wide")

    st.header("Chat with multiple Documents :books: all at once!!!!!")
    st.text_input("Ask a question about your documents: ")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your documents here", accept_multiple_files=True)
        st.button("Process")

if __name__ == "__main__":
    main()
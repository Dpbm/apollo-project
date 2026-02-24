import streamlit as st

st.write("# Genetic Syndrome statistics visualizer")

st.file_uploader("Upload Embeddings Pickle", type="p", accept_multiple_files=False, help="Insert the embeddings pickle file that has extension .p")


"""Generate dashboard using streamlit."""

import io

import streamlit as st

from analysis import load_pickle, generate_df, get_overall_statistics, generate_array_from_embeddings

st.header("Genetic Syndrome statistics visualizer",divider=True)

file = st.file_uploader("Upload Embeddings Pickle", type="p", accept_multiple_files=False, help="Insert the embeddings pickle file that has extension .p")

if file is not None:
    st.write(f"## Using file: {file.name}")
    
    dataset = load_pickle(io.BytesIO(file.getvalue()))
    df = generate_df(dataset)
    statistics = get_overall_statistics(df)

    st.dataframe(df)

    st.subheader("Overall Data Statistics", divider=True)

    table_data = {}
    for key,value in statistics.items():
        if("amount" in key):
            continue
        table_data[key] = str(value)

    st.table(table_data)

    st.subheader("Amount of Images per Syndrome", divider=True)
    st.table(statistics["syndromes_amount_of_images"], border="horizontal")

    st.subheader("Amount of Subjects per Syndrome", divider=True)
    st.table(statistics["syndromes_amount_of_subjects"], border="horizontal")




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
    table_data = {
            "Total Images": statistics["total_images"],
            "Total Syndromes": statistics["total_syndromes"],
            "Total Subjects": statistics["total_subjects"],
    }
    st.table(table_data)
    
    st.subheader("Embeddings Values", divider=True)
    st.caption("This section show how the embedding values are distributed.")
    for col_sufix, type_col in [("_min_value","Min"), ("_max_value", "Max")]:
        row = st.container(horizontal=True)
        with row:
            col1,col2,col3,col4 = st.columns(4)
            col1.metric(f"Min {type_col} Value", statistics["min"+col_sufix], None, border=True)
            col2.metric(f"Avg {type_col} Value", statistics["avg"+col_sufix], None, border=True)
            col3.metric(f"Median {type_col} Value", statistics["median"+col_sufix], None, border=True)
            col4.metric(f"Max {type_col} Value", statistics["max"+col_sufix], None, border=True)

    st.subheader("Amount of Images per Syndrome", divider=True)
    st.table(statistics["syndromes_amount_of_images"], border="horizontal")

    st.subheader("Amount of Subjects per Syndrome", divider=True)
    st.table(statistics["syndromes_amount_of_subjects"], border="horizontal")




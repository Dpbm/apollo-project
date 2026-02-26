"""Generate dashboard using streamlit."""

import io

import streamlit as st
import plotly.express as px
import pandas as pd

from analysis import (
        load_pickle, 
        generate_df, 
        get_overall_statistics, 
        generate_t_sne, 
        generate_array_from_embeddings,
        DEFAULT_PERPLEXITY,
        MIN_PERPLEXITY,
        MAX_PERPLEXITY,
        DEFAULT_EXAGGERATION,
        MIN_EXAGGERATION,
        MAX_EXAGGERATION,
        DEFAULT_MAX_ITER,
        MIN_MAX_ITER,
        MAX_MAX_ITER,
        DEFAULT_LEARNING_RATE,
        MIN_LEARNING_RATE,
        MAX_LEARNING_RATE
    )


if __name__ == "__main__":

    st.header("Genetic Syndrome statistics visualizer",divider=True)

    file = st.file_uploader("Upload Embeddings Pickle", type="p", accept_multiple_files=False, help="Insert the embeddings pickle file that has extension .p")

    if file is not None:
        st.write(f"## Using file: {file.name}")
        
        dataset = load_pickle(io.BytesIO(file.getvalue()))
        df = generate_df(dataset)
        statistics = get_overall_statistics(df)

        st.dataframe(df)

        st.header("Overall Data Statistics", divider=True)
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
        
        row = st.container(horizontal=True)
        with row:
            col1, col2 = st.columns(2)

            col1.subheader("Amount of Images per Syndrome", divider=True)
            col1.table({"Total Images":statistics["syndromes_amount_of_images"]}, border="horizontal")

            col2.subheader("Amount of Subjects per Syndrome", divider=True)
            col2.table({"Total Syndromes":statistics["syndromes_amount_of_subjects"]}, border="horizontal")
        
        embeddings_array = generate_array_from_embeddings(dataset,df)
        tsne_data = generate_t_sne(
                embeddings_array,
                perplexity=DEFAULT_PERPLEXITY if not "perplexity" in st.session_state else st.session_state["perplexity"],
                exaggeration=DEFAULT_EXAGGERATION if not "exaggeration" in st.session_state else st.session_state["exaggeration"],
                max_iter=DEFAULT_MAX_ITER if not "max_iter" in st.session_state else st.session_state["max_iter"],
                learning_rate=DEFAULT_LEARNING_RATE if not "learning_rate" in st.session_state else st.session_state["learning_rate"],
            )
        x_data = tsne_data[:,0]
        y_data = tsne_data[:,1]
        plot_data = pd.DataFrame({
            "x": x_data,
            "y": y_data,
            "label": df.syndrome
        })

        fig = px.scatter(
                plot_data,
                x="x",
                y="y",
                color="label",
                hover_name="label"
                )

        col1, col2 = st.columns(2)
        with col1:
            st.header("T-SNE plot")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.header("Hyper-parameters")
            st.session_state["perplexity"] = st.slider("Perplexity", MIN_PERPLEXITY, MAX_PERPLEXITY, DEFAULT_PERPLEXITY)
            st.session_state["exaggeration"] = st.slider("Exaggeration", MIN_EXAGGERATION, MAX_EXAGGERATION, DEFAULT_EXAGGERATION)
            st.session_state["max_iter"] = st.slider("Max Iter", MIN_MAX_ITER, MAX_MAX_ITER, DEFAULT_MAX_ITER)
            st.session_state["learning_rate"] = st.slider("Learning Rate", MIN_LEARNING_RATE, MAX_LEARNING_RATE, DEFAULT_LEARNING_RATE)


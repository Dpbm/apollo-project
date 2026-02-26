"""Generate dashboard using streamlit."""

import io
import logging

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
        MAX_LEARNING_RATE,
        OPTIONS_INIT,
        DEFAULT_INIT,
        DEFAULT_INIT_INDEX
    )
from constants import BANNER_IMAGE


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    st.image(BANNER_IMAGE, caption="Company banner")

    st.title("Genetic Syndrome statistics visualizer and Predictor")

    file = st.file_uploader("Upload Embeddings Pickle", type="p", accept_multiple_files=False, help="Insert the embeddings pickle file that has extension .p")

    if file is None:
        st.write("Add your pickle file to proceed with your analysis and model training!")
        st.write("Make sure your data inside is a dictionary that follows this format:")
        st.json({
            'syndrome_id': {
                'subject_id': {
                    'image_id': '[320-dimensional embedding ndarray]'
                    }
                }
            })
    else:
        # all data
        failed = False
        exception = None
        with st.status("Processing data..."):
            try:
                st.write("Loading pickle file..")
                dataset = load_pickle(io.BytesIO(file.getvalue()))

                st.write("Creating DataFrame..")
                df = generate_df(dataset)

                st.write("Retrieving statistics..")            
                statistics = get_overall_statistics(df)

                grouped_amount_of_images = df.groupby("syndrome").count()["image"].sort_values(ascending=True).reset_index()
                grouped_amount_of_subjects = df.groupby("syndrome")["subject"].nunique().sort_values(ascending=True).reset_index()
                grouped_statistics = pd.merge(grouped_amount_of_images, grouped_amount_of_subjects, on="syndrome")
                
                grouped_images_per_subject = df.groupby(["syndrome", "subject"])["image"].count().sort_values(ascending=True).reset_index()

                st.write("Parsing embeddings..")            
                embeddings_array = generate_array_from_embeddings(dataset,df)

                # plot data
                st.write("Reducing embeddings dimensions..")            
                tsne_data = generate_t_sne(
                        embeddings_array,
                        perplexity=DEFAULT_PERPLEXITY if not "perplexity" in st.session_state else st.session_state["perplexity"],
                        exaggeration=DEFAULT_EXAGGERATION if not "exaggeration" in st.session_state else st.session_state["exaggeration"],
                        max_iter=DEFAULT_MAX_ITER if not "max_iter" in st.session_state else st.session_state["max_iter"],
                        learning_rate=DEFAULT_LEARNING_RATE if not "learning_rate" in st.session_state else st.session_state["learning_rate"],
                        init=DEFAULT_INIT if not "init" in st.session_state else st.session_state["init"],
                    )
                x_data = tsne_data[:,0]
                y_data = tsne_data[:,1]
                plot_data = pd.DataFrame({
                    "x": x_data,
                    "y": y_data,
                    "label": df.syndrome
                })
                
            except Exception as error:
                logger.error(f"Failed on load data: {error}")
                failed = True
                exception = error

        # ------- ON SCREEN ELEMENTS --------
        if failed:
            st.error('An error occurred during processing. Please, try again with a different file', icon="🚨")
            st.exception(Exception("No Exception found!") if exception is None else exception)
            exit()

        st.write(f"Using file: {file.name}")
        st.space(size="small")
        
        st.header("Data Analysis and visualization", divider=True)
        st.dataframe(df)
        st.caption("Extracted data from dataset pickle file. This dataframe holds the main characteristics from images and their classes.")

        table_data = {
                "Total Images": statistics["total_images"],
                "Total Syndromes": statistics["total_syndromes"],
                "Total Subjects": statistics["total_subjects"],
        }
        st.table(table_data)
        
        fig = px.line(
                grouped_statistics,
                x="syndrome",
                y=["subject", "image"],
                title="Amount of Images/Subjects per Syndrome",
                markers=True,
                labels={
                    "value": "Amount",
                    "variable": "Type",
                    "syndrome": "Syndrome"
                }
            )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig)

        fig = px.line(
                grouped_images_per_subject,
                x="subject",
                y="image",
                color="syndrome",
                title="Amount of Images per Subject",
                markers=True,
                labels={
                    "image": "Amount of Images",
                    "subject":"Subject",
                    "syndrome":"Syndrome",
                },
            )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig)

        
        

        st.subheader("Embedding Values")
        for col_sufix, type_col in [("_min_value","Min"), ("_max_value", "Max")]:
            row = st.container(horizontal=True)
            with row:
                col1,col2,col3,col4 = st.columns(4)
                col1.metric(f"Min {type_col} Value", statistics["min"+col_sufix], None, border=True)
                col2.metric(f"Avg {type_col} Value", statistics["avg"+col_sufix], None, border=True)
                col3.metric(f"Median {type_col} Value", statistics["median"+col_sufix], None, border=True)
                col4.metric(f"Max {type_col} Value", statistics["max"+col_sufix], None, border=True)
        st.caption("In this section, we show how the min and max values from embeddings are distributed.")
        
        fig = px.scatter(
            df, 
            x="min_value", 
            y="max_value", 
            color="syndrome", 
            hover_name="image",
            labels={
                "max_value":"Max Value",
                "min_value":"Min Value",
                "syndrome":"Syndrome"
            },
            title="Dispersion of Embeddings Min/Max Values"
            )
        st.plotly_chart(fig)

        st.divider()


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
            st.session_state["init"] = st.selectbox("Init", OPTIONS_INIT, index=DEFAULT_INIT_INDEX)
            st.session_state["perplexity"] = st.slider("Perplexity", MIN_PERPLEXITY, MAX_PERPLEXITY, DEFAULT_PERPLEXITY)
            st.session_state["exaggeration"] = st.slider("Exaggeration", MIN_EXAGGERATION, MAX_EXAGGERATION, DEFAULT_EXAGGERATION)
            st.session_state["max_iter"] = st.slider("Max Iter", MIN_MAX_ITER, MAX_MAX_ITER, DEFAULT_MAX_ITER)
            st.session_state["learning_rate"] = st.slider("Learning Rate", MIN_LEARNING_RATE, MAX_LEARNING_RATE, DEFAULT_LEARNING_RATE)


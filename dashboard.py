"""Generate dashboard using streamlit."""

import io
import logging
from typing import Dict

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

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
from model import (
        Model, 
        ModelOutputMetrics, 
        KNNMetric, 
        NumerableMetrics,
        DEFAULT_FOLDS,
        DEFAULT_SELECTED_FOLD,
        OPTIONS_METRICS,
        DEFAULT_METRIC_INDEX,
        NUMERABLE_METRICS_LIST
    )
from constants import BANNER_IMAGE

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def only_numberable_metrics_for_overall(metrics:ModelOutputMetrics) -> Dict[KNNMetric, NumerableMetrics]:
    """
    Parse model metrics to return a dict with only 
    numerable metrics from overall metrics.
    """
    return {
        knn_metric: {
            metric: metrics[knn_metric]["overall"][metric] for metric in NUMERABLE_METRICS_LIST
        } for knn_metric in OPTIONS_METRICS
    }

def get_numerables_per_fold(metrics:ModelOutputMetrics) -> pd.DataFrame:
    """Get numerables for each fold."""
    df = pd.DataFrame(columns=("fold", "knn_metric", *NUMERABLE_METRICS_LIST))

    loc_i = 0
    for knn_metric in OPTIONS_METRICS:
        for fold in range(DEFAULT_FOLDS):
            df.loc[loc_i] = {
                "fold":fold,
                "knn_metric": knn_metric,
                **{metric:metrics[knn_metric]["per_fold"][fold][metric]  for metric in NUMERABLE_METRICS_LIST}
            }
            loc_i += 1

    return df

def roc_to_df_for_overall(metrics:ModelOutputMetrics, knn_metric:KNNMetric) -> pd.DataFrame:
    """Create a df for plotting roc-auc for overall."""
    df = None
    for col in ["auc", "fpr", "tpr"]:
        data = metrics[knn_metric]["overall"][col]
        tmp_df = pd.DataFrame({
            "class":list(data.keys()),
            col: list(data.values())
        })

        if(df is None):
            df = tmp_df.copy()
            continue

        df = pd.merge(df, tmp_df, on="class")
    
    df = df.explode(["fpr", "tpr"])
    df["fpr"] = pd.to_numeric(df["fpr"], errors="coerce")
    df["tpr"] = pd.to_numeric(df["tpr"], errors="coerce")

    df = df.dropna()
    df = df.sort_values(["class", "fpr"])

    return df

def roc_to_df_for_mean(metrics:ModelOutputMetrics, knn_metric:KNNMetric) -> pd.DataFrame:
    """Create a df for plotting roc-auc per fold mean."""
    data = metrics[knn_metric]["mean_roc"]
    fpr = metrics[knn_metric]["fprs_grid"]
    df = pd.DataFrame({
        "class": list(data.keys()),
        "auc": [value["auc"] for value in data.values()],
        "tpr":[value["tpr"] for value in data.values()],
        "fpr": [fpr  for _ in data.values()],
    })
    
    df = df.explode(["fpr", "tpr"])
    df["fpr"] = pd.to_numeric(df["fpr"], errors="coerce")
    df["tpr"] = pd.to_numeric(df["tpr"], errors="coerce")
    df = df.dropna()
    df = df.sort_values(["class", "fpr"])
    return df
    

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
                sorted_syndromes = list(df.syndrome.sort_values(ascending=True).unique())

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
            
                st.write("Creating model..")            
                model = Model(embeddings_array,df.syndrome)
                train_size, test_size, train_syndromes, test_syndromes = model.train_test_size

                st.write("Finding best parameters for KNN..")            
                model.find_best_params()

                st.write("Training model...")
                model.run_training()
                per_fold_df = get_numerables_per_fold(model.metrics)

                st.write("Evaluating model...")
                model.run_evaluation()


            except Exception as error:
                logger.error(f"Failed on load data: {error}")
                failed = True
                exception = error

        # ------- ON SCREEN ELEMENTS --------
        if failed:
            st.error('An error occurred during processing. Please, try again with a different file!', icon="🚨")
            st.exception(Exception("No Exception found!") if exception is None else exception)
            exit()

        tab1, tab2 = st.tabs(
                [":chart: Analysis", ":robot: Model"], default=":chart: Analysis"
            )
        
        with tab1:

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
                st.selectbox("Init", OPTIONS_INIT, index=DEFAULT_INIT_INDEX, key="init")
                st.slider("Perplexity", MIN_PERPLEXITY, MAX_PERPLEXITY, DEFAULT_PERPLEXITY, key="perplexity")
                st.slider("Exaggeration", MIN_EXAGGERATION, MAX_EXAGGERATION, DEFAULT_EXAGGERATION, key="exaggeration")
                st.slider("Max Iter", MIN_MAX_ITER, MAX_MAX_ITER, DEFAULT_MAX_ITER, key="max_iter")
                st.slider("Learning Rate", MIN_LEARNING_RATE, MAX_LEARNING_RATE, DEFAULT_LEARNING_RATE, key="learning_rate")

        with tab2:
            st.header("Model creation", divider=True)

            st.subheader("Best Parameters for each KNN metric (n_neighbors)")
            row = st.container(horizontal=True)
            with row:
                cols = st.columns(len(OPTIONS_METRICS))
                for i,metric in enumerate(OPTIONS_METRICS):
                    cols[i].metric(metric, model.best_params[metric]["n_neighbors"], None, border=True)

            st.space(size="small")

            st.subheader("Model Final Performance")
            st.table(only_numberable_metrics_for_overall(model.metrics))

            st.table({
                "train":{**train_syndromes, "total":train_size}, 
                "evaluation":{**test_syndromes,"total":test_size}
                })

            row = st.container(horizontal=True)
            with row:
                cols = st.columns(len(OPTIONS_METRICS))
                for i,metric in enumerate(OPTIONS_METRICS):
                    fig = px.imshow(
                        model.metrics[metric]["overall"]["confusion_matrix"], 
                        text_auto=True,
                        x=sorted_syndromes,
                        y=sorted_syndromes,
                        title=f"Confusion Matrix for {metric} KNN model"
                        )
                    fig.update_xaxes(side="top", type="category")
                    fig.update_yaxes(type="category")
                    cols[i].plotly_chart(fig, use_container_width=True)

            row = st.container(horizontal=True)
            with row:
                cols = st.columns(len(OPTIONS_METRICS))
                for i,metric in enumerate(OPTIONS_METRICS):
                    fig = px.imshow(
                        model.metrics[metric]["overall"]["f1"].reshape(1, -1), 
                        text_auto=True,
                        x=sorted_syndromes,
                        aspect="equal",
                        title=f"F1 for {metric} KNN model"
                        )
                    fig.update_xaxes(type="category")
                    fig.update_yaxes(showticklabels=False, visible=False)
                    cols[i].plotly_chart(fig, use_container_width=True)

            row = st.container(horizontal=True)
            with row:
                cols = st.columns(len(OPTIONS_METRICS))
                for i,metric in enumerate(OPTIONS_METRICS):
                    fig = px.line(
                        per_fold_df[per_fold_df["knn_metric"] == metric],
                        x="fold",
                        y=list(filter(lambda x: x not in ['knn_metric', 'fold'], list(per_fold_df.columns))),
                        title=f"Metrics for {metric} KNN model per fold",
                        labels={
                            "value":"Score",
                            "fold":"Fold",
                            "variable":"Metric"
                        }
                    )
                    cols[i].plotly_chart(fig)

            container = st.container()
            with container:

                selected_fold = DEFAULT_SELECTED_FOLD if not 'fold' in st.session_state else st.session_state['fold']
                metric = OPTIONS_METRICS[DEFAULT_METRIC_INDEX] if not 'knn_metric' in st.session_state else st.session_state['knn_metric']

                plots_row = st.container(horizontal=True)
                fig = px.imshow(
                        model.metrics[metric]["per_fold"][selected_fold]["confusion_matrix"], 
                        text_auto=True,
                        x=sorted_syndromes,
                        y=sorted_syndromes,
                        title=f"Confusion Matrix for fold {selected_fold} - {metric}"
                    )
                fig.update_xaxes(side="top", type="category")
                fig.update_yaxes(type="category")
                plots_row.plotly_chart(fig, use_container_width=True)

                fig = px.imshow(
                        model.metrics[metric]["per_fold"][selected_fold]["f1"].reshape(1, -1), 
                        text_auto=True,
                        x=sorted_syndromes,
                        aspect="equal",
                        title=f"F1 for fold {selected_fold} - {metric}"
                    )
                fig.update_xaxes(type="category")
                fig.update_yaxes(showticklabels=False, visible=False)
                plots_row.plotly_chart(fig, use_container_width=True)

                tweaks_row = st.container(horizontal=True)
                with tweaks_row:
                    st.select_slider("Fold index", options=list(range(DEFAULT_FOLDS)), value=DEFAULT_SELECTED_FOLD, key="fold")
                    st.selectbox("KNN Metric", options=OPTIONS_METRICS, index=DEFAULT_METRIC_INDEX, key="knn_metric")

            row = st.container(horizontal=True)
            with row:
                cols = st.columns(len(OPTIONS_METRICS))
                for i,metric in enumerate(OPTIONS_METRICS):
                    roc = roc_to_df_for_overall(model.metrics, metric)
                    fig = px.line(
                        roc,
                        x="fpr",
                        y="tpr",
                        color="class",
                        hover_name="class",
                        hover_data={
                                "fpr": True,
                                "tpr": True,
                                "auc": True,
                                "class": True
                            },
                        labels={
                            "tpr":"TPR",
                            "fpr":"FPR",
                            "class":"Class"
                        },
                        title=f"ROC-AUC model {metric}"
                    )
                    fig.update_xaxes(type="linear")
                    fig.update_yaxes(type="linear")
                    cols[i].plotly_chart(fig)
            
            row = st.container(horizontal=True)
            with row:
                cols = st.columns(len(OPTIONS_METRICS))
                for i,metric in enumerate(OPTIONS_METRICS):
                    roc = roc_to_df_for_mean(model.metrics, metric)
                    fig = px.line(
                            roc,
                            x="fpr",
                            y="tpr",
                            color="class",
                            hover_name="class",
                            hover_data={
                                    "fpr": True,
                                    "tpr": True,
                                    "auc": True,
                                    "class": True,
                                },
                            labels={
                                "tpr":"TPR",
                                "fpr":"FPR",
                                "class":"Class"
                            },
                            title=f"ROC-AUC model {metric} mean across folds"
                        )
                    fig.update_xaxes(type="linear")
                    fig.update_yaxes(type="linear")
                    cols[i].plotly_chart(fig)
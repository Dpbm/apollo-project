"""Do the data analysis and transformation"""

from typing import Dict, Any, TypedDict, Tuple
from io import BytesIO

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

import pandas as pd
import numpy as np

from constants import MAX_JOBS,RANDOM_STATE


MIN_PERPLEXITY = 5.0
MAX_PERPLEXITY = 50.0
DEFAULT_PERPLEXITY = 50.0

MIN_EXAGGERATION = 4.0
MAX_EXAGGERATION = 100.0
DEFAULT_EXAGGERATION = 12.0

MIN_MAX_ITER = 100
MAX_MAX_ITER = 6000
DEFAULT_MAX_ITER = 1000

MIN_LEARNING_RATE = 10.0
MAX_LEARNING_RATE = 1000.0
DEFAULT_LEARNING_RATE = 800.0

OPTIONS_INIT = ('pca', 'random')
DEFAULT_INIT_INDEX = 0
DEFAULT_INIT = OPTIONS_INIT[DEFAULT_INIT_INDEX]

type InitType = 'pca'|'random' # type: ignore

type Dataset = Dict[Any,Any]

class Statistics(TypedDict):
    """Statistics returning data type."""
    total_images:int
    total_syndromes:int
    total_subjects:int
    min_max_value:float
    max_max_value:float
    avg_max_value:float
    median_max_value:float
    min_min_value:float
    max_min_value:float
    avg_min_value:float
    median_min_value:float
    syndromes_amount_of_subjects:Dict[str,int]
    syndromes_amount_of_images:Dict[str,int]

def load_pickle(file:BytesIO) -> Dataset:
    """Load dataset pickle file."""
    return np.load(file, allow_pickle=True)

def generate_df(dataset:Dataset) -> pd.DataFrame:
    """Extract featues from pickle dataset and insert into a pandas DataFrame."""

    mapping = pd.DataFrame(columns=("syndrome", "subject", "image", "image_size", "max_value", "min_value"))

    index = 0
    for syndrome in dataset.keys():
      subjects = dataset[syndrome].keys()

      for subject in subjects:
        images = dataset[syndrome][subject].keys()

        for image_id in images:
          image = dataset[syndrome][subject][image_id]
          mapping.loc[index] = {"syndrome":syndrome, "subject":subject, "image":image_id, "image_size":len(image), "max_value":max(image), "min_value":min(image)}
          index += 1

    assert len(mapping.syndrome.unique()) >= 3, "You must have at least 3 different syndromes!"

    return mapping

def get_overall_statistics(df:pd.DataFrame) -> Statistics:
   """Get Statistics from dataframe.""" 

   total_images = df.shape[0]
   syndromes = df.syndrome.unique()
   total_syndromes = len(syndromes)
   total_subjects = len(df.subject.unique())

   min_max_value = df.max_value.min()
   max_max_value = df.max_value.max()
   avg_max_value = df.max_value.mean()
   median_max_value = df.max_value.median()
   
   min_min_value = df.min_value.min()
   max_min_value = df.min_value.max()
   avg_min_value = df.min_value.mean()
   median_min_value = df.min_value.median()

   statistics = {
        "total_images":total_images,
        "total_syndromes":total_syndromes,
        "total_subjects":total_subjects,
        "min_max_value":min_max_value,
        "max_max_value":max_max_value,
        "avg_max_value":avg_max_value,
        "median_max_value":median_max_value,
        "min_min_value":min_min_value,
        "max_min_value":max_min_value,
        "avg_min_value":avg_min_value,
        "median_min_value":median_min_value,
        "syndromes_amount_of_subjects":{},
        "syndromes_amount_of_images":{},
   }


   for syndrome in syndromes:
       amount_of_subjects = len(df[df.syndrome == syndrome].subject.unique())
       statistics["syndromes_amount_of_subjects"][syndrome] = amount_of_subjects

   for syndrome in syndromes:
       amount_of_images = len(df[df.syndrome == syndrome].image)
       statistics["syndromes_amount_of_images"][syndrome] = amount_of_images
    
   return statistics


def generate_array_from_embeddings(dataset:Dataset, df:pd.DataFrame) -> np.ndarray:
    """Generate an easier to handle data structure for those images."""

    total_images = df.shape[0]
    max_img_size = df.image_size.max()

    data_arr = np.ndarray((total_images,max_img_size), dtype=np.float32)

    for i,row in df.iterrows():
      syndrome,subject,image_i = row["syndrome"],row["subject"],row["image"]
      image = dataset[syndrome][subject][image_i]
      data_arr[i] = np.nan_to_num(image)

    return data_arr

def generate_t_sne(
        data:np.ndarray,
        perplexity:float=DEFAULT_PERPLEXITY,
        exaggeration:float=DEFAULT_EXAGGERATION,
        max_iter:int=DEFAULT_MAX_ITER,
        learning_rate:float=DEFAULT_LEARNING_RATE,
        init:InitType=DEFAULT_INIT
    ) -> np.ndarray:
    """Reduce dimensions from dataset using t-SNE."""
    tsne = TSNE(
            n_components=2, 
            learning_rate=learning_rate, 
            init=init, 
            perplexity=perplexity, 
            early_exaggeration=exaggeration,
            max_iter=max_iter,
            n_jobs=MAX_JOBS,
            random_state=RANDOM_STATE
            )
    data_arr_normalized = normalize(data,'l2',axis=0)
    return tsne.fit_transform(data_arr_normalized)

def generate_default_t_sne(data:np.ndarray) -> np.ndarray:
    """Reduce dimensions from dataset using t-SNE with default sklearn parameters."""

    tsne = TSNE(n_components=2, n_jobs=MAX_JOBS, random_state=RANDOM_STATE)
    data_arr_normalized = normalize(data,'l2',axis=0)

    return tsne.fit_transform(data_arr_normalized)

def under_sample(embeddings:np.ndarray, syndromes:pd.Series) -> Tuple[pd.Series, np.ndarray]:
    """Uses RepeatedEditedNearestNeighbours to reduce the noisy on the data"""
    knn = RepeatedEditedNearestNeighbours()

    X_resampled, y_resampled = knn.fit_resample(embeddings,syndromes)
    return y_resampled, X_resampled

def get_syndromes_amount(syndromes:pd.Series) -> pd.DataFrame:
   """Count the occurrences of each syndrome."""
   return syndromes.value_counts().reset_index()

def join_under_sampled_and_raw_for_comparison(under_sampled:pd.Series, raw:pd.DataFrame) -> pd.DataFrame:
   """Join Series and Data Frame for comparison."""

   under_sampled_c_syndromes_count = get_syndromes_amount(under_sampled)
   under_sampled_c_syndromes_count["type"] = "under_sampled"

   raw_c = raw.copy()
   raw_c_syndromes_count = get_syndromes_amount(raw_c.syndrome)
   raw_c_syndromes_count["type"] = "raw"


   return pd.concat([under_sampled_c_syndromes_count, raw_c_syndromes_count], ignore_index=True).sort_values(by='syndrome')
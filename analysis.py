"""Do the data analysis and transformation"""

from typing import Dict, Any, TypedDict
from io import BytesIO

import pandas as pd
import numpy as np

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

    total_images = mapping.shape[0]
    max_img_size = df.image_size.max()

    data_arr = np.ndarray((total_images,max_img_size), dtype=np.float32)

    for i,row in mapping.iterrows():
      syndrome,subject,image_i = row["syndrome"],row["subject"],row["image"]
      image = dataset[syndrome][subject][image_i]
      data_arr[i] = np.nan_to_num(image)

    return data_arr



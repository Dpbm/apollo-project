"""Defined the model class and its attributes."""

from typing import TypedDict, Dict, List, Tuple

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
        roc_auc_score, 
        roc_curve, 
        f1_score, 
        top_k_accuracy_score, 
        auc, 
        accuracy_score, 
        confusion_matrix, 
        precision_score, 
        recall_score
    )

import numpy as np
import pandas as pd

from constants import MAX_JOBS,RANDOM_STATE

DEFAULT_FOLDS = 10
DEFAULT_SELECTED_FOLD = 0

DEFAULT_METRIC_INDEX = 0
OPTIONS_METRICS = ["euclidean", "cosine"]

type KNNMetric = 'euclidean' | 'cosine' # type: ignore

class Params(TypedDict):
    """Model Parameters Dict Typed."""
    n_neighbors:int

    @classmethod
    def create_default(cls):
        return Params(n_neighbors=1)


type ParamsPerKNNMetric = Dict[KNNMetric,Params]

NUMERABLE_METRICS_LIST = ["roc_auc", "top_k", "accuracy", "precision", "recall"]
class NumerableMetrics(TypedDict):
    """Metrics that are a single number."""
    roc_auc:float
    top_k:float
    accuracy:float
    precision:float
    recall:float

class ModelMetrics(TypedDict, NumerableMetrics):
    """Model metrics for the test stage."""
    confusion_matrix:np.ndarray
    f1:List[float]
    fpr:Dict[str,np.ndarray] 
    tpr:Dict[str,np.ndarray] 
    auc:Dict[str,np.ndarray] 

    @classmethod
    def create_default(cls):
        return ModelMetrics(
            roc_auc=0.0,
            f1=0.0,
            top_k=0.0,
            accuracy=0.0,
            confusion_matrix=np.empty(1),
            precision=0.0,
            recall=0.0,
            auc={},
            fpr={},
            tpr={},
        )
    
class RocMetrics(TypedDict):
    """Metrics for Mean ROC-AUC."""
    tpr:np.ndarray
    auc:np.ndarray


class ModelOutputMetrics(TypedDict):
    """The Output metrics for test and folds."""
    overall: ModelMetrics
    per_fold: List[ModelMetrics]
    fprs_grid: np.ndarray # base for mean roc
    mean_roc: Dict[str,RocMetrics]

    @classmethod
    def create_default(cls):
        return ModelOutputMetrics(
                overall=ModelMetrics.create_default(),
                per_fold=[],
                fprs_grid=np.linspace(0.0, 1.0, 100),
                mean_roc={}
            )

type ModelMetricsPerKNNMetric = Dict[KNNMetric, ModelOutputMetrics]

class Model:
    """The KNN model based on Sklearn's but with some extra functionality."""

    def __init__(self, X:np.ndarray, y:pd.Series, folds:int=DEFAULT_FOLDS):
        """Setup model object."""

        self._folds = folds
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        self._train_size = self._y_train.shape[0]
        self._test_size = self._y_test.shape[0]
        self._train_proportion_syndromes = self._y_train.value_counts().to_dict()
        self._test_proportion_syndromes = self._y_test.value_counts().to_dict()
        
        # default "best"
        self._best_params = {metric: Params.create_default() for metric in OPTIONS_METRICS}
        self._fit_models = {metric: None for metric in OPTIONS_METRICS}
        self._model_metrics = {metric:ModelOutputMetrics.create_default() for metric in OPTIONS_METRICS}

    @property
    def best_params(self) -> ParamsPerKNNMetric:
        """Get the best params found."""
        return self._best_params
    
    @property
    def metrics(self) -> ModelOutputMetrics:
        """Get metrics."""
        return self._model_metrics
    
    @property
    def train_test_size(self) -> Tuple[int,int,Dict[str,int],Dict[str,int]]:
        """Get Train - Test - Train classes - Test classes size."""
        return (
            self._train_size, 
            self._test_size, 
            self._train_proportion_syndromes,
            self._test_proportion_syndromes
        ) 

    def find_best_params(self, neighbors_range:np.arange=np.arange(1,16)):
        """Do a search for the best parameters using GridSearchCV for KNN."""
        for metric in OPTIONS_METRICS:

            model = KNeighborsClassifier(metric=metric)
            params = {"n_neighbors":neighbors_range}
            gcv = GridSearchCV(
                model,
                params,
                cv=StratifiedKFold(
                    n_splits=self._folds, 
                    random_state=RANDOM_STATE, 
                    shuffle=True),
                n_jobs=MAX_JOBS
            )

            gcv.fit(self._X_train, self._y_train)
            self._best_params[metric] = gcv.best_params_

    def run_training(self):
        """Run training method for each metric."""
        for metric in OPTIONS_METRICS:
            self._train_by_knn_metric(metric)
        

    def _train_by_knn_metric(self,metric:KNNMetric):
        """Train model using an specific KNN metric and calculates the folds metrics."""

        cv = StratifiedKFold(n_splits=self._folds, random_state=RANDOM_STATE, shuffle=True)

        # for safely getting the indexed values, we need to convert y_train 
        # from pd.Series to np.ndarray
        y_train_np = self._y_train.to_numpy()
        
        params = self._best_params[metric]
        n_neighbors = params["n_neighbors"]
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

        for train_idx, test_idx in cv.split(self._X_train, y_train_np):
            X_train_fold, X_test_fold = self._X_train[train_idx], self._X_train[test_idx]
            y_train_fold, y_test_fold = y_train_np[train_idx], y_train_np[test_idx]

            model.fit(X_train_fold, y_train_fold)

            classes = model.classes_
            y_pred_matrix = model.predict_proba(X_test_fold)
            y_pred = classes[y_pred_matrix.argmax(axis=1)]

            metrics = self._calculate_metrics(classes, y_pred_matrix, y_pred, y_test_fold)
            self._model_metrics[metric]["per_fold"].append(metrics)

        self._fit_models[metric] = model
        self._calculate_mean_auc_per_fold(metric)

    def run_evaluation(self):
        """Run test method for each metric."""
        for metric in OPTIONS_METRICS:
            self._test_by_knn_metric(metric)

    def _test_by_knn_metric(self, metric:KNNMetric):
        """Test model using an specific KNN metric and calculates the metrics."""
        model = self._fit_models[metric]
        assert model is not None, "You must train the model before testing!"

        classes = model.classes_
        y_pred_matrix = model.predict_proba(self._X_test)
        y_pred = classes[y_pred_matrix.argmax(axis=1)]

        metrics = self._calculate_metrics(classes, y_pred_matrix, y_pred, self._y_test)
        self._model_metrics[metric]["overall"] = metrics

    def _calculate_metrics(self, classes:List[str], y_pred_matrix:np.ndarray, y_pred:np.ndarray, y:np.ndarray) -> ModelMetrics:
        """Get metrics from model predictions."""
        fprs = {}
        tprs = {}
        aucs = {}
        # calculate ROC_AUC for each class
        for i,class_ in enumerate(classes):
            y_test_binary = (y == class_).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_matrix[:,i])
            auc_value = auc(fpr, tpr)

            fprs[class_] = fpr
            tprs[class_] = tpr
            aucs[class_] = auc_value


        return ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            f1=f1_score(y, y_pred, average=None),
            confusion_matrix=confusion_matrix(y,y_pred),
            precision=precision_score(y, y_pred, average="micro"),
            recall=recall_score(y, y_pred, average="micro"),
            top_k=top_k_accuracy_score(y, y_pred_matrix, labels=classes),
            roc_auc=roc_auc_score(y, y_pred_matrix, multi_class="ovr"),
            tpr=tprs,
            fpr=fprs,
            auc=aucs
        )
    
    def _calculate_mean_auc_per_fold(self, knn_metric:KNNMetric):
        """
        Calculate the average AUC per fold.

        It's based on methods that are shown at:
        - https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        - https://stackoverflow.com/questions/47876999/how-to-compute-average-roc-for-cross-validated-for-multiclass
        """

        model = self._fit_models[knn_metric]
        assert model is not None, "You must train the model before calculating mean AUC!"
        classes = model.classes_

        fprs_grid = self._model_metrics[knn_metric]["fprs_grid"]

        pre_computed_metrics = self._model_metrics[knn_metric]

        for class_ in classes:
            tprs_history = []
            aucs_history = []
            

            for fold in range(self._folds):

                fpr = pre_computed_metrics["per_fold"][fold]["fpr"][class_]
                tpr = pre_computed_metrics["per_fold"][fold]["tpr"][class_]
                auc_value = pre_computed_metrics["per_fold"][fold]["auc"][class_]

                new_tpr = np.interp(fprs_grid, fpr, tpr)
                new_tpr[0] = 0.0

                tprs_history.append(new_tpr)
                aucs_history.append(auc_value)

            
            mean_tpr = np.mean(tprs_history, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(fprs_grid, mean_tpr)

            self._model_metrics[knn_metric]["mean_roc"][class_] = RocMetrics(
                                                                    auc=mean_auc, 
                                                                    tpr=mean_tpr
                                                                )
                                                            
"""Generates a list of time-series model classes for the pipeline."""

from typing import Any, List

from ml_grid.model_classes_time_series.ArsenalClassifier_module import Arsenal_class
from ml_grid.model_classes_time_series.CNNClassifier_module import CNNClassifier_class
from ml_grid.model_classes_time_series.ContractableBOSSClassifier_module import (
    ContractableBOSS_class,
)
from ml_grid.model_classes_time_series.EncoderClassifier_module import (
    EncoderClassifier_class,
)
from ml_grid.model_classes_time_series.FCNClassifier_module import FCNClassifier_class
from ml_grid.model_classes_time_series.FreshPRINCEClassifier_module import (
    FreshPRINCEClassifier_class,
)
from ml_grid.model_classes_time_series.HIVECOTEV1Classifier_module import (
    HIVECOTEV1_class,
)
from ml_grid.model_classes_time_series.HIVECOTEV2Classifier_module import (
    HIVECOTEV2_class,
)
from ml_grid.model_classes_time_series.InceptionTimeClassifer_module import (
    InceptionTimeClassifier_class,
)
from ml_grid.model_classes_time_series.IndividualInceptionClassifier_module import (
    IndividualInceptionClassifier_class,
)
from ml_grid.model_classes_time_series.InidividualTDEClassifier_module import (
    IndividualTDE_class,
)
from ml_grid.model_classes_time_series.KNeighborsTimeSeriesClassifier_module import (
    KNeighborsTimeSeriesClassifier_class,
)


# from ml_grid.model_classes import LogisticRegression_class
from ml_grid.model_classes_time_series.MLPClassifier_module import MLPClassifier_class
from ml_grid.model_classes_time_series.MUSEClassifier_module import MUSE_class
from ml_grid.model_classes_time_series.OrdinalTDEClassifier_module import (
    OrdinalTDE_class,
)
from ml_grid.model_classes_time_series.ResNetClassifier_module import (
    ResNetClassifier_class,
)
from ml_grid.model_classes_time_series.SignatureClassifier_module import (
    SignatureClassifier_class,
)
from ml_grid.model_classes_time_series.SummaryClassifier_module import (
    SummaryClassifier_class,
)
from ml_grid.model_classes_time_series.TSFreshClassifier_module import (
    TSFreshClassifier_class,
)
from ml_grid.model_classes_time_series.TapNetClassifier_module import (
    TapNetClassifier_class,
)
from ml_grid.model_classes_time_series.TemporalDictionaryEnsembleClassifier_module import (
    TemporalDictionaryEnsemble_class,
)
from ml_grid.model_classes_time_series.elasticEnsembleClassifier_module import (
    ElasticEnsemble_class,
)
from ml_grid.model_classes_time_series.rocketClassifier_module import (
    RocketClassifier_class,
)
from ml_grid.model_classes_time_series.shapeDTWClassifier_module import ShapeDTW_class
from ml_grid.pipeline.data import pipe


def get_model_class_list_ts(ml_grid_object: pipe) -> List[Any]:
    """Generates a list of instantiated time-series model classes.

    This function creates a hardcoded list of time-series model classes from
    the `aeon` library, instantiating each one with the provided `ml_grid_object`.

    Args:
        ml_grid_object (pipe): The main data pipeline object, which contains data and global parameters.

    Returns:
        List[Any]: A list of instantiated time-series model class objects.
    """

    model_class_list = [
        KNeighborsTimeSeriesClassifier_class(ml_grid_object),
        Arsenal_class(ml_grid_object),
        CNNClassifier_class(ml_grid_object),
        # ContractableBOSS_class(ml_grid_object), #univar
        InceptionTimeClassifier_class(ml_grid_object),
        # HIVECOTEV1_class(ml_grid_object), #univar
        HIVECOTEV2_class(ml_grid_object),
        FreshPRINCEClassifier_class(ml_grid_object),
        FCNClassifier_class(ml_grid_object),
        EncoderClassifier_class(ml_grid_object),
        IndividualInceptionClassifier_class(ml_grid_object),
        IndividualTDE_class(ml_grid_object),
        MLPClassifier_class(ml_grid_object),
        MUSE_class(ml_grid_object),
        OrdinalTDE_class(ml_grid_object),
        ResNetClassifier_class(ml_grid_object),
        RocketClassifier_class(ml_grid_object),
        ShapeDTW_class(ml_grid_object),
        SignatureClassifier_class(ml_grid_object),
        SummaryClassifier_class(ml_grid_object),
        TapNetClassifier_class(ml_grid_object),
        TemporalDictionaryEnsemble_class(ml_grid_object),
        TSFreshClassifier_class(ml_grid_object),
        ElasticEnsemble_class(ml_grid_object),
    ]

    return model_class_list

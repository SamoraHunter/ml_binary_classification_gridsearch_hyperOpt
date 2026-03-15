"""Generates a list of time-series model classes for the pipeline."""

import logging
from typing import Any, Dict, List, Optional

from ml_grid.model_classes_time_series.ArsenalClassifier_module import Arsenal_class
from ml_grid.model_classes_time_series.Catch22Classifer_module import (
    Catch22Classifier_class,
)
from ml_grid.model_classes_time_series.CNNClassifier_module import CNNClassifier_class
from ml_grid.model_classes_time_series.ContractableBOSSClassifier_module import (
    ContractableBOSS_class,
)
from ml_grid.model_classes_time_series.elasticEnsembleClassifier_module import (
    ElasticEnsemble_class,
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
from ml_grid.model_classes_time_series.rocketClassifier_module import (
    RocketClassifier_class,
)

# from ml_grid.model_classes_time_series.shapeDTWClassifier_module import ShapeDTW_class # deprecated
from ml_grid.model_classes_time_series.SignatureClassifier_module import (
    SignatureClassifier_class,
)
from ml_grid.model_classes_time_series.SummaryClassifier_module import (
    SummaryClassifier_class,
)

# from ml_grid.model_classes_time_series.TapNetClassifier_module import (
#     TapNetClassifier_class,
# ) # removed from aeon
from ml_grid.model_classes_time_series.TemporalDictionaryEnsembleClassifier_module import (
    TemporalDictionaryEnsemble_class,
)
from ml_grid.model_classes_time_series.TimeSeriesForestClassifier_module import (
    TimeSeriesForestClassifier_class,
)
from ml_grid.model_classes_time_series.TSFreshClassifier_module import (
    TSFreshClassifier_class,
)
from ml_grid.pipeline.data import pipe

TS_MODEL_CLASS_MAP = {
    "KNeighborsTimeSeriesClassifier": KNeighborsTimeSeriesClassifier_class,
    "TimeSeriesForestClassifier": TimeSeriesForestClassifier_class,
    "Arsenal": Arsenal_class,
    "CNNClassifier": CNNClassifier_class,
    "InceptionTimeClassifier": InceptionTimeClassifier_class,
    "HIVECOTEV2": HIVECOTEV2_class,
    "FreshPRINCEClassifier": FreshPRINCEClassifier_class,
    "FCNClassifier": FCNClassifier_class,
    "EncoderClassifier": EncoderClassifier_class,
    "IndividualInceptionClassifier": IndividualInceptionClassifier_class,
    "IndividualTDE": IndividualTDE_class,
    "MLPClassifier": MLPClassifier_class,
    "MUSE": MUSE_class,
    "OrdinalTDE": OrdinalTDE_class,
    "ResNetClassifier": ResNetClassifier_class,
    "RocketClassifier": RocketClassifier_class,
    "SignatureClassifier": SignatureClassifier_class,
    "SummaryClassifier": SummaryClassifier_class,
    "TemporalDictionaryEnsemble": TemporalDictionaryEnsemble_class,
    "TSFreshClassifier": TSFreshClassifier_class,
    "ElasticEnsemble": ElasticEnsemble_class,
    "Catch22Classifier": Catch22Classifier_class,
    # Univariate models, often disabled by default
    "ContractableBOSS": ContractableBOSS_class,
    "HIVECOTEV1": HIVECOTEV1_class,
}


def get_model_class_list_ts(ml_grid_object: pipe) -> List[Any]:
    """Generates a list of instantiated time-series model classes based on config.

    This function reads the `model_class_dict` from the `ml_grid_object` (which
    is populated from the `ts_models` section of the config file) and
    dynamically instantiates the requested model classes.

    Args:
        ml_grid_object (pipe): The main data pipeline object, which contains data and global parameters.

    Returns:
        List[Any]: A list of instantiated time-series model class objects.
    """
    logger = logging.getLogger("ml_grid")
    model_class_dict: Optional[Dict[str, bool]] = ml_grid_object.model_class_dict

    if model_class_dict is None:
        logger.warning(
            "model_class_dict is None, no time-series models will be loaded."
        )
        return []

    model_class_list = []
    for class_name, include in model_class_dict.items():
        if include:
            model_class = TS_MODEL_CLASS_MAP.get(class_name)
            if model_class:
                # All TS model classes take ml_grid_object as the only argument
                try:
                    model_instance = model_class(ml_grid_object)
                    model_class_list.append(model_instance)
                except Exception as e:
                    logger.error(
                        f"Failed to instantiate {class_name}: {e}", exc_info=True
                    )
            else:
                logger.warning(
                    f"Could not find time-series model class '{class_name}' in TS_MODEL_CLASS_MAP."
                )

    return model_class_list

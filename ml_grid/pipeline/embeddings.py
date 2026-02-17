"""Module for applying dimensionality reduction techniques (embeddings).

Designed for automated data pipelines that prepare features for binary classification.
Focuses on methods suitable for sparse, high-dimensional data with reproducible transforms.
"""

import logging
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

EmbeddingMethod = Literal[
    "svd",
    "pca",
    "nmf",
    "lda",
    "random_gaussian",
    "random_sparse",
    "select_kbest_f",
    "select_kbest_mi",
]


def create_embedding_pipeline(
    method: EmbeddingMethod = "svd",
    n_components: int = 64,
    scale: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Creates a scikit-learn pipeline for dimensionality reduction.

    This function constructs a pipeline optimized for automated preprocessing
    in classification pipelines. All methods support fit/transform pattern for
    proper train/test separation.

    Args:
        method (EmbeddingMethod): The embedding method to use:
            - "svd": TruncatedSVD - best for sparse matrices (TF-IDF, count vectors)
            - "pca": PCA - standard choice for dense data
            - "nmf": Non-negative Matrix Factorization - for non-negative sparse data
            - "lda": Linear Discriminant Analysis - supervised, maximizes class separation
            - "random_gaussian": Gaussian Random Projection - fast, preserves distances
            - "random_sparse": Sparse Random Projection - very fast for sparse data
            - "select_kbest_f": F-statistic feature selection - supervised, linear relationships
            - "select_kbest_mi": Mutual information feature selection - supervised, non-linear
            Defaults to "svd".
        n_components (int, optional): Target number of dimensions.
            Note: LDA is limited to n_classes - 1 (max 1 for binary). Defaults to 64.
        scale (bool, optional): Whether to apply StandardScaler before embedding.
            Note: Scaling converts sparse to dense - set False for sparse data. Defaults to True.
        **kwargs: Additional keyword arguments for the embedding method:
            - SVD: n_iter (int), random_state (int)
            - PCA: random_state (int), svd_solver (str)
            - NMF: init (str), max_iter (int), random_state (int)
            - Random Projection: eps (float), random_state (int)
            - SelectKBest: (no additional params typically needed)

    Returns:
        Pipeline: A scikit-learn pipeline configured with the specified steps.

    Raises:
        ValueError: If an unsupported embedding method is provided.

    Examples:
        >>> # Sparse TF-IDF data (no scaling)
        >>> pipe = create_embedding_pipeline("svd", n_components=128, scale=False)

        >>> # Dense numerical features
        >>> pipe = create_embedding_pipeline("pca", n_components=50, scale=True)

        >>> # Supervised feature selection
        >>> pipe = create_embedding_pipeline("select_kbest_f", n_components=100)

        >>> # Fast random projection for very high dims
        >>> pipe = create_embedding_pipeline("random_sparse", n_components=200,
        ...                                   scale=False, random_state=42)
    """
    steps = []

    # Add scaling if requested (converts sparse to dense!)
    if scale:
        steps.append(("scaler", StandardScaler()))

    method_lower = method.lower()

    # Unsupervised methods - work with sparse data
    if method_lower == "svd":
        default_params = {"random_state": 42}
        default_params.update(kwargs)
        steps.append(
            ("embed", TruncatedSVD(n_components=n_components, **default_params))
        )

    elif method_lower == "nmf":
        default_params = {"init": "nndsvda", "random_state": 42, "max_iter": 200}
        default_params.update(kwargs)
        steps.append(("embed", NMF(n_components=n_components, **default_params)))

    # Unsupervised methods - require dense data
    elif method_lower == "pca":
        default_params = {"random_state": 42}
        default_params.update(kwargs)
        steps.append(("embed", PCA(n_components=n_components, **default_params)))

    # Random projection methods - very fast, work with sparse
    elif method_lower == "random_gaussian":
        default_params = {"random_state": 42}
        default_params.update(kwargs)
        steps.append(
            (
                "embed",
                GaussianRandomProjection(n_components=n_components, **default_params),
            )
        )

    elif method_lower == "random_sparse":
        default_params = {"random_state": 42, "density": "auto"}
        default_params.update(kwargs)
        steps.append(
            (
                "embed",
                SparseRandomProjection(n_components=n_components, **default_params),
            )
        )

    # Supervised methods - require labels
    elif method_lower == "lda":
        logger = logging.getLogger("ml_grid")
        # LDA limited to n_classes - 1, so max 1 for binary classification
        effective_components = min(n_components, 1)
        if effective_components != n_components:
            logger.warning(
                f"LDA with binary classification limited to 1 component (requested {n_components}). Adjusting n_components to 1."
            )
        steps.append(
            (
                "embed",
                LinearDiscriminantAnalysis(n_components=effective_components, **kwargs),
            )
        )

    elif method_lower == "select_kbest_f":
        steps.append(("embed", SelectKBest(score_func=f_classif, k=n_components)))

    elif method_lower == "select_kbest_mi":
        default_params = {"random_state": 42}
        default_params.update(kwargs)

        def score_func(X, y):
            return mutual_info_classif(X, y, **default_params)

        steps.append(("embed", SelectKBest(score_func=score_func, k=n_components)))

    else:
        supported = [
            "svd",
            "pca",
            "nmf",
            "lda",
            "random_gaussian",
            "random_sparse",
            "select_kbest_f",
            "select_kbest_mi",
        ]
        raise ValueError(
            f"Unsupported embedding method: {method}. "
            f"Supported methods: {', '.join(supported)}"
        )

    return Pipeline(steps)


def apply_embedding(
    X: Union[pd.DataFrame, np.ndarray],
    pipeline: Pipeline,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
) -> np.ndarray:
    """Applies a pre-configured embedding pipeline to the data.

    Args:
        X (Union[pd.DataFrame, np.ndarray]): The input feature data.
        pipeline (Pipeline): The scikit-learn pipeline to apply.
        y (Optional[Union[pd.Series, np.ndarray]], optional): Target labels,
            required for supervised methods (lda, select_kbest_*). Defaults to None.

    Returns:
        np.ndarray: The transformed data with reduced dimensionality.

    Raises:
        ValueError: If supervised method is used without providing labels.

    Examples:
        >>> # Unsupervised
        >>> X = np.random.rand(100, 500)
        >>> pipe = create_embedding_pipeline("svd", n_components=64)
        >>> X_reduced = apply_embedding(X, pipe)

        >>> # Supervised
        >>> y = np.random.randint(0, 2, 100)
        >>> pipe = create_embedding_pipeline("lda", n_components=1)
        >>> X_reduced = apply_embedding(X, pipe, y=y)
    """
    # Check if pipeline contains supervised methods
    supervised_types = (LinearDiscriminantAnalysis, SelectKBest)
    has_supervised = any(
        isinstance(step[1], supervised_types) for step in pipeline.steps
    )

    if has_supervised and y is None:
        raise ValueError(
            "Supervised methods (lda, select_kbest_*) require target labels. "
            "Please provide the y parameter."
        )

    if y is not None:
        # Pass labels to the embedding step
        return pipeline.fit_transform(X, y)
    else:
        return pipeline.fit_transform(X)


def transform_new_data(
    X: Union[pd.DataFrame, np.ndarray], fitted_pipeline: Pipeline
) -> np.ndarray:
    """Transforms new data using an already-fitted pipeline.

    Critical for proper train/test separation in production pipelines.

    Args:
        X (Union[pd.DataFrame, np.ndarray]): New data to transform.
        fitted_pipeline (Pipeline): A pipeline that has already been fitted.

    Returns:
        np.ndarray: The transformed data.

    Examples:
        >>> # Fit on training data
        >>> pipe = create_embedding_pipeline("svd", n_components=64)
        >>> X_train_reduced = apply_embedding(X_train, pipe)

        >>> # Transform test data with same fitted pipeline
        >>> X_test_reduced = transform_new_data(X_test, pipe)
    """
    return fitted_pipeline.transform(X)


def get_method_recommendation(
    is_sparse: bool,
    has_labels: bool,
    n_features: int,
    n_samples: int,
    is_nonnegative: bool = False,
) -> Dict[str, Any]:
    """Recommends the best embedding method based on data characteristics.

    Args:
        is_sparse (bool): Whether the data is sparse (e.g., TF-IDF, one-hot encoded).
        has_labels (bool): Whether labels are available for supervised methods.
        n_features (int): Number of input features.
        n_samples (int): Number of samples.
        is_nonnegative (bool): Whether all data values are non-negative.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - method: Recommended method name
            - scale: Whether to apply scaling
            - rationale: Explanation of recommendation
            - alternatives: List of other suitable methods

    Examples:
        >>> # Sparse TF-IDF data for classification
        >>> rec = get_method_recommendation(
        ...     is_sparse=True, has_labels=True,
        ...     n_features=10000, n_samples=5000
        ... )
        >>> print(rec['method'])
        'svd'
    """
    recommendations = []

    # Sparse data considerations
    if is_sparse:
        if is_nonnegative:
            recommendations.append(
                {
                    "method": "nmf",
                    "scale": False,
                    "rationale": "NMF works well with sparse, non-negative data and provides interpretable components",
                    "alternatives": ["svd", "random_sparse"],
                }
            )
        else:
            recommendations.append(
                {
                    "method": "svd",
                    "scale": False,
                    "rationale": "TruncatedSVD is optimized for sparse matrices and doesn't require densification",
                    "alternatives": (
                        ["random_sparse", "nmf"]
                        if is_nonnegative
                        else ["random_sparse"]
                    ),
                }
            )

    # Dense data considerations
    else:
        if has_labels and n_samples > n_features:
            recommendations.append(
                {
                    "method": "select_kbest_f",
                    "scale": True,
                    "rationale": "Feature selection with F-statistic is fast and effective for classification with labels",
                    "alternatives": ["pca", "lda", "select_kbest_mi"],
                }
            )
        else:
            recommendations.append(
                {
                    "method": "pca",
                    "scale": True,
                    "rationale": "PCA is the standard choice for dense numerical data",
                    "alternatives": ["random_gaussian"],
                }
            )

    # Very high dimensional data
    if n_features > 10000:
        recommendations.append(
            {
                "method": "random_sparse" if is_sparse else "random_gaussian",
                "scale": False,
                "rationale": "Random projection is very fast for extremely high-dimensional data",
                "alternatives": ["svd"] if is_sparse else ["pca"],
            }
        )

    # Return the first (primary) recommendation
    return (
        recommendations[0]
        if recommendations
        else {
            "method": "svd",
            "scale": False,
            "rationale": "Default choice for general use",
            "alternatives": ["pca"],
        }
    )


def get_explained_variance(
    pipeline: Pipeline, X: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> Optional[np.ndarray]:
    """Extracts explained variance information from fitted pipeline if available.

    Only works with methods that have explained_variance_ratio_ attribute (PCA, SVD).

    Args:
        pipeline (Pipeline): A fitted pipeline.
        X (Optional[Union[pd.DataFrame, np.ndarray]]): Data to compute variance on
            if not already fitted.

    Returns:
        Optional[np.ndarray]: Array of explained variance ratios, or None if not applicable.

    Examples:
        >>> pipe = create_embedding_pipeline("pca", n_components=10)
        >>> X_reduced = apply_embedding(X, pipe)
        >>> variance = get_explained_variance(pipe)
        >>> print(f"Total variance explained: {variance.sum():.2%}")
    """
    if X is not None and not hasattr(pipeline, "named_steps"):
        pipeline.fit(X)

    embed_step = pipeline.named_steps.get("embed")

    if hasattr(embed_step, "explained_variance_ratio_"):
        return embed_step.explained_variance_ratio_

    return None


def recommend_n_components(
    pipeline: Pipeline,
    X: Union[pd.DataFrame, np.ndarray],
    variance_threshold: float = 0.95,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Optional[int]:
    """Recommends number of components to retain a target variance level.

    Only works with methods that provide explained variance (PCA, SVD).

    Args:
        pipeline (Pipeline): A pipeline with a variance-based method.
        X (Union[pd.DataFrame, np.ndarray]): The data to analyze.
        variance_threshold (float): Target cumulative variance to retain (0-1).
        y (Optional[Union[pd.Series, np.ndarray]]): Labels if needed.

    Returns:
        Optional[int]: Recommended number of components, or None if not applicable.

    Examples:
        >>> # Fit with high n_components first
        >>> pipe = create_embedding_pipeline("pca", n_components=100)
        >>> apply_embedding(X, pipe)
        >>> n_opt = recommend_n_components(pipe, X, variance_threshold=0.95)
        >>> print(f"Use {n_opt} components for 95% variance")
    """
    variance_ratios = get_explained_variance(pipeline, X)

    if variance_ratios is None:
        return None

    cumsum = np.cumsum(variance_ratios)
    n_components = np.argmax(cumsum >= variance_threshold) + 1

    return int(n_components)

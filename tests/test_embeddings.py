"""Unit and integration tests for the embedding module."""

import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from ml_grid.pipeline.embeddings import (
    create_embedding_pipeline,
    apply_embedding,
    transform_new_data,
    get_method_recommendation,
    get_explained_variance,
    recommend_n_components,
)
from ml_grid.pipeline.data import pipe


class TestEmbeddings(unittest.TestCase):
    """Test suite for embedding functionality."""

    def setUp(self):
        """Set up common test data and configurations."""
        np.random.seed(42)
        self.X_dense = np.random.rand(100, 200)
        self.X_df = pd.DataFrame(self.X_dense)
        self.X_sparse = sparse.random(
            100, 200, density=0.1, format="csr", random_state=42
        )
        self.y_binary = np.random.randint(0, 2, 100)
        self.n_components = 32

    def test_create_embedding_pipeline_pca(self):
        """Test creating a PCA embedding pipeline with and without scaling."""
        # Test with scaling
        pipeline = create_embedding_pipeline(
            method="pca", n_components=self.n_components, scale=True
        )
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline.steps), 2)
        self.assertIsInstance(pipeline.steps[0][1], StandardScaler)
        self.assertIsInstance(pipeline.steps[1][1], PCA)
        self.assertEqual(pipeline.steps[1][1].n_components, self.n_components)

        # Test without scaling
        pipeline_no_scale = create_embedding_pipeline(
            method="pca", n_components=64, scale=False
        )
        self.assertEqual(len(pipeline_no_scale.steps), 1)
        self.assertIsInstance(pipeline_no_scale.steps[0][1], PCA)
        self.assertEqual(pipeline_no_scale.steps[0][1].n_components, 64)

    def test_create_embedding_pipeline_svd(self):
        """Test creating an SVD embedding pipeline."""
        pipeline = create_embedding_pipeline(
            method="svd", n_components=self.n_components, scale=False
        )
        self.assertIsInstance(pipeline, Pipeline)
        # Should only have embed step (no scaling for sparse data)
        self.assertEqual(len(pipeline.steps), 1)
        self.assertIsInstance(pipeline.steps[0][1], TruncatedSVD)
        self.assertEqual(pipeline.steps[0][1].n_components, self.n_components)

    def test_create_embedding_pipeline_nmf(self):
        """Test creating an NMF embedding pipeline."""
        pipeline = create_embedding_pipeline(
            method="nmf", n_components=self.n_components, scale=False
        )
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(pipeline.steps[0][1], NMF)

    def test_create_embedding_pipeline_lda(self):
        """Test creating an LDA embedding pipeline for binary classification."""
        # LDA should be limited to 1 component for binary classification
        pipeline = create_embedding_pipeline(method="lda", n_components=10, scale=True)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(pipeline.steps[1][1], LinearDiscriminantAnalysis)
        # Check that it's limited to 1 component
        self.assertEqual(pipeline.steps[1][1].n_components, 1)

    def test_create_embedding_pipeline_random_projection(self):
        """Test creating random projection pipelines."""
        # Gaussian random projection
        pipeline_gaussian = create_embedding_pipeline(
            method="random_gaussian", n_components=self.n_components, scale=False
        )
        self.assertIsInstance(pipeline_gaussian.steps[0][1], GaussianRandomProjection)

        # Sparse random projection
        pipeline_sparse = create_embedding_pipeline(
            method="random_sparse", n_components=self.n_components, scale=False
        )
        self.assertIsInstance(pipeline_sparse.steps[0][1], SparseRandomProjection)

    def test_create_embedding_pipeline_feature_selection(self):
        """Test creating feature selection pipelines."""
        # F-statistic feature selection
        pipeline_f = create_embedding_pipeline(
            method="select_kbest_f", n_components=self.n_components, scale=True
        )
        self.assertIsInstance(pipeline_f.steps[1][1], SelectKBest)

        # Mutual information feature selection
        pipeline_mi = create_embedding_pipeline(
            method="select_kbest_mi", n_components=self.n_components, scale=True
        )
        self.assertIsInstance(pipeline_mi.steps[1][1], SelectKBest)

    def test_create_embedding_pipeline_invalid_method(self):
        """Test that an invalid method raises a ValueError."""
        with self.assertRaises(ValueError):
            create_embedding_pipeline(method="invalid_method")

    def test_apply_embedding_shape_unsupervised(self):
        """
        Test that apply_embedding returns the correct output shape for
        unsupervised methods.
        """
        pipeline = create_embedding_pipeline(
            method="svd", n_components=self.n_components
        )

        # Test with numpy array
        X_embedded_np = apply_embedding(self.X_dense, pipeline)
        self.assertEqual(X_embedded_np.shape, (100, self.n_components))

        # Test with pandas DataFrame
        X_embedded_df = apply_embedding(self.X_df, pipeline)
        self.assertEqual(X_embedded_df.shape, (100, self.n_components))

    def test_apply_embedding_shape_supervised(self):
        """Test that apply_embedding works correctly with supervised methods."""
        # LDA (limited to 1 component for binary)
        pipeline_lda = create_embedding_pipeline(
            method="lda", n_components=5, scale=True
        )
        X_embedded_lda = apply_embedding(self.X_dense, pipeline_lda, y=self.y_binary)
        self.assertEqual(X_embedded_lda.shape, (100, 1))

        # Feature selection
        pipeline_kbest = create_embedding_pipeline(
            method="select_kbest_f", n_components=self.n_components, scale=True
        )
        X_embedded_kbest = apply_embedding(
            self.X_dense, pipeline_kbest, y=self.y_binary
        )
        self.assertEqual(X_embedded_kbest.shape, (100, self.n_components))

    def test_apply_embedding_supervised_without_labels_raises_error(self):
        """Test that supervised methods raise error when labels are not provided."""
        pipeline = create_embedding_pipeline(method="lda", n_components=1, scale=True)

        with self.assertRaises(ValueError) as context:
            apply_embedding(self.X_dense, pipeline)

        self.assertIn("Supervised methods", str(context.exception))
        self.assertIn("require target labels", str(context.exception))

    def test_apply_embedding_sparse_data(self):
        """Test embedding with sparse data (common for text features)."""
        pipeline = create_embedding_pipeline(
            method="svd", n_components=self.n_components, scale=False
        )

        X_embedded = apply_embedding(self.X_sparse, pipeline)
        self.assertEqual(X_embedded.shape, (100, self.n_components))
        # Output should be dense
        self.assertIsInstance(X_embedded, np.ndarray)

    def test_transform_new_data(self):
        """Test transforming new data with a fitted pipeline."""
        # Fit on training data
        pipeline = create_embedding_pipeline(
            method="pca", n_components=self.n_components, scale=False
        )
        apply_embedding(self.X_dense[:70], pipeline)

        # Transform test data
        X_test = self.X_dense[70:]
        X_test_embedded = transform_new_data(X_test, pipeline)

        self.assertEqual(X_test_embedded.shape, (30, self.n_components))
        # Check that transform doesn't change the pipeline
        self.assertTrue(hasattr(pipeline.named_steps["embed"], "components_"))

    def test_get_method_recommendation_sparse(self):
        """Test method recommendation for sparse data."""
        rec = get_method_recommendation(
            is_sparse=True,
            has_labels=False,
            n_features=5000,
            n_samples=1000,
            is_nonnegative=False,
        )

        self.assertEqual(rec["method"], "svd")
        self.assertFalse(rec["scale"])
        self.assertIn("rationale", rec)
        self.assertIn("alternatives", rec)

    def test_get_method_recommendation_dense_supervised(self):
        """Test method recommendation for dense data with labels."""
        rec = get_method_recommendation(
            is_sparse=False,
            has_labels=True,
            n_features=200,
            n_samples=1000,
            is_nonnegative=False,
        )

        self.assertEqual(rec["method"], "select_kbest_f")
        self.assertTrue(rec["scale"])

    def test_get_method_recommendation_high_dimensional(self):
        """Test method recommendation for very high-dimensional data."""
        rec = get_method_recommendation(
            is_sparse=True,
            has_labels=False,
            n_features=50000,
            n_samples=1000,
            is_nonnegative=False,
        )

        self.assertIn(rec["method"], ["random_sparse", "svd"])

    def test_get_method_recommendation_nonnegative(self):
        """Test method recommendation for non-negative sparse data."""
        rec = get_method_recommendation(
            is_sparse=True,
            has_labels=False,
            n_features=5000,
            n_samples=1000,
            is_nonnegative=True,
        )

        self.assertEqual(rec["method"], "nmf")
        self.assertFalse(rec["scale"])

    def test_get_explained_variance_pca(self):
        """Test extracting explained variance from PCA."""
        pipeline = create_embedding_pipeline(method="pca", n_components=10)
        apply_embedding(self.X_dense, pipeline)

        variance = get_explained_variance(pipeline)
        self.assertIsNotNone(variance)
        self.assertEqual(len(variance), 10)
        self.assertTrue(np.all(variance >= 0))
        self.assertTrue(np.all(variance <= 1))

    def test_get_explained_variance_svd(self):
        """Test extracting explained variance from SVD."""
        pipeline = create_embedding_pipeline(method="svd", n_components=10, scale=False)
        apply_embedding(self.X_sparse, pipeline)

        variance = get_explained_variance(pipeline)
        self.assertIsNotNone(variance)
        self.assertEqual(len(variance), 10)

    def test_get_explained_variance_not_applicable(self):
        """Test that methods without variance return None."""
        pipeline = create_embedding_pipeline(method="random_gaussian", n_components=10)
        apply_embedding(self.X_dense, pipeline)

        variance = get_explained_variance(pipeline)
        self.assertIsNone(variance)

    def test_recommend_n_components(self):
        """Test automatic recommendation of n_components."""
        pipeline = create_embedding_pipeline(method="pca", n_components=50)
        apply_embedding(self.X_dense, pipeline)

        n_recommended = recommend_n_components(
            pipeline, self.X_dense, variance_threshold=0.90
        )

        self.assertIsNotNone(n_recommended)
        self.assertIsInstance(n_recommended, int)
        self.assertGreater(n_recommended, 0)
        self.assertLessEqual(n_recommended, 50)

    def test_recommend_n_components_not_applicable(self):
        """Test that methods without variance return None for recommendation."""
        pipeline = create_embedding_pipeline(method="random_gaussian", n_components=10)
        apply_embedding(self.X_dense, pipeline)

        n_recommended = recommend_n_components(pipeline, self.X_dense)
        self.assertIsNone(n_recommended)

    def test_embedding_with_kwargs(self):
        """Test passing additional kwargs to embedding methods."""
        # Test SVD with custom parameters
        pipeline = create_embedding_pipeline(
            method="svd", n_components=10, scale=False, n_iter=10, random_state=123
        )
        self.assertEqual(pipeline.steps[0][1].n_iter, 10)
        self.assertEqual(pipeline.steps[0][1].random_state, 123)

        # Test NMF with custom parameters
        pipeline_nmf = create_embedding_pipeline(
            method="nmf", n_components=10, scale=False, max_iter=100, init="random"
        )
        self.assertEqual(pipeline_nmf.steps[0][1].max_iter, 100)
        self.assertEqual(pipeline_nmf.steps[0][1].init, "random")

    def test_embedding_integration_in_pipe(self):
        """Test the embedding step within the main data pipeline (`pipe`)."""
        # Create a dummy dataset for the pipe class
        data = pd.DataFrame(np.random.rand(50, 25))
        data.columns = [f"feature_{i}" for i in range(25)]
        data["outcome_var_1"] = np.random.randint(0, 2, 50)
        data.to_csv("test_data_for_embedding.csv", index=False)

        embedding_dim = 8

        # Parameters to enable embedding
        local_param_dict = {
            "use_embedding": True,
            "embedding_method": "pca",
            "embedding_dim": embedding_dim,
            "scale_features_before_embedding": True,
            "outcome_var_n": "1",
            "scale": False,
            "feature_n": None,  # Set to None to explicitly disable feature selection
            "corr": 1.0,
            "percent_missing": 100,
            "data": {col: True for col in data.columns if "feature" in col},
            "param_space_size": "xsmall",
        }

        # Instantiate the pipeline with embeddings enabled
        ml_grid_object = pipe(
            file_name="test_data_for_embedding.csv",
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=".",
            param_space_index=0,
            experiment_dir=".",
        )

        # The final feature matrix X should have a number of columns
        # equal to the embedding dimension.
        final_feature_count = ml_grid_object.X_train.shape[1]
        self.assertEqual(final_feature_count, embedding_dim)

        # Check if column names are correct
        expected_cols = [f"embed_{i}" for i in range(embedding_dim)]
        self.assertListEqual(list(ml_grid_object.X_train.columns), expected_cols)

    def test_embedding_integration_supervised_methods(self):
        """Test integration with supervised embedding methods."""
        # Create a dummy dataset
        data = pd.DataFrame(np.random.rand(100, 25))
        data.columns = [f"feature_{i}" for i in range(25)]
        data["outcome_var_1"] = np.random.randint(0, 2, 100)
        data.to_csv("test_data_for_embedding_supervised.csv", index=False)

        # Test with feature selection
        local_param_dict = {
            "use_embedding": True,
            "embedding_method": "select_kbest_f",
            "embedding_dim": 15,
            "scale_features_before_embedding": True,
            "outcome_var_n": "1",
            "scale": False,
            "feature_n": None,
            "corr": 1.0,
            "percent_missing": 100,
            "data": {col: True for col in data.columns if "feature" in col},
            "param_space_size": "xsmall",
        }

        ml_grid_object = pipe(
            file_name="test_data_for_embedding_supervised.csv",
            drop_term_list=[],
            local_param_dict=local_param_dict,
            base_project_dir=".",
            param_space_index=0,
            experiment_dir=".",
        )

        self.assertEqual(ml_grid_object.X_train.shape[1], 15)


if __name__ == "__main__":
    unittest.main()

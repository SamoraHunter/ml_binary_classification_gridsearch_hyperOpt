# plot_global_importance.py
"""
Global importance analysis plotting module for ML results analysis.
This module trains a meta-model on the experimental parameters to determine
which settings have the most significant impact on the target metric.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List
import warnings
import ast
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_grid.results_processing.core import get_clean_data

class GlobalImportancePlotter:
    """
    Analyzes the entire experimental space to determine which parameters
    (feature categories, pipeline settings, algorithms) have the most
    impact on the target metric.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the plotter.

        Args:
            data: Results DataFrame, must contain columns for experimental parameters and performance metrics.
        """
        self.data = data
        self.clean_data = get_clean_data(data)
        self.logger = logging.getLogger('ml_grid')
        
        # Define all potential predictors from other plotters
        self.feature_categories = [
            'age', 'sex', 'bmi', 'ethnicity', 'bloods', 'diagnostic_order', 'drug_order',
            'annotation_n', 'meta_sp_annotation_n', 'meta_sp_annotation_mrc_n',
            'annotation_mrc_n', 'core_02', 'bed', 'vte_status', 'hosp_site',
            'core_resus', 'news', 'date_time_stamp'
        ]
        self.pipeline_categorical_params = ['resample', 'scale', 'param_space_size', 'percent_missing']
        self.pipeline_continuous_params = [
            'nb_size', 'X_train_size', 'X_test_orig_size', 'X_test_size', 
            'n_fits', 't_fits', 'run_time'
        ]
        self.algorithm_col = 'method_name'

        plt.style.use('default')
        sns.set_palette("viridis")

    def _prepare_data_for_importance_analysis(
        self, metric: str = 'auc'
    ) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, List[str], List[str]]:
        """Prepares the data for modeling by selecting features, handling types,
        and setting up a preprocessing pipeline.

        Args:
            metric (str, optional): The target metric to predict.
                Defaults to 'auc'.

        Returns:
            Tuple[pd.DataFrame, pd.Series, ColumnTransformer, List[str], List[str]]:
            A tuple containing the feature DataFrame (X), the target series (y),
            the configured preprocessor, a list of numeric feature names, and a
            list of categorical feature names.
        """
        all_predictors = (self.feature_categories + 
                          self.pipeline_categorical_params + 
                          self.pipeline_continuous_params + 
                          [self.algorithm_col])
        
        # Filter to columns that actually exist in the data
        available_predictors = [p for p in all_predictors if p in self.clean_data.columns]
        
        if not available_predictors:
            raise ValueError("No predictor columns found for global importance analysis.")

        # Drop rows with missing target metric
        analysis_df = self.clean_data[available_predictors + [metric]].dropna(subset=[metric]).copy()
        
        y = analysis_df[metric]
        X = analysis_df.drop(columns=[metric])

        # Convert boolean-like string columns to integer
        for col in self.feature_categories + ['scale']:
            if col in X.columns and X[col].dtype != bool and X[col].dtype != int:
                try:
                    # Handle 'True'/'False' strings
                    if X[col].apply(type).eq(str).all():
                         X[col] = X[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                    X[col] = X[col].astype(int)
                except Exception:
                    warnings.warn(f"Could not convert column '{col}' to int. It will be treated as categorical.", stacklevel=2)

        # Identify column types for the preprocessor
        numeric_features = [p for p in self.pipeline_continuous_params if p in X.columns]
        
        categorical_features = [p for p in self.pipeline_categorical_params if p in X.columns and p != 'scale']
        categorical_features.append(self.algorithm_col)
        
        # Add any feature categories that failed conversion to int
        for col in self.feature_categories:
            if col in X.columns and X[col].dtype not in [int, float, bool]:
                categorical_features.append(col)

        # Binary features are numeric (0/1) and don't need special handling beyond imputation
        binary_features = [p for p in self.feature_categories if p in X.columns and p not in categorical_features]
        if 'scale' in X.columns and 'scale' not in categorical_features:
            binary_features.append('scale')
            
        numeric_features.extend(binary_features)
        
        # Remove duplicates
        numeric_features = sorted(list(set(numeric_features)))
        categorical_features = sorted(list(set(categorical_features)))

        # Create preprocessing pipelines for numeric and categorical features
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

        # Create a column transformer to apply different transformations to different columns
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
        
        return X, y, preprocessor, numeric_features, categorical_features

    def plot_global_importance(
        self,
        metric: str = 'auc',
        top_n: int = 30,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Trains a model to predict a metric from experimental parameters and plots importances.

        This method trains a RandomForestRegressor on the various pipeline and
        algorithm parameters to predict the outcome of a given performance metric.
        parameters and plots the resulting feature importances.

        Args:
            metric (str, optional): The target metric to predict. Defaults to 'auc'.
            top_n (int, optional): The number of top important features to plot.
                Defaults to 30.
            figsize (Tuple[int, int], optional): The figure size for the plot.
                Defaults to (12, 10).
        """
        self.logger.info(f"Running Global Importance Analysis for metric: {metric.upper()}")
        try:
            X, y, preprocessor, numeric_features, categorical_features = self._prepare_data_for_importance_analysis(metric)
        except ValueError as e:
            self.logger.error(f"Could not prepare data for global importance analysis: {e}")
            return

        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
        model_pipeline.fit(X, y)

        importances = model_pipeline.named_steps['regressor'].feature_importances_
        
        try:
            ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        except AttributeError: # older sklearn compatibility
            ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_features)

        all_feature_names = list(numeric_features) + list(ohe_feature_names)

        importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=figsize)
        ax = sns.barplot(x='importance', y='feature', data=importance_df, orient='h', palette='plasma', hue='feature', legend=False)
        
        ax.set_title(f'Global Parameter Importance for Predicting {metric.upper()}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance (Gini Impurity Reduction)', fontsize=12)
        ax.set_ylabel('Experimental Parameter', fontsize=12)
        
        ax.bar_label(ax.containers[0], fmt='%.4f', padding=3, fontsize=9)
            
        plt.tight_layout()
        plt.show()
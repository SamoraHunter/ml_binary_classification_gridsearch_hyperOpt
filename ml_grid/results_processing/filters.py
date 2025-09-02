# filters.py
"""
Data filtering and querying module for ML results analysis.
Provides flexible filtering capabilities with outcome variable stratification.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from core import get_clean_data


class ResultsFilter:
    """
    A class for filtering and querying ML results data with outcome stratification support.
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with results data."""
        self.data = data
        self.original_data = data.copy()
    
    def reset_data(self):
        """Reset data to original state."""
        self.data = self.original_data.copy()
    
    def filter_by_algorithm(self, algorithms: Union[str, List[str]]) -> pd.DataFrame:
        """Filter data by specific algorithms."""
        if isinstance(algorithms, str):
            algorithms = [algorithms]
        
        filtered_data = self.data[self.data['method_name'].isin(algorithms)]
        return filtered_data
    
    def filter_by_outcome(self, outcomes: Union[str, List[str]]) -> pd.DataFrame:
        """Filter data by specific outcome variables."""
        if 'outcome_variable' not in self.data.columns:
            raise ValueError("outcome_variable column not found in data")
        
        if isinstance(outcomes, str):
            outcomes = [outcomes]
        
        filtered_data = self.data[self.data['outcome_variable'].isin(outcomes)]
        return filtered_data
    
    def filter_by_metric_threshold(self, metric: str, threshold: float, above: bool = True) -> pd.DataFrame:
        """Filter data based on metric threshold."""
        if metric not in self.data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        if above:
            filtered_data = self.data[self.data[metric] >= threshold]
        else:
            filtered_data = self.data[self.data[metric] <= threshold]
        
        return filtered_data
    
    def filter_by_run_timestamp(self, timestamps: Union[str, List[str]]) -> pd.DataFrame:
        """Filter data by specific run timestamps."""
        if isinstance(timestamps, str):
            timestamps = [timestamps]
        
        filtered_data = self.data[self.data['run_timestamp'].isin(timestamps)]
        return filtered_data
    
    def filter_successful_runs(self) -> pd.DataFrame:
        """Filter out failed runs."""
        return get_clean_data(self.data, remove_failed=True)
    
    def filter_by_feature_count(self, min_features: int = None, max_features: int = None) -> pd.DataFrame:
        """Filter by number of features used."""
        if 'n_features' not in self.data.columns:
            raise ValueError("n_features column not found in data")
        
        filtered_data = self.data.copy()
        
        if min_features is not None:
            filtered_data = filtered_data[filtered_data['n_features'] >= min_features]
        
        if max_features is not None:
            filtered_data = filtered_data[filtered_data['n_features'] <= max_features]
        
        return filtered_data
    
    def filter_by_sample_size(self, min_train_size: int = None, max_train_size: int = None) -> pd.DataFrame:
        """Filter by training sample size."""
        if 'X_train_size' not in self.data.columns:
            raise ValueError("X_train_size column not found in data")
        
        filtered_data = self.data.copy()
        
        if min_train_size is not None:
            filtered_data = filtered_data[filtered_data['X_train_size'] >= min_train_size]
        
        if max_train_size is not None:
            filtered_data = filtered_data[filtered_data['X_train_size'] <= max_train_size]
        
        return filtered_data
    
    def get_top_performers(self, metric: str = 'auc', n: int = 10, 
                          stratify_by_outcome: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get top N performing configurations.
        
        Args:
            metric: Performance metric to rank by
            n: Number of top performers to return
            stratify_by_outcome: If True, return top performers for each outcome variable
        
        Returns:
            DataFrame of top performers, or dict of DataFrames if stratified
        """
        successful_data = self.filter_successful_runs()
        
        if not stratify_by_outcome:
            return successful_data.nlargest(n, metric)
        
        # Stratify by outcome variable
        if 'outcome_variable' not in successful_data.columns:
            raise ValueError("outcome_variable column not found for stratification")
        
        top_performers_by_outcome = {}
        
        for outcome in successful_data['outcome_variable'].unique():
            outcome_data = successful_data[successful_data['outcome_variable'] == outcome]
            if len(outcome_data) > 0:
                top_performers_by_outcome[outcome] = outcome_data.nlargest(n, metric)
        
        return top_performers_by_outcome
    
    def get_algorithm_performance_summary(self, metric: str = 'auc', 
                                        stratify_by_outcome: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get performance summary by algorithm.
        
        Args:
            metric: Metric to summarize
            stratify_by_outcome: If True, provide summary for each outcome variable
        
        Returns:
            Summary DataFrame or dict of summary DataFrames if stratified
        """
        successful_data = self.filter_successful_runs()
        
        def _calculate_summary(data):
            return data.groupby('method_name')[metric].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4).sort_values('mean', ascending=False)
        
        if not stratify_by_outcome:
            return _calculate_summary(successful_data)
        
        # Stratify by outcome variable
        if 'outcome_variable' not in successful_data.columns:
            raise ValueError("outcome_variable column not found for stratification")
        
        summaries_by_outcome = {}
        
        for outcome in successful_data['outcome_variable'].unique():
            outcome_data = successful_data[successful_data['outcome_variable'] == outcome]
            if len(outcome_data) > 0:
                summaries_by_outcome[outcome] = _calculate_summary(outcome_data)
        
        return summaries_by_outcome
    
    def find_feature_usage_patterns(self, min_frequency: int = 5, 
                                   stratify_by_outcome: bool = False) -> Union[Dict[str, int], Dict[str, Dict[str, int]]]:
        """
        Find common feature usage patterns.
        Requires decoded_features column.
        
        Args:
            min_frequency: Minimum frequency for a feature to be included
            stratify_by_outcome: If True, provide patterns for each outcome variable
        
        Returns:
            Dictionary of feature counts, or nested dict if stratified
        """
        if 'decoded_features' not in self.data.columns:
            raise ValueError("decoded_features column not found")
        
        def _get_feature_counts(data):
            feature_counts = {}
            
            for features_list in data['decoded_features']:
                if isinstance(features_list, list):
                    for feature in features_list:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            # Filter by minimum frequency
            return {feature: count for feature, count in feature_counts.items() 
                    if count >= min_frequency}
        
        if not stratify_by_outcome:
            return _get_feature_counts(self.data)
        
        # Stratify by outcome variable
        if 'outcome_variable' not in self.data.columns:
            raise ValueError("outcome_variable column not found for stratification")
        
        patterns_by_outcome = {}
        
        for outcome in self.data['outcome_variable'].unique():
            outcome_data = self.data[self.data['outcome_variable'] == outcome]
            if len(outcome_data) > 0:
                patterns_by_outcome[outcome] = _get_feature_counts(outcome_data)
        
        return patterns_by_outcome
    
    def compare_algorithms_across_outcomes(self, algorithms: List[str] = None, 
                                         metric: str = 'auc') -> pd.DataFrame:
        """
        Compare algorithm performance across different outcome variables.
        
        Args:
            algorithms: List of algorithms to compare (None for all)
            metric: Performance metric to compare
        
        Returns:
            Pivot table with algorithms as rows and outcomes as columns
        """
        if 'outcome_variable' not in self.data.columns:
            raise ValueError("outcome_variable column not found")
        
        successful_data = self.filter_successful_runs()
        
        if algorithms is not None:
            successful_data = successful_data[
                successful_data['method_name'].isin(algorithms)
            ]
        
        # Create pivot table
        comparison_table = successful_data.pivot_table(
            values=metric,
            index='method_name',
            columns='outcome_variable',
            aggfunc='mean'
        ).round(4)
        
        return comparison_table
    
    def get_outcome_difficulty_ranking(self, metric: str = 'auc') -> pd.DataFrame:
        """
        Rank outcome variables by difficulty (based on average performance).
        
        Args:
            metric: Performance metric to use for ranking
        
        Returns:
            DataFrame with outcomes ranked by difficulty
        """
        if 'outcome_variable' not in self.data.columns:
            raise ValueError("outcome_variable column not found")
        
        successful_data = self.filter_successful_runs()
        
        difficulty_ranking = successful_data.groupby('outcome_variable')[metric].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        # Sort by mean performance (ascending for difficulty, assuming higher metric = better)
        difficulty_ranking = difficulty_ranking.sort_values('mean', ascending=True)
        difficulty_ranking.reset_index(inplace=True)
        difficulty_ranking['difficulty_rank'] = range(1, len(difficulty_ranking) + 1)
        
        return difficulty_ranking
    
    def filter_by_cross_outcome_performance(self, metric: str = 'auc', 
                                          min_outcomes: int = 2,
                                          percentile_threshold: float = 75) -> pd.DataFrame:
        """
        Find algorithms/configurations that perform well across multiple outcomes.
        
        Args:
            metric: Performance metric to evaluate
            min_outcomes: Minimum number of outcomes an algorithm must be tested on
            percentile_threshold: Percentile threshold for "good" performance
        
        Returns:
            DataFrame of configurations that perform well across outcomes
        """
        if 'outcome_variable' not in self.data.columns:
            raise ValueError("outcome_variable column not found")
        
        successful_data = self.filter_successful_runs()
        
        # Calculate percentile threshold for the metric
        threshold_value = np.percentile(successful_data[metric].dropna(), percentile_threshold)
        
        # Find configurations that meet threshold across multiple outcomes
        good_performers = successful_data[successful_data[metric] >= threshold_value]
        
        # Group by algorithm and parameter combination (if available)
        grouping_cols = ['method_name']
        if 'parameter_sample' in successful_data.columns:
            grouping_cols.append('parameter_sample')
        
        cross_outcome_performers = good_performers.groupby(grouping_cols).agg({
            'outcome_variable': 'nunique',
            metric: ['count', 'mean', 'std']
        }).round(4)
        
        # Flatten column names
        cross_outcome_performers.columns = [
            '_'.join(col).strip() if col[1] else col[0] 
            for col in cross_outcome_performers.columns.values
        ]
        
        # Filter by minimum outcomes requirement
        cross_outcome_performers = cross_outcome_performers[
            cross_outcome_performers['outcome_variable_nunique'] >= min_outcomes
        ]
        
        # Sort by number of outcomes and then by mean performance
        cross_outcome_performers = cross_outcome_performers.sort_values([
            'outcome_variable_nunique', f'{metric}_mean'
        ], ascending=[False, False])
        
        return cross_outcome_performers.reset_index()


class OutcomeComparator:
    """Specialized class for comparing outcomes and their characteristics."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with results data."""
        if 'outcome_variable' not in data.columns:
            raise ValueError("outcome_variable column required for outcome comparison")
        
        self.data = data
    
    def get_outcome_characteristics(self) -> pd.DataFrame:
        """
        Get characteristics of each outcome variable based on available metadata.
        """
        characteristics = []
        
        for outcome in self.data['outcome_variable'].unique():
            outcome_data = self.data[self.data['outcome_variable'] == outcome]
            
            char_dict = {'outcome_variable': outcome}
            
            # Basic counts
            char_dict['total_experiments'] = len(outcome_data)
            char_dict['successful_experiments'] = len(outcome_data[outcome_data['failed'] == 0])
            char_dict['success_rate'] = char_dict['successful_experiments'] / char_dict['total_experiments']
            
            # Sample characteristics (take mode/median)
            sample_chars = ['n_unique_out', 'outcome_var_n', 'percent_missing']
            for char in sample_chars:
                if char in outcome_data.columns:
                    char_dict[char] = outcome_data[char].median()
            
            # Performance characteristics
            successful_data = outcome_data[outcome_data['failed'] == 0]
            if len(successful_data) > 0:
                for metric in ['auc', 'f1', 'precision', 'recall', 'accuracy']:
                    if metric in successful_data.columns:
                        char_dict[f'{metric}_mean'] = successful_data[metric].mean()
                        char_dict[f'{metric}_std'] = successful_data[metric].std()
            
            characteristics.append(char_dict)
        
        return pd.DataFrame(characteristics).round(4)
    
    def find_similar_outcomes(self, reference_outcome: str, 
                            similarity_metrics: List[str] = None) -> pd.DataFrame:
        """
        Find outcomes similar to a reference outcome based on performance patterns.
        
        Args:
            reference_outcome: Reference outcome to compare against
            similarity_metrics: Metrics to use for similarity (default: ['auc', 'f1'])
        
        Returns:
            DataFrame with similarity scores
        """
        if similarity_metrics is None:
            similarity_metrics = ['auc', 'f1']
        
        # Get successful runs only
        successful_data = self.data[self.data['failed'] == 0]
        
        # Get reference outcome performance
        ref_data = successful_data[successful_data['outcome_variable'] == reference_outcome]
        if len(ref_data) == 0:
            raise ValueError(f"No successful data found for reference outcome: {reference_outcome}")
        
        ref_performance = {metric: ref_data[metric].mean() for metric in similarity_metrics}
        
        # Calculate similarity for other outcomes
        similarities = []
        
        for outcome in self.data['outcome_variable'].unique():
            if outcome == reference_outcome:
                continue
            
            outcome_data = successful_data[successful_data['outcome_variable'] == outcome]
            if len(outcome_data) == 0:
                continue
            
            # Calculate similarity based on performance metrics
            similarity_score = 0
            for metric in similarity_metrics:
                if metric in outcome_data.columns:
                    outcome_perf = outcome_data[metric].mean()
                    # Use inverse of absolute difference as similarity
                    diff = abs(ref_performance[metric] - outcome_perf)
                    similarity_score += 1 / (1 + diff)  # Higher score = more similar
            
            similarities.append({
                'outcome_variable': outcome,
                'similarity_score': similarity_score / len(similarity_metrics),
                'sample_size': len(outcome_data)
            })
        
        similarity_df = pd.DataFrame(similarities)
        return similarity_df.sort_values('similarity_score', ascending=False)
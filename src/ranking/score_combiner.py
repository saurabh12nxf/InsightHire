#!/usr/bin/env python3
"""
Score Combiner - Utility functions for combining and normalizing scores
Provides reusable functions for score manipulation and combination strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats


class ScoreCombiner:
    """
    Utility class for combining, normalizing, and manipulating candidate scores.
    Provides various score combination strategies and normalization methods.
    """
    
    @staticmethod
    def weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Calculate weighted average of scores
        
        Args:
            scores: Dictionary of score_name -> score_value
            weights: Dictionary of score_name -> weight_value
            
        Returns:
            Weighted average score
        """
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for score_name, score_value in scores.items():
            weight = weights.get(score_name, 0.0)
            total_weighted_score += score_value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_score / total_weight
    
    @staticmethod
    def normalize_scores(series: pd.Series, method: str = 'minmax', 
                        target_range: Tuple[float, float] = (0.0, 100.0)) -> pd.Series:
        """
        Normalize scores using various methods
        
        Args:
            series: Pandas series of scores to normalize
            method: Normalization method ('minmax', 'zscore', 'percentile')
            target_range: Target range for normalized scores
            
        Returns:
            Normalized scores
        """
        
        min_target, max_target = target_range
        
        if method == 'minmax':
            # Min-Max normalization
            min_score = series.min()
            max_score = series.max()
            
            if max_score == min_score:
                return pd.Series([min_target] * len(series), index=series.index)
            
            normalized = (series - min_score) / (max_score - min_score)
            return normalized * (max_target - min_target) + min_target
        
        elif method == 'zscore':
            # Z-score normalization
            z_scores = stats.zscore(series)
            # Map to target range (assumes z-scores between -3 and 3)
            normalized = (z_scores + 3) / 6  # Map to 0-1
            return normalized * (max_target - min_target) + min_target
        
        elif method == 'percentile':
            # Percentile-based normalization
            percentiles = series.rank(pct=True)
            return percentiles * (max_target - min_target) + min_target
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def combine_scores_multiplicative(scores: List[float], weights: Optional[List[float]] = None) -> float:
        """
        Combine scores using multiplicative approach
        Good for scenarios where all components must be reasonably good
        
        Args:
            scores: List of scores to combine
            weights: Optional weights for each score
            
        Returns:
            Combined score
        """
        
        if not scores:
            return 0.0
        
        if weights is None:
            weights = [1.0] * len(scores)
        
        # Convert scores to 0-1 range for multiplication
        normalized_scores = [max(0.0, min(1.0, score / 100.0)) for score in scores]
        
        combined = 1.0
        total_weight = sum(weights)
        
        for score, weight in zip(normalized_scores, weights):
            # Use weighted geometric mean approach
            combined *= score ** (weight / total_weight)
        
        return combined * 100.0  # Convert back to 0-100
    
    @staticmethod
    def combine_scores_harmonic(scores: List[float], weights: Optional[List[float]] = None) -> float:
        """
        Combine scores using harmonic mean
        Heavily penalizes low scores - good for must-have requirements
        
        Args:
            scores: List of scores to combine
            weights: Optional weights for each score
            
        Returns:
            Combined score using harmonic mean
        """
        
        if not scores:
            return 0.0
        
        if weights is None:
            weights = [1.0] * len(scores)
        
        # Avoid division by zero
        safe_scores = [max(0.1, score) for score in scores]
        
        weighted_reciprocals = sum(weight / score for score, weight in zip(safe_scores, weights))
        total_weight = sum(weights)
        
        if weighted_reciprocals == 0:
            return 0.0
        
        return total_weight / weighted_reciprocals
    
    @staticmethod
    def apply_threshold_penalties(scores: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
        """
        Apply penalty based on threshold violations
        
        Args:
            scores: Series of scores
            thresholds: Dictionary of threshold_name -> threshold_value
            
        Returns:
            Scores with penalties applied
        """
        
        penalized_scores = scores.copy()
        
        for threshold_name, threshold_value in thresholds.items():
            if threshold_name == 'min_score':
                penalty_mask = scores < threshold_value
                penalty_amount = (threshold_value - scores[penalty_mask]) * 0.5
                penalized_scores[penalty_mask] -= penalty_amount
            
            elif threshold_name == 'max_score':
                penalty_mask = scores > threshold_value
                penalty_amount = (scores[penalty_mask] - threshold_value) * 0.2
                penalized_scores[penalty_mask] -= penalty_amount
        
        # Ensure scores stay within bounds
        return penalized_scores.clip(0, 100)
    
    @staticmethod
    def calculate_confidence_score(candidate_scores: Dict[str, float], 
                                 score_reliabilities: Dict[str, float]) -> float:
        """
        Calculate confidence score based on score reliability and coverage
        
        Args:
            candidate_scores: Dictionary of score_name -> score_value
            score_reliabilities: Dictionary of score_name -> reliability (0-1)
            
        Returns:
            Confidence score (0-100)
        """
        
        if not candidate_scores:
            return 0.0
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for score_name, score_value in candidate_scores.items():
            reliability = score_reliabilities.get(score_name, 0.5)
            
            # Confidence is higher for extreme scores (very good or very bad)
            # and lower for middle scores (uncertain)
            score_certainty = abs(score_value - 50.0) / 50.0
            
            score_confidence = reliability * score_certainty
            
            total_confidence += score_confidence
            total_weight += reliability
        
        if total_weight == 0:
            return 50.0  # Neutral confidence
        
        return (total_confidence / total_weight) * 100.0
    
    @staticmethod
    def detect_score_anomalies(candidates_df: pd.DataFrame, 
                             score_columns: List[str],
                             z_threshold: float = 2.5) -> pd.DataFrame:
        """
        Detect anomalous score patterns that might indicate data issues
        
        Args:
            candidates_df: DataFrame with candidate scores
            score_columns: List of score column names to analyze
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags added
        """
        
        anomaly_df = candidates_df.copy()
        
        for score_col in score_columns:
            if score_col in anomaly_df.columns:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(anomaly_df[score_col].fillna(50)))
                
                # Flag anomalies
                anomaly_col = f'{score_col}_anomaly'
                anomaly_df[anomaly_col] = z_scores > z_threshold
                
                # Calculate anomaly severity
                severity_col = f'{score_col}_anomaly_severity'
                anomaly_df[severity_col] = z_scores
        
        # Overall anomaly flag
        anomaly_columns = [col for col in anomaly_df.columns if col.endswith('_anomaly')]
        if anomaly_columns:
            anomaly_df['has_anomalies'] = anomaly_df[anomaly_columns].any(axis=1)
            anomaly_df['anomaly_count'] = anomaly_df[anomaly_columns].sum(axis=1)
        
        return anomaly_df
    
    @staticmethod
    def create_score_tiers(candidates_df: pd.DataFrame, 
                          score_column: str,
                          tier_definitions: Optional[List[Tuple[str, float, float]]] = None) -> pd.DataFrame:
        """
        Create score tiers for candidate categorization
        
        Args:
            candidates_df: DataFrame with candidate scores
            score_column: Column name containing scores to tier
            tier_definitions: List of (tier_name, min_score, max_score) tuples
            
        Returns:
            DataFrame with tier column added
        """
        
        if tier_definitions is None:
            tier_definitions = [
                ('Elite', 85, 100),
                ('Excellent', 75, 85),
                ('Good', 65, 75),
                ('Fair', 50, 65),
                ('Poor', 0, 50)
            ]
        
        tier_df = candidates_df.copy()
        tier_column = f'{score_column}_tier'
        
        # Initialize with default tier
        tier_df[tier_column] = 'Unclassified'
        
        # Apply tier definitions
        for tier_name, min_score, max_score in tier_definitions:
            mask = (tier_df[score_column] >= min_score) & (tier_df[score_column] < max_score)
            tier_df.loc[mask, tier_column] = tier_name
        
        # Handle edge case for maximum score
        if tier_definitions:
            max_tier = max(tier_definitions, key=lambda x: x[2])
            tier_df.loc[tier_df[score_column] >= max_tier[2], tier_column] = max_tier[0]
        
        return tier_df
    
    @staticmethod
    def calculate_score_stability(candidates_df: pd.DataFrame, 
                                score_columns: List[str]) -> pd.DataFrame:
        """
        Calculate score stability (consistency across different scoring methods)
        
        Args:
            candidates_df: DataFrame with multiple score columns
            score_columns: List of score column names to analyze
            
        Returns:
            DataFrame with stability metrics added
        """
        
        stability_df = candidates_df.copy()
        
        # Get scores for analysis
        score_data = stability_df[score_columns]
        
        # Calculate stability metrics
        stability_df['score_mean'] = score_data.mean(axis=1)
        stability_df['score_std'] = score_data.std(axis=1)
        stability_df['score_range'] = score_data.max(axis=1) - score_data.min(axis=1)
        
        # Stability score (lower std and range = higher stability)
        max_std = stability_df['score_std'].max()
        max_range = stability_df['score_range'].max()
        
        if max_std > 0:
            std_stability = 1 - (stability_df['score_std'] / max_std)
        else:
            std_stability = 1.0
        
        if max_range > 0:
            range_stability = 1 - (stability_df['score_range'] / max_range)
        else:
            range_stability = 1.0
        
        stability_df['score_stability'] = ((std_stability + range_stability) / 2) * 100
        
        return stability_df
    
    @staticmethod
    def create_ensemble_score(candidates_df: pd.DataFrame,
                            score_columns: List[str],
                            ensemble_method: str = 'weighted_average',
                            weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Create ensemble score from multiple scoring approaches
        
        Args:
            candidates_df: DataFrame with multiple scores
            score_columns: List of score columns to combine
            ensemble_method: Method to combine scores
            weights: Optional weights for scoring methods
            
        Returns:
            Series with ensemble scores
        """
        
        if weights is None:
            weights = {col: 1.0 for col in score_columns}
        
        ensemble_scores = []
        
        for idx, candidate in candidates_df.iterrows():
            scores = {col: candidate.get(col, 50.0) for col in score_columns}
            
            if ensemble_method == 'weighted_average':
                score = ScoreCombiner.weighted_average(scores, weights)
            elif ensemble_method == 'multiplicative':
                score_values = list(scores.values())
                weight_values = [weights.get(col, 1.0) for col in score_columns]
                score = ScoreCombiner.combine_scores_multiplicative(score_values, weight_values)
            elif ensemble_method == 'harmonic':
                score_values = list(scores.values())
                weight_values = [weights.get(col, 1.0) for col in score_columns]
                score = ScoreCombiner.combine_scores_harmonic(score_values, weight_values)
            else:
                # Default to simple average
                score = sum(scores.values()) / len(scores)
            
            ensemble_scores.append(score)
        
        return pd.Series(ensemble_scores, index=candidates_df.index)


# Utility functions for common score operations
def rank_candidates(candidates_df: pd.DataFrame, 
                   score_column: str,
                   ascending: bool = False) -> pd.DataFrame:
    """Rank candidates by score"""
    ranked_df = candidates_df.copy()
    ranked_df['rank'] = ranked_df[score_column].rank(method='dense', ascending=ascending)
    return ranked_df.sort_values('rank')


def filter_by_percentile(candidates_df: pd.DataFrame,
                        score_column: str,
                        percentile: float = 75.0) -> pd.DataFrame:
    """Filter candidates above given percentile"""
    threshold = candidates_df[score_column].quantile(percentile / 100.0)
    return candidates_df[candidates_df[score_column] >= threshold]


def create_score_summary(candidates_df: pd.DataFrame,
                        score_columns: List[str]) -> Dict[str, Dict[str, float]]:
    """Create summary statistics for score columns"""
    summary = {}
    
    for col in score_columns:
        if col in candidates_df.columns:
            summary[col] = {
                'mean': candidates_df[col].mean(),
                'median': candidates_df[col].median(),
                'std': candidates_df[col].std(),
                'min': candidates_df[col].min(),
                'max': candidates_df[col].max(),
                'count': candidates_df[col].count()
            }
    
    return summary


if __name__ == "__main__":
    print("Score Combiner - Utility functions for score manipulation")
    print("This module provides reusable functions for:")
    print("• Score normalization and combination")
    print("• Anomaly detection in scores")  
    print("• Score stability analysis")
    print("• Ensemble scoring methods")
    print("• Statistical summaries")
    
    # Example usage
    print("\nExample: Combining scores with different methods")
    sample_scores = [85.0, 72.0, 91.0, 68.0]
    sample_weights = [0.4, 0.3, 0.2, 0.1]
    
    combiner = ScoreCombiner()
    
    weighted_avg = combiner.combine_scores_multiplicative(sample_scores, sample_weights)
    harmonic_mean = combiner.combine_scores_harmonic(sample_scores, sample_weights)
    
    print(f"Multiplicative combination: {weighted_avg:.1f}")
    print(f"Harmonic mean combination: {harmonic_mean:.1f}")
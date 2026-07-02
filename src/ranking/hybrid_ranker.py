#!/usr/bin/env python3
"""
Hybrid Ranker - Combines multiple scoring approaches into final rankings
Intelligently weights and combines semantic, rule-based, feature, and behavioral scores
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class HybridRanker:
    """
    Hybrid ranking engine that combines multiple scoring approaches.
    Creates final candidate rankings using intelligent score combination.
    """
    
    def __init__(self, jd_profile: Dict[str, Any], job_requirements: JobRequirements):
        self.jd_profile = jd_profile
        self.job_requirements = job_requirements
        
        # Get matching strategy from JD profile or use defaults
        self.matching_strategy = jd_profile.get('matching_strategy', {
            'technical_weight': 0.25,
            'career_weight': 0.25,
            'behavioral_weight': 0.25,
            'trajectory_weight': 0.15,
            'credibility_weight': 0.10
        })
        
        # Hybrid scoring weights - how to combine different score types
        self.hybrid_weights = {
            'feature_scores': 0.40,      # Feature engineering scores (Phase 3)
            'semantic_similarity': 0.25,  # AI embedding similarity
            'rule_based_score': 0.25,    # Hard constraints and logic
            'behavioral_bonus': 0.10      # Availability and engagement bonus
        }
        
        # Scoring configuration
        self.config = {
            'min_credibility_threshold': 60.0,    # Minimum credibility to be considered
            'hard_constraint_bonus': 10.0,        # Bonus for passing all hard constraints
            'semantic_similarity_threshold': 0.3, # Minimum semantic similarity
            'experience_mismatch_penalty': 15.0,  # Penalty for major experience mismatch
            'availability_bonus_threshold': 80.0  # Availability score for bonus
        }
    
    def calculate_hybrid_scores(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final hybrid scores combining all approaches
        
        Args:
            candidates_df: DataFrame with all previous scores
            
        Returns:
            DataFrame with final hybrid scores and rankings
        """
        
        print("Calculating hybrid scores...")
        
        # Validate required columns
        required_columns = [
            'overall_candidate_score', 'technical_fit_score', 'career_score',
            'behavioral_score', 'trajectory_score', 'credibility_score'
        ]
        
        missing_columns = [col for col in required_columns if col not in candidates_df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print("Some score components may be missing.")
        
        # Initialize hybrid scores
        candidates_df['hybrid_score'] = 0.0
        candidates_df['score_breakdown'] = [{}] * len(candidates_df)
        candidates_df['final_rank'] = 0
        candidates_df['is_recommended'] = False
        
        for idx, candidate in candidates_df.iterrows():
            score_components = {}
            
            # 1. Feature-based score (from Phase 3)
            feature_score = self._calculate_feature_component_score(candidate)
            score_components['feature_score'] = feature_score
            
            # 2. Semantic similarity score
            semantic_score = self._get_semantic_score(candidate)
            score_components['semantic_score'] = semantic_score
            
            # 3. Rule-based score  
            rule_score = self._get_rule_based_score(candidate)
            score_components['rule_score'] = rule_score
            
            # 4. Behavioral bonus
            behavioral_bonus = self._calculate_behavioral_bonus(candidate)
            score_components['behavioral_bonus'] = behavioral_bonus
            
            # 5. Apply penalties and bonuses
            penalties = self._calculate_penalties(candidate)
            bonuses = self._calculate_bonuses(candidate)
            
            score_components['penalties'] = penalties
            score_components['bonuses'] = bonuses
            
            # Combine scores using hybrid weights
            hybrid_score = (
                feature_score * self.hybrid_weights['feature_scores'] +
                semantic_score * self.hybrid_weights['semantic_similarity'] +
                rule_score * self.hybrid_weights['rule_based_score'] +
                behavioral_bonus * self.hybrid_weights['behavioral_bonus']
            )
            
            # Apply penalties and bonuses
            hybrid_score += bonuses
            hybrid_score += penalties  # penalties are negative
            
            # Ensure score is within 0-100 range
            hybrid_score = max(0.0, min(100.0, hybrid_score))
            
            # Update DataFrame
            candidates_df.at[idx, 'hybrid_score'] = hybrid_score
            candidates_df.at[idx, 'score_breakdown'] = score_components
        
        # Calculate rankings
        candidates_df['final_rank'] = candidates_df['hybrid_score'].rank(method='dense', ascending=False).astype(int)
        
        # Mark recommended candidates (top performers who meet quality thresholds)
        candidates_df['is_recommended'] = self._determine_recommendations(candidates_df)
        
        print(f"Hybrid scoring completed for {len(candidates_df)} candidates")
        
        # Summary statistics
        avg_score = candidates_df['hybrid_score'].mean()
        recommended_count = candidates_df['is_recommended'].sum()
        
        print(f"Average hybrid score: {avg_score:.1f}")
        print(f"Recommended candidates: {recommended_count}/{len(candidates_df)} ({recommended_count/len(candidates_df)*100:.1f}%)")
        
        return candidates_df
    
    def _calculate_feature_component_score(self, candidate: pd.Series) -> float:
        """Calculate weighted feature component score"""
        
        # Get individual feature scores
        technical_score = candidate.get('technical_fit_score', 50.0)
        career_score = candidate.get('career_score', 50.0)
        behavioral_score = candidate.get('behavioral_score', 50.0)
        trajectory_score = candidate.get('trajectory_score', 50.0)
        credibility_score = candidate.get('credibility_score', 80.0)
        
        # Weight according to job requirements (from matching strategy)
        weighted_score = (
            technical_score * self.matching_strategy.get('technical_weight', 0.25) +
            career_score * self.matching_strategy.get('career_weight', 0.25) +
            behavioral_score * self.matching_strategy.get('behavioral_weight', 0.25) +
            trajectory_score * self.matching_strategy.get('trajectory_weight', 0.15) +
            credibility_score * self.matching_strategy.get('credibility_weight', 0.10)
        )
        
        return weighted_score
    
    def _get_semantic_score(self, candidate: pd.Series) -> float:
        """Get semantic similarity score"""
        
        # Check if semantic similarity was calculated
        if 'semantic_similarity_score' in candidate:
            return candidate['semantic_similarity_score']
        elif 'semantic_similarity' in candidate:
            return candidate['semantic_similarity'] * 100  # Convert to 0-100 scale
        else:
            # No semantic similarity available - use neutral score
            return 50.0
    
    def _get_rule_based_score(self, candidate: pd.Series) -> float:
        """Get rule-based score"""
        
        if 'rule_based_score' in candidate:
            return candidate['rule_based_score']
        else:
            # No rule-based score - estimate from feature scores
            required_coverage = candidate.get('required_skills_coverage', 50.0)
            experience_fit = candidate.get('experience_fit_score', 50.0)
            return (required_coverage + experience_fit) / 2
    
    def _calculate_behavioral_bonus(self, candidate: pd.Series) -> float:
        """Calculate behavioral bonus score"""
        
        behavioral_score = candidate.get('behavioral_score', 50.0)
        availability_score = candidate.get('availability_score', 50.0)
        engagement_score = candidate.get('engagement_score', 50.0)
        
        # Behavioral bonus is average of behavioral factors
        behavioral_bonus = (behavioral_score + availability_score + engagement_score) / 3
        
        return behavioral_bonus
    
    def _calculate_penalties(self, candidate: pd.Series) -> float:
        """Calculate penalties for various red flags"""
        
        total_penalty = 0.0
        
        # Credibility penalty
        credibility_score = candidate.get('credibility_score', 80.0)
        if credibility_score < self.config['min_credibility_threshold']:
            penalty = (self.config['min_credibility_threshold'] - credibility_score) * 0.3
            total_penalty -= penalty
        
        # Experience mismatch penalty
        experience_fit = candidate.get('experience_fit_score', 50.0)
        if experience_fit < 30.0:  # Major experience mismatch
            total_penalty -= self.config['experience_mismatch_penalty']
        
        # Hard constraints penalty
        if 'hard_constraints_passed' in candidate and not candidate['hard_constraints_passed']:
            total_penalty -= 20.0  # Significant penalty for failing hard constraints
        
        # Low technical fit penalty (for technical roles)
        if self.job_requirements.role_type == 'technical':
            technical_fit = candidate.get('technical_fit_score', 50.0)
            if technical_fit < 20.0:
                total_penalty -= 15.0
        
        return total_penalty
    
    def _calculate_bonuses(self, candidate: pd.Series) -> float:
        """Calculate bonuses for exceptional qualities"""
        
        total_bonus = 0.0
        
        # Hard constraints bonus
        if 'hard_constraints_passed' in candidate and candidate['hard_constraints_passed']:
            total_bonus += self.config['hard_constraint_bonus']
        
        # High availability bonus
        availability_score = candidate.get('availability_score', 50.0)
        if availability_score >= self.config['availability_bonus_threshold']:
            total_bonus += 5.0
        
        # High credibility bonus
        credibility_score = candidate.get('credibility_score', 80.0)
        if credibility_score >= 90.0:
            total_bonus += 3.0
        
        # GitHub activity bonus (for technical roles)
        if self.job_requirements.role_type == 'technical':
            github_score = candidate.get('github_activity_score', 0)
            if github_score > 70:
                total_bonus += 5.0
        
        # High semantic similarity bonus
        semantic_similarity = candidate.get('semantic_similarity', 0.5)
        if semantic_similarity > 0.7:
            total_bonus += 5.0
        
        # Perfect skills match bonus
        required_coverage = candidate.get('required_skills_coverage', 0.0)
        if required_coverage >= 80.0:
            total_bonus += 8.0
        
        return total_bonus
    
    def _determine_recommendations(self, candidates_df: pd.DataFrame) -> pd.Series:
        """Determine which candidates to recommend"""
        
        recommendations = pd.Series([False] * len(candidates_df), index=candidates_df.index)
        
        # Recommendation criteria
        min_hybrid_score = 60.0
        min_credibility = 70.0
        
        # Candidates must meet minimum thresholds
        meets_score_threshold = candidates_df['hybrid_score'] >= min_hybrid_score
        meets_credibility = candidates_df['credibility_score'] >= min_credibility
        
        # Hard constraints (if available)
        if 'hard_constraints_passed' in candidates_df.columns:
            passes_constraints = candidates_df['hard_constraints_passed']
        else:
            passes_constraints = pd.Series([True] * len(candidates_df), index=candidates_df.index)
        
        # Combine all criteria
        recommendations = meets_score_threshold & meets_credibility & passes_constraints
        
        return recommendations
    
    def get_final_rankings(self, candidates_df: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
        """
        Get final candidate rankings
        
        Args:
            candidates_df: DataFrame with hybrid scores
            top_k: Number of top candidates to return
            
        Returns:
            DataFrame with top candidates ranked by hybrid score
        """
        
        if 'hybrid_score' not in candidates_df.columns:
            print("Error: Hybrid scores not calculated. Run calculate_hybrid_scores first.")
            return candidates_df.head(top_k)
        
        # Sort by hybrid score (descending) and take top k
        top_candidates = candidates_df.nlargest(top_k, 'hybrid_score').copy()
        
        # Reset rank based on this subset
        top_candidates['final_rank'] = range(1, len(top_candidates) + 1)
        
        return top_candidates
    
    def analyze_hybrid_results(self, candidates_df: pd.DataFrame, top_k: int = 20):
        """Analyze hybrid ranking results"""
        
        print("\n" + "="*70)
        print("HYBRID RANKING ANALYSIS - FINAL RESULTS")
        print("="*70)
        
        if 'hybrid_score' not in candidates_df.columns:
            print("Error: Hybrid scores not calculated.")
            return
        
        # Overall statistics
        total_candidates = len(candidates_df)
        recommended_candidates = candidates_df['is_recommended'].sum()
        avg_score = candidates_df['hybrid_score'].mean()
        median_score = candidates_df['hybrid_score'].median()
        
        print(f"📊 OVERALL STATISTICS:")
        print(f"   Total Candidates: {total_candidates}")
        print(f"   Recommended: {recommended_candidates} ({recommended_candidates/total_candidates*100:.1f}%)")
        print(f"   Average Score: {avg_score:.1f}")
        print(f"   Median Score: {median_score:.1f}")
        print(f"   Score Range: {candidates_df['hybrid_score'].min():.1f} - {candidates_df['hybrid_score'].max():.1f}")
        
        # Top candidates
        top_candidates = candidates_df.nlargest(top_k, 'hybrid_score')
        
        print(f"\n🏆 TOP {top_k} CANDIDATES:")
        print("-" * 70)
        print(f"{'Rank':<4} {'Candidate ID':<15} {'Hybrid':<6} {'Feature':<7} {'Semantic':<8} {'Rule':<6} {'Rec':<4}")
        print("-" * 70)
        
        for i, (idx, candidate) in enumerate(top_candidates.iterrows(), 1):
            candidate_id = candidate['candidate_id']
            hybrid_score = candidate['hybrid_score']
            feature_score = candidate.get('overall_candidate_score', 0)
            semantic_score = candidate.get('semantic_similarity_score', candidate.get('semantic_similarity', 0.5) * 100)
            rule_score = candidate.get('rule_based_score', 0)
            is_recommended = '✓' if candidate.get('is_recommended', False) else '✗'
            
            print(f"{i:3d}. {candidate_id:<15} {hybrid_score:5.1f}  {feature_score:5.1f}   {semantic_score:6.1f}   {rule_score:4.1f}  {is_recommended}")
        
        # Score breakdown for top candidate
        if len(top_candidates) > 0:
            top_candidate = top_candidates.iloc[0]
            breakdown = top_candidate.get('score_breakdown', {})
            
            print(f"\n🔍 TOP CANDIDATE SCORE BREAKDOWN ({top_candidate['candidate_id']}):")
            print(f"   Hybrid Score: {top_candidate['hybrid_score']:.1f}")
            for component, score in breakdown.items():
                print(f"   {component:20s}: {score:5.1f}")
        
        # Score distribution
        print(f"\n📈 SCORE DISTRIBUTION:")
        score_bins = [(90, 100, "Exceptional"), (80, 90, "Excellent"), (70, 80, "Very Good"), 
                     (60, 70, "Good"), (50, 60, "Fair"), (0, 50, "Poor")]
        
        for min_score, max_score, label in score_bins:
            if min_score == 0:
                count = (candidates_df['hybrid_score'] < max_score).sum()
            else:
                count = ((candidates_df['hybrid_score'] >= min_score) & 
                        (candidates_df['hybrid_score'] < max_score)).sum()
            percentage = count / len(candidates_df) * 100
            print(f"   {label:12s} ({min_score:2d}-{max_score:2d}): {count:3d} ({percentage:5.1f}%)")
        
        print(f"\n✅ Checkpoint 4: Review the top {top_k} candidates above")
        print(f"Ask yourself:")
        print(f"1. Would YOU hire them for this role?")
        print(f"2. Why did Candidate 4 rank above Candidate 5?")
        print(f"3. Does the ranking look human and logical?")
        
        return top_candidates
    
    def export_rankings(self, candidates_df: pd.DataFrame, 
                       output_path: str = 'Data/processed/final_rankings.csv',
                       top_k: int = 100) -> Path:
        """Export final rankings to CSV"""
        
        # Get top candidates
        top_candidates = self.get_final_rankings(candidates_df, top_k)
        
        # Select columns for export
        export_columns = [
            'final_rank', 'candidate_id', 'hybrid_score', 
            'overall_candidate_score', 'technical_fit_score', 'career_score',
            'behavioral_score', 'credibility_score', 'is_recommended'
        ]
        
        # Add optional columns if they exist
        optional_columns = [
            'semantic_similarity_score', 'rule_based_score', 
            'required_skills_coverage', 'availability_score'
        ]
        
        for col in optional_columns:
            if col in top_candidates.columns:
                export_columns.append(col)
        
        # Filter to existing columns
        available_columns = [col for col in export_columns if col in top_candidates.columns]
        export_df = top_candidates[available_columns].copy()
        
        # Round scores to 1 decimal place
        score_columns = [col for col in export_df.columns if 'score' in col.lower()]
        for col in score_columns:
            export_df[col] = export_df[col].round(1)
        
        # Save to CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_file, index=False)
        
        print(f"\n📁 Rankings exported to: {output_file}")
        print(f"   Top {len(export_df)} candidates saved")
        print(f"   Columns: {', '.join(available_columns)}")
        
        return output_file


if __name__ == "__main__":
    print("Hybrid Ranker - Final Candidate Ranking Engine")
    print("This module combines all scoring approaches into final rankings.")
    print("\nExample usage:")
    print("1. Load candidate features with all previous scores")
    print("2. ranker = HybridRanker(jd_profile, job_requirements)")
    print("3. candidates_df = ranker.calculate_hybrid_scores(candidates_df)")
    print("4. top_candidates = ranker.get_final_rankings(candidates_df, top_k=100)")
    print("5. ranker.analyze_hybrid_results(candidates_df)")
    print("6. ranker.export_rankings(candidates_df, 'rankings.csv')")
#!/usr/bin/env python3
"""
Confidence Engine - Measures AI confidence in ranking decisions
Distinguishes between candidate score (how good) vs confidence (how sure we are)
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


class ConfidenceEngine:
    """
    Calculates confidence scores for ranking decisions.
    
    High confidence = Rich data, consistent signals, clear match/mismatch
    Low confidence = Sparse data, conflicting signals, borderline cases
    """
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
        
        # Confidence weights for different data quality factors
        self.confidence_weights = {
            'data_completeness': 0.25,     # How complete is the candidate profile
            'signal_consistency': 0.30,    # Do different signals agree
            'evidence_strength': 0.25,     # How strong is the evidence
            'decision_clarity': 0.20       # How clear is the match/mismatch
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_skills_for_confidence': 3,
            'min_experience_for_confidence': 1,
            'profile_completeness_threshold': 0.7,
            'github_confidence_boost': 15.0,
            'consistency_penalty_threshold': 20.0
        }
    
    def calculate_confidence_scores(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence scores for all candidates
        
        Args:
            candidates_df: DataFrame with candidate features and scores
            
        Returns:
            DataFrame with confidence scores added
        """
        
        print("Calculating confidence scores...")
        
        confidence_scores = []
        confidence_breakdowns = []
        
        for idx, candidate in candidates_df.iterrows():
            # Calculate confidence components
            data_completeness = self._calculate_data_completeness(candidate)
            signal_consistency = self._calculate_signal_consistency(candidate)
            evidence_strength = self._calculate_evidence_strength(candidate)
            decision_clarity = self._calculate_decision_clarity(candidate)
            
            # Combine confidence components
            confidence_score = (
                data_completeness * self.confidence_weights['data_completeness'] +
                signal_consistency * self.confidence_weights['signal_consistency'] +
                evidence_strength * self.confidence_weights['evidence_strength'] +
                decision_clarity * self.confidence_weights['decision_clarity']
            )
            
            # Ensure confidence is between 0-100
            confidence_score = max(0.0, min(100.0, confidence_score))
            
            confidence_scores.append(confidence_score)
            confidence_breakdowns.append({
                'data_completeness': data_completeness,
                'signal_consistency': signal_consistency,
                'evidence_strength': evidence_strength,
                'decision_clarity': decision_clarity
            })
        
        candidates_df['confidence_score'] = confidence_scores
        candidates_df['confidence_breakdown'] = confidence_breakdowns
        
        # Add confidence categories
        candidates_df['confidence_level'] = candidates_df['confidence_score'].apply(self._categorize_confidence)
        
        print(f"Confidence calculation completed for {len(candidates_df)} candidates")
        
        # Summary statistics
        avg_confidence = candidates_df['confidence_score'].mean()
        high_confidence_count = (candidates_df['confidence_score'] >= 75).sum()
        
        print(f"Average confidence: {avg_confidence:.1f}")
        print(f"High confidence candidates (≥75): {high_confidence_count}/{len(candidates_df)} ({high_confidence_count/len(candidates_df)*100:.1f}%)")
        
        return candidates_df
    
    def _calculate_data_completeness(self, candidate: pd.Series) -> float:
        """Calculate confidence based on data completeness"""
        
        completeness_score = 0.0
        
        # Profile completeness (if available)
        profile_completeness = candidate.get('profile_quality_score', 50.0)
        completeness_score += profile_completeness * 0.4
        
        # Skills data completeness
        total_relevant_skills = candidate.get('total_relevant_skills', 0)
        if total_relevant_skills >= self.quality_thresholds['min_skills_for_confidence']:
            completeness_score += 25.0
        else:
            completeness_score += total_relevant_skills * 8.0  # Partial credit
        
        # Experience data
        years_experience = candidate.get('total_years_experience', 0)
        if years_experience >= self.quality_thresholds['min_experience_for_confidence']:
            completeness_score += 20.0
        
        # Career history completeness
        avg_job_duration = candidate.get('average_job_duration_months', 0)
        if avg_job_duration > 0:
            completeness_score += 15.0  # Has career history
        
        return min(completeness_score, 100.0)
    
    def _calculate_signal_consistency(self, candidate: pd.Series) -> float:
        """Calculate confidence based on signal consistency"""
        
        # Get main scores
        technical_score = candidate.get('technical_fit_score', 50.0)
        career_score = candidate.get('career_score', 50.0)
        behavioral_score = candidate.get('behavioral_score', 50.0)
        credibility_score = candidate.get('credibility_score', 80.0)
        
        main_scores = [technical_score, career_score, behavioral_score]
        
        # Calculate score variance (lower variance = higher consistency)
        score_variance = np.var(main_scores)
        consistency_score = max(0.0, 100.0 - score_variance * 0.5)  # Penalty for high variance
        
        # Credibility consistency check
        if credibility_score < 70.0:
            consistency_score -= 20.0  # Penalty for low credibility
        
        # Check for specific inconsistencies
        inconsistency_penalty = 0.0
        
        # Technical claims vs GitHub activity (for technical roles)
        if self.job_requirements.role_type == 'technical':
            github_score = candidate.get('github_activity_score', 0)
            if technical_score > 50.0 and github_score < 10.0:
                inconsistency_penalty += 15.0  # Claims technical skills but no GitHub
        
        # Experience vs skill claims consistency
        years_experience = candidate.get('total_years_experience', 0)
        if technical_score > 70.0 and years_experience < 2.0:
            inconsistency_penalty += 10.0  # High technical score but low experience
        
        consistency_score -= inconsistency_penalty
        
        return max(0.0, min(100.0, consistency_score))
    
    def _calculate_evidence_strength(self, candidate: pd.Series) -> float:
        """Calculate confidence based on strength of evidence"""
        
        evidence_score = 0.0
        
        # Strong technical evidence
        technical_fit = candidate.get('technical_fit_score', 0)
        required_coverage = candidate.get('required_skills_coverage', 0)
        
        if required_coverage >= 80.0:
            evidence_score += 30.0  # Strong skill match
        elif required_coverage >= 50.0:
            evidence_score += 20.0  # Moderate skill match
        elif required_coverage >= 25.0:
            evidence_score += 10.0  # Some skill match
        
        # GitHub activity (for technical roles)
        if self.job_requirements.role_type == 'technical':
            github_score = candidate.get('github_activity_score', 0)
            if github_score > 50:
                evidence_score += 20.0
            elif github_score > 20:
                evidence_score += 10.0
        
        # Experience strength
        experience_fit = candidate.get('experience_fit_score', 50.0)
        if experience_fit >= 80.0:
            evidence_score += 15.0
        elif experience_fit >= 60.0:
            evidence_score += 10.0
        
        # Behavioral evidence strength
        engagement_score = candidate.get('engagement_score', 50.0)
        availability_score = candidate.get('availability_score', 50.0)
        
        if engagement_score >= 70.0:
            evidence_score += 10.0
        if availability_score >= 80.0:
            evidence_score += 10.0
        
        # Career progression evidence
        trajectory_score = candidate.get('trajectory_score', 50.0)
        if trajectory_score >= 75.0:
            evidence_score += 15.0
        
        return min(evidence_score, 100.0)
    
    def _calculate_decision_clarity(self, candidate: pd.Series) -> float:
        """Calculate confidence based on decision clarity"""
        
        # Get main scores
        overall_score = candidate.get('overall_candidate_score', 50.0)
        hybrid_score = candidate.get('hybrid_score', 50.0)
        
        # Use hybrid score if available, otherwise overall score
        main_score = hybrid_score if hybrid_score > 0 else overall_score
        
        # Clear decisions (very high or very low scores) = high confidence
        # Borderline decisions (middle scores) = low confidence
        
        if main_score >= 80.0:
            clarity_score = 90.0  # Clear positive decision
        elif main_score >= 70.0:
            clarity_score = 75.0  # Good candidate
        elif main_score >= 60.0:
            clarity_score = 60.0  # Decent candidate
        elif main_score >= 40.0:
            clarity_score = 40.0  # Borderline - low confidence
        elif main_score >= 30.0:
            clarity_score = 60.0  # Clear negative - somewhat confident
        else:
            clarity_score = 80.0  # Very clear negative - high confidence
        
        # Adjust based on hard constraints
        constraints_passed = candidate.get('hard_constraints_passed', True)
        if not constraints_passed:
            clarity_score += 15.0  # Clear rejection - high confidence
        
        return min(clarity_score, 100.0)
    
    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize confidence score into levels"""
        
        if confidence_score >= 85.0:
            return 'Very High'
        elif confidence_score >= 70.0:
            return 'High'
        elif confidence_score >= 55.0:
            return 'Medium'
        elif confidence_score >= 40.0:
            return 'Low'
        else:
            return 'Very Low'
    
    def analyze_confidence_results(self, candidates_df: pd.DataFrame, top_k: int = 20):
        """Analyze confidence calculation results"""
        
        print("\n" + "="*60)
        print("CONFIDENCE ENGINE ANALYSIS")
        print("="*60)
        
        if 'confidence_score' not in candidates_df.columns:
            print("Error: Confidence scores not calculated.")
            return
        
        # Overall statistics
        total_candidates = len(candidates_df)
        avg_confidence = candidates_df['confidence_score'].mean()
        
        print(f"📊 CONFIDENCE STATISTICS:")
        print(f"   Total Candidates: {total_candidates}")
        print(f"   Average Confidence: {avg_confidence:.1f}")
        
        # Confidence level distribution
        confidence_counts = candidates_df['confidence_level'].value_counts()
        print(f"\n📈 CONFIDENCE DISTRIBUTION:")
        for level, count in confidence_counts.items():
            percentage = count / total_candidates * 100
            print(f"   {level:10s}: {count:3d} candidates ({percentage:5.1f}%)")
        
        # Top candidates with confidence
        top_candidates = candidates_df.nlargest(top_k, 'hybrid_score')
        
        print(f"\n🏆 TOP {top_k} CANDIDATES WITH CONFIDENCE:")
        print(f"{'Rank':<4} {'Candidate ID':<15} {'Score':<6} {'Confidence':<10} {'Level':<10}")
        print("-" * 60)
        
        for i, (idx, candidate) in enumerate(top_candidates.iterrows(), 1):
            candidate_id = candidate['candidate_id']
            score = candidate.get('hybrid_score', candidate.get('overall_candidate_score', 0))
            confidence = candidate['confidence_score']
            level = candidate['confidence_level']
            
            print(f"{i:3d}. {candidate_id:<15} {score:5.1f}  {confidence:8.1f}  {level:<10}")
        
        # High score vs low confidence cases (potential issues)
        high_score_low_conf = candidates_df[
            (candidates_df['hybrid_score'] > 60) & 
            (candidates_df['confidence_score'] < 50)
        ]
        
        if len(high_score_low_conf) > 0:
            print(f"\n⚠️  HIGH SCORE, LOW CONFIDENCE CASES ({len(high_score_low_conf)}):")
            for idx, candidate in high_score_low_conf.iterrows():
                print(f"   {candidate['candidate_id']}: Score={candidate['hybrid_score']:.1f}, Confidence={candidate['confidence_score']:.1f}")
        
        print(f"\n✅ Checkpoint 1: Every candidate now has confidence_score (0-100)")
        print(f"Score ≠ Confidence. High scores with low confidence need manual review.")
    
    def get_confidence_summary(self, candidates_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of confidence analysis"""
        
        if 'confidence_score' not in candidates_df.columns:
            return {}
        
        return {
            'total_candidates': len(candidates_df),
            'average_confidence': candidates_df['confidence_score'].mean(),
            'confidence_distribution': candidates_df['confidence_level'].value_counts().to_dict(),
            'high_confidence_count': (candidates_df['confidence_score'] >= 75).sum(),
            'low_confidence_high_score': len(candidates_df[
                (candidates_df['hybrid_score'] > 60) & 
                (candidates_df['confidence_score'] < 50)
            ])
        }


if __name__ == "__main__":
    print("Confidence Engine - AI Decision Confidence Measurement")
    print("This module calculates how confident the AI is about its ranking decisions.")
    print("\nExample usage:")
    print("1. Load candidates with ranking scores")
    print("2. engine = ConfidenceEngine(job_requirements)")
    print("3. candidates_df = engine.calculate_confidence_scores(candidates_df)")
    print("4. engine.analyze_confidence_results(candidates_df)")
#!/usr/bin/env python3
"""
Honeypot Detector - Advanced fake candidate detection system
Builds on Phase 3 credibility features with enhanced detection algorithms
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import re
from collections import Counter

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class HoneypotDetector:
    """
    Advanced honeypot detection system that identifies fake/synthetic candidates.
    
    Detection methods:
    1. Profile inconsistencies
    2. Impossible timelines
    3. Skill-experience mismatches
    4. Pattern recognition
    5. Statistical anomalies
    """
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
        
        # Detection weights
        self.detection_weights = {
            'profile_inconsistencies': 0.25,
            'timeline_violations': 0.20,
            'skill_mismatches': 0.20,
            'pattern_anomalies': 0.20,
            'statistical_outliers': 0.15
        }
        
        # Honeypot indicators
        self.honeypot_keywords = [
            'test', 'fake', 'dummy', 'sample', 'example', 'placeholder',
            'lorem', 'ipsum', 'temp', 'trial', 'demo'
        ]
        
        # Suspicious patterns
        self.suspicious_patterns = {
            'perfect_scores': [90, 95, 100],  # Suspiciously perfect scores
            'round_numbers': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'common_durations': [12, 24, 36, 48, 60],  # Suspiciously round job durations
        }
    
    def detect_honeypots(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run comprehensive honeypot detection on all candidates
        
        Args:
            candidates_df: DataFrame with candidate data and scores
            
        Returns:
            DataFrame with honeypot detection results added
        """
        
        print("Running honeypot detection...")
        
        honeypot_scores = []
        honeypot_flags = []
        detection_details = []
        
        for idx, candidate in candidates_df.iterrows():
            # Run all detection methods
            inconsistency_score = self._detect_profile_inconsistencies(candidate)
            timeline_score = self._detect_timeline_violations(candidate)
            skill_mismatch_score = self._detect_skill_mismatches(candidate)
            pattern_score = self._detect_pattern_anomalies(candidate)
            statistical_score = self._detect_statistical_outliers(candidate, candidates_df)
            
            # Combine detection scores
            honeypot_risk = (
                inconsistency_score * self.detection_weights['profile_inconsistencies'] +
                timeline_score * self.detection_weights['timeline_violations'] +
                skill_mismatch_score * self.detection_weights['skill_mismatches'] +
                pattern_score * self.detection_weights['pattern_anomalies'] +
                statistical_score * self.detection_weights['statistical_outliers']
            )
            
            # Convert to 0-100 risk score (higher = more likely honeypot)
            honeypot_risk = min(100.0, honeypot_risk * 100)
            
            # Flag as honeypot if risk is high
            is_honeypot = honeypot_risk >= 70.0
            
            honeypot_scores.append(honeypot_risk)
            honeypot_flags.append(is_honeypot)
            
            # Store detection details
            detection_details.append({
                'profile_inconsistencies': inconsistency_score,
                'timeline_violations': timeline_score,
                'skill_mismatches': skill_mismatch_score,
                'pattern_anomalies': pattern_score,
                'statistical_outliers': statistical_score,
                'overall_risk': honeypot_risk
            })
        
        # Add results to DataFrame
        candidates_df['honeypot_risk_score'] = honeypot_scores
        candidates_df['honeypot_flag'] = honeypot_flags
        candidates_df['honeypot_details'] = detection_details
        
        # Update credibility scores based on honeypot detection
        candidates_df['enhanced_credibility_score'] = self._calculate_enhanced_credibility(candidates_df)
        
        # Summary statistics
        total_candidates = len(candidates_df)
        honeypot_count = sum(honeypot_flags)
        avg_risk = np.mean(honeypot_scores)
        
        print(f"Honeypot detection completed for {total_candidates} candidates")
        print(f"Detected honeypots: {honeypot_count}/{total_candidates} ({honeypot_count/total_candidates*100:.1f}%)")
        print(f"Average risk score: {avg_risk:.1f}")
        
        return candidates_df
    
    def _detect_profile_inconsistencies(self, candidate: pd.Series) -> float:
        """Detect inconsistencies within candidate profile"""
        
        inconsistency_score = 0.0
        
        # Use existing credibility score as baseline
        credibility_score = candidate.get('credibility_score', 80.0)
        base_inconsistency = max(0.0, (80.0 - credibility_score) / 80.0)
        inconsistency_score += base_inconsistency * 0.5
        
        # Check for honeypot keywords in text fields
        candidate_id = candidate.get('candidate_id', '')
        if any(keyword in candidate_id.lower() for keyword in self.honeypot_keywords):
            inconsistency_score += 0.3
        
        # Skill credibility issues
        skill_credibility = candidate.get('skill_credibility_score', 80.0)
        if skill_credibility < 50.0:
            inconsistency_score += 0.2
        
        # Experience consistency issues
        exp_consistency = candidate.get('experience_consistency_score', 80.0)
        if exp_consistency < 50.0:
            inconsistency_score += 0.2
        
        # GitHub credibility for technical roles
        if self.job_requirements.role_type == 'technical':
            github_credibility = candidate.get('github_credibility_score', 80.0)
            if github_credibility < 40.0:
                inconsistency_score += 0.15
        
        return min(inconsistency_score, 1.0)
    
    def _detect_timeline_violations(self, candidate: pd.Series) -> float:
        """Detect impossible or suspicious timelines"""
        
        timeline_score = 0.0
        
        # Check for impossible timeline flags
        impossible_timelines = candidate.get('impossible_timelines', 0)
        if impossible_timelines > 0:
            timeline_score += min(impossible_timelines * 0.2, 0.4)
        
        # Suspicious job durations (too short or too perfect)
        avg_duration = candidate.get('average_job_duration_months', 24)
        
        # Very short average durations (< 6 months) are suspicious
        if avg_duration < 6:
            timeline_score += 0.3
        
        # Suspiciously round durations (exactly 12, 24, 36 months)
        if avg_duration in self.suspicious_patterns['round_numbers']:
            timeline_score += 0.1
        
        # Experience vs age inconsistencies
        years_experience = candidate.get('total_years_experience', 0)
        if years_experience > 40:  # Impossible for most people
            timeline_score += 0.4
        elif years_experience > 30:  # Very high experience
            timeline_score += 0.2
        
        return min(timeline_score, 1.0)
    
    def _detect_skill_mismatches(self, candidate: pd.Series) -> float:
        """Detect skill-experience mismatches"""
        
        mismatch_score = 0.0
        
        # Suspicious skill claims
        suspicious_skills = candidate.get('suspicious_skill_claims', 0)
        if suspicious_skills > 0:
            mismatch_score += min(suspicious_skills * 0.15, 0.5)
        
        # Expert claims with minimal experience
        technical_score = candidate.get('technical_fit_score', 0)
        years_experience = candidate.get('total_years_experience', 0)
        
        # High technical score but very low experience
        if technical_score > 70 and years_experience < 2:
            mismatch_score += 0.3
        
        # Many relevant skills but low experience
        relevant_skills = candidate.get('total_relevant_skills', 0)
        if relevant_skills > 10 and years_experience < 3:
            mismatch_score += 0.2
        
        # Perfect skills coverage with low experience
        skills_coverage = candidate.get('required_skills_coverage', 0)
        if skills_coverage == 100.0 and years_experience < 3:
            mismatch_score += 0.2
        
        return min(mismatch_score, 1.0)
    
    def _detect_pattern_anomalies(self, candidate: pd.Series) -> float:
        """Detect suspicious patterns in candidate data"""
        
        pattern_score = 0.0
        
        # Perfect or suspiciously round scores
        scores_to_check = [
            candidate.get('technical_fit_score', 0),
            candidate.get('career_score', 0),
            candidate.get('behavioral_score', 0),
            candidate.get('trajectory_score', 0)
        ]
        
        perfect_scores = sum(1 for score in scores_to_check if score in self.suspicious_patterns['perfect_scores'])
        round_scores = sum(1 for score in scores_to_check if score in self.suspicious_patterns['round_numbers'])
        
        if perfect_scores >= 2:
            pattern_score += 0.3
        elif perfect_scores >= 1:
            pattern_score += 0.15
        
        if round_scores >= 3:
            pattern_score += 0.2
        
        # Suspiciously high profile completeness
        profile_quality = candidate.get('profile_quality_score', 50.0)
        if profile_quality >= 95.0:
            pattern_score += 0.1
        
        # Too many skills (unrealistic for most candidates)
        relevant_skills = candidate.get('total_relevant_skills', 0)
        if relevant_skills > 20:
            pattern_score += 0.2
        elif relevant_skills > 15:
            pattern_score += 0.1
        
        # Suspicious availability patterns
        availability_score = candidate.get('availability_score', 50.0)
        notice_period = candidate.get('notice_period_days', 60)
        
        if availability_score == 100.0 and notice_period == 0:
            pattern_score += 0.1  # Too perfect availability
        
        return min(pattern_score, 1.0)
    
    def _detect_statistical_outliers(self, candidate: pd.Series, all_candidates: pd.DataFrame) -> float:
        """Detect statistical outliers that might indicate synthetic data"""
        
        outlier_score = 0.0
        
        # Check if candidate is an outlier in multiple dimensions
        outlier_dimensions = 0
        
        # Technical score outlier
        technical_scores = all_candidates['technical_fit_score']
        candidate_technical = candidate.get('technical_fit_score', 0)
        
        if len(technical_scores) > 10:  # Need sufficient data
            technical_z = abs((candidate_technical - technical_scores.mean()) / technical_scores.std())
            if technical_z > 2.5:  # More than 2.5 standard deviations
                outlier_dimensions += 1
        
        # Career score outlier
        career_scores = all_candidates['career_score']
        candidate_career = candidate.get('career_score', 0)
        
        if len(career_scores) > 10:
            career_z = abs((candidate_career - career_scores.mean()) / career_scores.std())
            if career_z > 2.5:
                outlier_dimensions += 1
        
        # Experience outlier
        experience_values = all_candidates['total_years_experience']
        candidate_exp = candidate.get('total_years_experience', 0)
        
        if len(experience_values) > 10:
            exp_z = abs((candidate_exp - experience_values.mean()) / experience_values.std())
            if exp_z > 2.5:
                outlier_dimensions += 1
        
        # Multiple outlier dimensions indicate potential synthetic data
        if outlier_dimensions >= 3:
            outlier_score += 0.4
        elif outlier_dimensions >= 2:
            outlier_score += 0.2
        elif outlier_dimensions >= 1:
            outlier_score += 0.1
        
        return min(outlier_score, 1.0)
    
    def _calculate_enhanced_credibility(self, candidates_df: pd.DataFrame) -> pd.Series:
        """Calculate enhanced credibility score incorporating honeypot detection"""
        
        enhanced_scores = []
        
        for idx, candidate in candidates_df.iterrows():
            base_credibility = candidate.get('credibility_score', 80.0)
            honeypot_risk = candidate.get('honeypot_risk_score', 0.0)
            
            # Reduce credibility based on honeypot risk
            credibility_penalty = honeypot_risk * 0.8  # Up to 80% penalty for high risk
            enhanced_credibility = base_credibility - credibility_penalty
            
            # Ensure credibility stays within bounds
            enhanced_credibility = max(0.0, min(100.0, enhanced_credibility))
            
            enhanced_scores.append(enhanced_credibility)
        
        return pd.Series(enhanced_scores, index=candidates_df.index)
    
    def analyze_honeypot_results(self, candidates_df: pd.DataFrame):
        """Analyze honeypot detection results"""
        
        print("\n" + "="*60)
        print("HONEYPOT DETECTION ANALYSIS")
        print("="*60)
        
        if 'honeypot_flag' not in candidates_df.columns:
            print("Error: Honeypot detection not completed.")
            return
        
        # Overall statistics
        total_candidates = len(candidates_df)
        honeypots_detected = candidates_df['honeypot_flag'].sum()
        avg_risk = candidates_df['honeypot_risk_score'].mean()
        
        print(f"📊 DETECTION SUMMARY:")
        print(f"   Total Candidates: {total_candidates}")
        print(f"   Honeypots Detected: {honeypots_detected} ({honeypots_detected/total_candidates*100:.1f}%)")
        print(f"   Average Risk Score: {avg_risk:.1f}")
        
        # Risk distribution
        risk_bins = [(0, 25, 'Low'), (25, 50, 'Medium'), (50, 70, 'High'), (70, 100, 'Critical')]
        
        print(f"\n📈 RISK DISTRIBUTION:")
        for min_risk, max_risk, label in risk_bins:
            count = ((candidates_df['honeypot_risk_score'] >= min_risk) & 
                    (candidates_df['honeypot_risk_score'] < max_risk)).sum()
            if min_risk == 70:  # Last bin should be inclusive
                count = (candidates_df['honeypot_risk_score'] >= min_risk).sum()
            percentage = count / total_candidates * 100
            print(f"   {label:8s} ({min_risk:2d}-{max_risk:2d}): {count:3d} candidates ({percentage:5.1f}%)")
        
        # High-risk candidates
        high_risk_candidates = candidates_df[candidates_df['honeypot_risk_score'] >= 70]
        
        if len(high_risk_candidates) > 0:
            print(f"\n🚨 HIGH-RISK CANDIDATES (≥70% risk):")
            for idx, candidate in high_risk_candidates.iterrows():
                candidate_id = candidate['candidate_id']
                risk_score = candidate['honeypot_risk_score']
                print(f"   {candidate_id}: {risk_score:.1f}% risk")
        
        print(f"\n✅ Checkpoint 3: Honeypot detection completed")
        print(f"   Enhanced credibility_score and honeypot_flag generated for all candidates")
    
    def generate_honeypot_report(self, candidates_df: pd.DataFrame,
                               output_path: str = 'Data/processed/honeypot_report.txt') -> Path:
        """Generate detailed honeypot detection report"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("HONEYPOT DETECTION REPORT\n")
            f.write("="*40 + "\n\n")
            
            f.write(f"Job: {self.job_requirements.role_title}\n")
            f.write(f"Detection Date: {pd.Timestamp.now()}\n\n")
            
            # Summary statistics
            total = len(candidates_df)
            honeypots = candidates_df['honeypot_flag'].sum()
            avg_risk = candidates_df['honeypot_risk_score'].mean()
            
            f.write("SUMMARY:\n")
            f.write(f"  Total Candidates: {total}\n")
            f.write(f"  Honeypots Detected: {honeypots} ({honeypots/total*100:.1f}%)\n")
            f.write(f"  Average Risk Score: {avg_risk:.1f}\n\n")
            
            # High-risk candidates details
            high_risk = candidates_df[candidates_df['honeypot_risk_score'] >= 50].sort_values(
                'honeypot_risk_score', ascending=False
            )
            
            f.write("HIGH-RISK CANDIDATES:\n")
            f.write("-" * 30 + "\n")
            
            for idx, candidate in high_risk.iterrows():
                details = candidate['honeypot_details']
                f.write(f"\n{candidate['candidate_id']} (Risk: {candidate['honeypot_risk_score']:.1f}%)\n")
                f.write(f"  Profile Inconsistencies: {details['profile_inconsistencies']:.2f}\n")
                f.write(f"  Timeline Violations: {details['timeline_violations']:.2f}\n")
                f.write(f"  Skill Mismatches: {details['skill_mismatches']:.2f}\n")
                f.write(f"  Pattern Anomalies: {details['pattern_anomalies']:.2f}\n")
                f.write(f"  Statistical Outliers: {details['statistical_outliers']:.2f}\n")
        
        print(f"Honeypot report saved to: {output_file}")
        return output_file


if __name__ == "__main__":
    print("Honeypot Detector - Advanced Fake Candidate Detection")
    print("This module identifies synthetic/fake candidates using multiple detection methods.")
    print("\nExample usage:")
    print("1. Load candidates with existing credibility scores")
    print("2. detector = HoneypotDetector(job_requirements)")
    print("3. candidates_df = detector.detect_honeypots(candidates_df)")
    print("4. detector.analyze_honeypot_results(candidates_df)")
    print("5. detector.generate_honeypot_report(candidates_df)")
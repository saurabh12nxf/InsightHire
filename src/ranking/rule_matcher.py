#!/usr/bin/env python3
"""
Rule-Based Matcher - Logical rule-based candidate evaluation
Applies hard constraints and business logic for candidate filtering and scoring
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


class RuleBasedMatcher:
    """
    Rule-based matching engine that applies logical constraints and business rules.
    Handles hard requirements, deal-breakers, and preference-based scoring.
    """
    
    def __init__(self, jd_profile: Dict[str, Any], job_requirements: JobRequirements):
        self.jd_profile = jd_profile
        self.job_requirements = job_requirements
        
        # Rule weights (configurable)
        self.rule_weights = {
            'required_skills': 1.0,        # Must-have skills
            'preferred_skills': 0.6,       # Nice-to-have skills
            'experience_range': 0.8,       # Experience requirements
            'behavioral_requirements': 0.7, # GitHub, response rate, etc.
            'positive_signals': 0.4,       # Bonus factors
            'negative_signals': -0.8,      # Penalty factors
            'company_preferences': 0.3,    # Company type preferences
            'availability': 0.5            # Notice period, open to work
        }
        
    def calculate_rule_based_scores(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rule-based scores for all candidates
        
        Args:
            candidates_df: DataFrame with candidate features
            
        Returns:
            DataFrame with added rule-based scores
        """
        
        print("Calculating rule-based scores...")
        
        # Initialize scores
        candidates_df['rule_based_score'] = 0.0
        candidates_df['hard_constraints_passed'] = True
        candidates_df['rule_breakdown'] = [{}] * len(candidates_df)
        
        for idx, candidate in candidates_df.iterrows():
            score_breakdown = {}
            total_score = 0.0
            hard_constraints_passed = True
            
            # 1. Required Skills Check (Hard Constraint)
            required_score, required_passed = self._evaluate_required_skills(candidate)
            score_breakdown['required_skills'] = required_score
            total_score += required_score * self.rule_weights['required_skills']
            hard_constraints_passed &= required_passed
            
            # 2. Preferred Skills Bonus
            preferred_score = self._evaluate_preferred_skills(candidate)
            score_breakdown['preferred_skills'] = preferred_score
            total_score += preferred_score * self.rule_weights['preferred_skills']
            
            # 3. Experience Range Check
            experience_score, exp_passed = self._evaluate_experience_range(candidate)
            score_breakdown['experience_range'] = experience_score
            total_score += experience_score * self.rule_weights['experience_range']
            hard_constraints_passed &= exp_passed
            
            # 4. Behavioral Requirements
            behavioral_score, behavioral_passed = self._evaluate_behavioral_requirements(candidate)
            score_breakdown['behavioral_requirements'] = behavioral_score
            total_score += behavioral_score * self.rule_weights['behavioral_requirements']
            hard_constraints_passed &= behavioral_passed
            
            # 5. Positive Signals Bonus
            positive_score = self._evaluate_positive_signals(candidate)
            score_breakdown['positive_signals'] = positive_score
            total_score += positive_score * self.rule_weights['positive_signals']
            
            # 6. Negative Signals Penalty
            negative_score = self._evaluate_negative_signals(candidate)
            score_breakdown['negative_signals'] = negative_score
            total_score += negative_score * self.rule_weights['negative_signals']
            
            # 7. Company Preferences
            company_score = self._evaluate_company_preferences(candidate)
            score_breakdown['company_preferences'] = company_score
            total_score += company_score * self.rule_weights['company_preferences']
            
            # 8. Availability Check
            availability_score = self._evaluate_availability(candidate)
            score_breakdown['availability'] = availability_score
            total_score += availability_score * self.rule_weights['availability']
            
            # Normalize to 0-100 scale
            normalized_score = max(0.0, min(100.0, total_score * 20))  # Scale factor
            
            # Update DataFrame
            candidates_df.at[idx, 'rule_based_score'] = normalized_score
            candidates_df.at[idx, 'hard_constraints_passed'] = hard_constraints_passed
            candidates_df.at[idx, 'rule_breakdown'] = score_breakdown
        
        print(f"Rule-based scoring completed for {len(candidates_df)} candidates")
        
        # Summary statistics
        passed_constraints = candidates_df['hard_constraints_passed'].sum()
        avg_score = candidates_df['rule_based_score'].mean()
        
        print(f"Candidates passing hard constraints: {passed_constraints}/{len(candidates_df)} ({passed_constraints/len(candidates_df)*100:.1f}%)")
        print(f"Average rule-based score: {avg_score:.1f}")
        
        return candidates_df
    
    def _evaluate_required_skills(self, candidate: pd.Series) -> Tuple[float, bool]:
        """Evaluate required skills - hard constraint"""
        
        required_skills = self.jd_profile.get('required_skills', [])
        if not required_skills:
            return 1.0, True  # No requirements = pass
        
        # Get candidate's skills coverage
        required_coverage = candidate.get('required_skills_coverage', 0.0) / 100.0
        
        # Hard constraint: must have at least 50% of required skills
        hard_constraint_passed = required_coverage >= 0.5
        
        # Score based on coverage percentage
        score = required_coverage
        
        return score, hard_constraint_passed
    
    def _evaluate_preferred_skills(self, candidate: pd.Series) -> float:
        """Evaluate preferred skills - bonus points"""
        
        preferred_skills = self.jd_profile.get('preferred_skills', [])
        if not preferred_skills:
            return 0.0  # No bonus if no preferred skills
        
        # Get candidate's preferred skills coverage
        preferred_coverage = candidate.get('preferred_skills_coverage', 0.0) / 100.0
        
        # Score is directly the coverage percentage
        return preferred_coverage
    
    def _evaluate_experience_range(self, candidate: pd.Series) -> Tuple[float, bool]:
        """Evaluate experience requirements"""
        
        exp_range = self.jd_profile.get('experience_range', [0, 20])
        candidate_experience = candidate.get('total_years_experience', 0)
        
        min_exp, max_exp = exp_range
        
        # Hard constraint check
        if min_exp > 0:
            hard_constraint_passed = candidate_experience >= min_exp * 0.8  # 20% tolerance
        else:
            hard_constraint_passed = True
        
        # Scoring based on fit within range
        if min_exp <= candidate_experience <= max_exp:
            score = 1.0  # Perfect fit
        elif candidate_experience < min_exp:
            # Under-qualified
            gap = min_exp - candidate_experience
            score = max(0.0, 1.0 - gap * 0.2)  # Penalty for each year under
        else:
            # Over-qualified  
            excess = candidate_experience - max_exp
            score = max(0.6, 1.0 - excess * 0.1)  # Smaller penalty for over-qualification
        
        return score, hard_constraint_passed
    
    def _evaluate_behavioral_requirements(self, candidate: pd.Series) -> Tuple[float, bool]:
        """Evaluate behavioral requirements"""
        
        behavioral_reqs = self.jd_profile.get('behavioral_requirements', {})
        score = 0.0
        hard_constraint_passed = True
        total_checks = 0
        
        # GitHub requirement
        if behavioral_reqs.get('requires_github', False):
            total_checks += 1
            has_github = candidate.get('has_github_activity', False)
            github_score = candidate.get('github_activity_score', 0)
            
            if has_github and github_score > 20:
                score += 1.0
            elif has_github:
                score += 0.5
            else:
                hard_constraint_passed = False
                score += 0.0
        
        # Response rate requirement
        min_response_rate = behavioral_reqs.get('min_response_rate')
        if min_response_rate:
            total_checks += 1
            candidate_response_rate = candidate.get('engagement_score', 50) / 100.0
            
            if candidate_response_rate >= min_response_rate:
                score += 1.0
            else:
                score += candidate_response_rate / min_response_rate
                if candidate_response_rate < min_response_rate * 0.7:  # 30% tolerance
                    hard_constraint_passed = False
        
        # Notice period requirement
        max_notice_period = behavioral_reqs.get('max_notice_period_days')
        if max_notice_period:
            total_checks += 1
            candidate_notice = candidate.get('notice_period_days', 60)
            
            if candidate_notice is None:
                candidate_notice = 60  # Default assumption
            
            if candidate_notice <= max_notice_period:
                score += 1.0
            else:
                # Linear penalty
                excess_days = candidate_notice - max_notice_period
                score += max(0.0, 1.0 - excess_days / 90.0)  # 90 days = full penalty
        
        # Normalize score
        if total_checks > 0:
            score = score / total_checks
        else:
            score = 1.0  # No behavioral requirements
        
        return score, hard_constraint_passed
    
    def _evaluate_positive_signals(self, candidate: pd.Series) -> float:
        """Evaluate positive signals - bonus factors"""
        
        positive_signals = self.jd_profile.get('positive_signals', [])
        if not positive_signals:
            return 0.0
        
        score = 0.0
        
        # Check various positive indicators
        # Note: In a real system, you'd analyze candidate text/profile for these signals
        # For now, we use available numeric features as proxies
        
        # High GitHub activity (proxy for open source contributions)
        if 'open source' in ' '.join(positive_signals).lower():
            github_score = candidate.get('github_activity_score', 0)
            if github_score > 60:
                score += 0.3
        
        # Leadership (proxy from trajectory score)
        if any(signal in ['leadership', 'mentoring'] for signal in positive_signals):
            trajectory_score = candidate.get('trajectory_score', 50)
            if trajectory_score > 70:
                score += 0.2
        
        # Learning agility (proxy from career progression)
        if 'learning' in ' '.join(positive_signals).lower():
            growth_velocity = candidate.get('growth_velocity_score', 50)
            if growth_velocity > 70:
                score += 0.2
        
        # High availability (proxy for immediate joiner)
        if 'availability' in ' '.join(positive_signals).lower():
            availability_score = candidate.get('availability_score', 50)
            if availability_score > 80:
                score += 0.2
        
        # High engagement (proxy for customer-facing skills)
        if any(signal in ['customer', 'communication'] for signal in positive_signals):
            engagement_score = candidate.get('engagement_score', 50)
            if engagement_score > 70:
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _evaluate_negative_signals(self, candidate: pd.Series) -> float:
        """Evaluate negative signals - penalty factors"""
        
        negative_signals = self.jd_profile.get('negative_signals', [])
        if not negative_signals:
            return 0.0
        
        penalty = 0.0
        
        # Check for negative indicators
        # Note: In real system, would analyze candidate text for these patterns
        # For now, use available features as proxies
        
        # Consulting-only background
        if 'consulting only' in ' '.join(negative_signals).lower():
            product_experience_ratio = candidate.get('product_experience_ratio', 0.5)
            if product_experience_ratio < 0.3:  # Less than 30% product experience
                penalty += 0.3
        
        # No technical background (for technical roles)
        if 'no technical' in ' '.join(negative_signals).lower():
            technical_fit = candidate.get('technical_fit_score', 50)
            if technical_fit < 20:
                penalty += 0.5
        
        # Job hopping pattern
        if 'unstable' in ' '.join(negative_signals).lower():
            stability_score = candidate.get('stability_score', 50)
            if stability_score < 40:
                penalty += 0.2
        
        # Low credibility
        if 'fake' in ' '.join(negative_signals).lower():
            credibility_score = candidate.get('credibility_score', 80)
            if credibility_score < 60:
                penalty += 0.4
        
        return -penalty  # Return negative value for penalty
    
    def _evaluate_company_preferences(self, candidate: pd.Series) -> float:
        """Evaluate company type preferences"""
        
        company_preferences = self.jd_profile.get('company_preferences', {})
        if not company_preferences:
            return 0.0
        
        # Get candidate's company type score (would need to be calculated in features)
        # For now, use product experience ratio as proxy
        product_ratio = candidate.get('product_experience_ratio', 0.5)
        
        # Score based on company preferences
        if product_ratio > 0.7:
            # Strong product background
            return company_preferences.get('product_companies', 0.5)
        elif product_ratio > 0.3:
            # Mixed background
            return (company_preferences.get('product_companies', 0.5) + 
                   company_preferences.get('consulting', 0.4)) / 2
        else:
            # Mostly consulting background
            return company_preferences.get('consulting', 0.4)
    
    def _evaluate_availability(self, candidate: pd.Series) -> float:
        """Evaluate candidate availability"""
        
        availability_score = candidate.get('availability_score', 50) / 100.0
        is_open_to_work = candidate.get('is_open_to_work', False)
        
        # Bonus for being open to work
        if is_open_to_work:
            availability_score += 0.2
        
        return min(availability_score, 1.0)
    
    def filter_hard_constraints(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Filter candidates who don't meet hard constraints"""
        
        if 'hard_constraints_passed' not in candidates_df.columns:
            print("Warning: Hard constraints not calculated. Run calculate_rule_based_scores first.")
            return candidates_df
        
        passed_candidates = candidates_df[candidates_df['hard_constraints_passed']].copy()
        failed_count = len(candidates_df) - len(passed_candidates)
        
        print(f"Hard constraint filtering: {len(passed_candidates)} passed, {failed_count} failed")
        
        return passed_candidates
    
    def analyze_rule_based_results(self, candidates_df: pd.DataFrame, top_k: int = 10):
        """Analyze rule-based scoring results"""
        
        print("\n" + "="*60)
        print("RULE-BASED MATCHING ANALYSIS")
        print("="*60)
        
        if 'rule_based_score' not in candidates_df.columns:
            print("Error: Rule-based scores not calculated.")
            return
        
        # Overall statistics
        total_candidates = len(candidates_df)
        passed_constraints = candidates_df['hard_constraints_passed'].sum()
        avg_score = candidates_df['rule_based_score'].mean()
        
        print(f"Total Candidates Evaluated: {total_candidates}")
        print(f"Hard Constraints Passed: {passed_constraints} ({passed_constraints/total_candidates*100:.1f}%)")
        print(f"Average Rule-Based Score: {avg_score:.1f}")
        
        # Top candidates
        top_candidates = candidates_df.nlargest(top_k, 'rule_based_score')
        
        print(f"\n🏆 TOP {top_k} CANDIDATES BY RULE-BASED SCORE:")
        print("-" * 60)
        
        for i, (idx, candidate) in enumerate(top_candidates.iterrows(), 1):
            candidate_id = candidate['candidate_id']
            rule_score = candidate['rule_based_score']
            constraints_passed = candidate['hard_constraints_passed']
            
            print(f"{i:2d}. {candidate_id:15s} | Score: {rule_score:5.1f} | Constraints: {'✓' if constraints_passed else '✗'}")
            
            # Show breakdown if available
            breakdown = candidate.get('rule_breakdown', {})
            if breakdown:
                print(f"    Required Skills: {breakdown.get('required_skills', 0):.2f} | "
                      f"Experience: {breakdown.get('experience_range', 0):.2f} | "
                      f"Behavioral: {breakdown.get('behavioral_requirements', 0):.2f}")
        
        print(f"\n✅ Checkpoint 3: Review the top 10 candidates above")
        print(f"Ask yourself: Does each score make sense based on the job requirements?")
        
        # Score distribution
        print(f"\n📊 SCORE DISTRIBUTION:")
        score_ranges = [
            (80, 100, "Excellent"),
            (60, 80, "Good"), 
            (40, 60, "Fair"),
            (0, 40, "Poor")
        ]
        
        for min_score, max_score, label in score_ranges:
            count = ((candidates_df['rule_based_score'] >= min_score) & 
                    (candidates_df['rule_based_score'] < max_score)).sum()
            if min_score == 80:  # Last range should be inclusive
                count = (candidates_df['rule_based_score'] >= min_score).sum()
            percentage = count / len(candidates_df) * 100
            print(f"   {label:10s} ({min_score:2d}-{max_score:2d}): {count:3d} candidates ({percentage:5.1f}%)")


if __name__ == "__main__":
    print("Rule-Based Matcher")
    print("This module requires JD profile and candidate features to test.")
    print("\nExample usage:")
    print("1. Create JD profile using jd_parser.py")
    print("2. Load candidate features from feature extraction")
    print("3. matcher = RuleBasedMatcher(jd_profile, job_requirements)")
    print("4. candidates_df = matcher.calculate_rule_based_scores(candidates_df)")
    print("5. filtered_df = matcher.filter_hard_constraints(candidates_df)")
    print("6. matcher.analyze_rule_based_results(candidates_df)")
#!/usr/bin/env python3
"""
Behavioral Features Extractor - Analyzes candidate engagement and availability signals
Measures responsiveness, GitHub activity, and availability against job requirements
"""

from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class BehavioralFeaturesExtractor:
    """
    Analyzes candidate behavioral signals for hiring likelihood and job fit.
    Focuses on engagement, availability, and professional activity.
    """
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
    
    def extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract behavioral features for a candidate
        
        Args:
            candidate: Candidate data dictionary
            
        Returns:
            Dictionary with behavioral feature scores
        """
        
        redrob_signals = candidate.get('redrob_signals', {})
        profile = candidate.get('profile', {})
        
        # Calculate availability score
        availability_score = self._calculate_availability_score(redrob_signals, profile)
        
        # Calculate engagement score  
        engagement_score = self._calculate_engagement_score(redrob_signals)
        
        # Calculate GitHub activity score
        github_score = self._calculate_github_score(redrob_signals)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(redrob_signals)
        
        # Calculate profile quality score
        profile_quality_score = self._calculate_profile_quality_score(redrob_signals)
        
        # Overall behavioral score
        behavioral_score = self._calculate_overall_behavioral_score(
            availability_score, engagement_score, github_score,
            reliability_score, profile_quality_score
        )
        
        return {
            'behavioral_score': round(behavioral_score, 2),
            'availability_score': round(availability_score, 2),
            'engagement_score': round(engagement_score, 2),
            'github_activity_score': round(github_score, 2),
            'reliability_score': round(reliability_score, 2),
            'profile_quality_score': round(profile_quality_score, 2),
            'is_open_to_work': redrob_signals.get('open_to_work', False),
            'notice_period_days': redrob_signals.get('notice_period_days', None),
            'has_github_activity': github_score > 0
        }
    
    def _calculate_availability_score(self, redrob_signals: Dict, profile: Dict) -> float:
        """
        Calculate availability score based on notice period and work status
        """
        
        # Open to work status
        open_to_work = redrob_signals.get('open_to_work', False)
        notice_period = redrob_signals.get('notice_period_days', 90)  # Default 90 days
        last_active_days = redrob_signals.get('last_active_days', 30)  # Days since last active
        
        base_score = 50.0
        
        # Open to work bonus
        if open_to_work:
            base_score += 25.0
        
        # Notice period scoring
        if notice_period == 0:  # Immediate joiner
            base_score += 25.0
        elif notice_period <= 30:  # 1 month
            base_score += 20.0
        elif notice_period <= 60:  # 2 months
            base_score += 10.0
        elif notice_period <= 90:  # 3 months (standard)
            base_score += 0.0
        else:  # More than 3 months
            penalty = min((notice_period - 90) * 0.2, 20.0)
            base_score -= penalty
        
        # Recent activity bonus
        if last_active_days <= 7:  # Very recent
            base_score += 10.0
        elif last_active_days <= 30:  # Recent
            base_score += 5.0
        elif last_active_days > 90:  # Inactive
            base_score -= 15.0
        
        # Job requirements adjustment
        if self.job_requirements.max_notice_period_days:
            if notice_period <= self.job_requirements.max_notice_period_days:
                base_score += 15.0  # Meets requirements
            else:
                base_score -= 25.0  # Doesn't meet requirements
        
        return max(min(base_score, 100.0), 0.0)
    
    def _calculate_engagement_score(self, redrob_signals: Dict) -> float:
        """
        Calculate engagement score based on recruiter interactions
        """
        
        response_rate = redrob_signals.get('recruiter_response_rate', 0.0) * 100
        profile_views = redrob_signals.get('profile_views_last_month', 0)
        saved_by_recruiters = redrob_signals.get('saved_by_recruiters', 0)
        
        # Response rate score (most important)
        response_score = response_rate  # Already 0-100
        
        # Profile views score (normalized to 0-100)
        if profile_views == 0:
            views_score = 0.0
        elif profile_views <= 5:
            views_score = 30.0
        elif profile_views <= 15:
            views_score = 60.0
        elif profile_views <= 30:
            views_score = 80.0
        else:
            views_score = 100.0
        
        # Saved by recruiters score
        if saved_by_recruiters == 0:
            saved_score = 0.0
        elif saved_by_recruiters <= 2:
            saved_score = 40.0
        elif saved_by_recruiters <= 5:
            saved_score = 70.0
        else:
            saved_score = 100.0
        
        # Weighted combination
        engagement_score = (
            response_score * 0.6 +      # Response rate most important
            views_score * 0.25 +        # Profile visibility
            saved_score * 0.15          # Recruiter interest
        )
        
        # Job requirements adjustment
        if self.job_requirements.min_response_rate:
            if response_rate >= (self.job_requirements.min_response_rate * 100):
                engagement_score += 10.0  # Bonus for meeting requirement
            else:
                engagement_score -= 15.0  # Penalty for not meeting requirement
        
        return max(min(engagement_score, 100.0), 0.0)
    
    def _calculate_github_score(self, redrob_signals: Dict) -> float:
        """
        Calculate GitHub activity score
        """
        
        github_score = redrob_signals.get('github_activity_score', -1)
        
        # Handle missing GitHub data
        if github_score == -1 or github_score is None:
            # If GitHub is required by job, penalize heavily
            if self.job_requirements.requires_github:
                return 0.0
            else:
                return 50.0  # Neutral if not required
        
        # GitHub score is already 0-100, just apply job requirements
        adjusted_score = github_score
        
        if self.job_requirements.requires_github:
            if github_score > 0:
                adjusted_score += 20.0  # Bonus for having GitHub when required
            else:
                adjusted_score = 0.0    # No GitHub when required
        
        return max(min(adjusted_score, 100.0), 0.0)
    
    def _calculate_reliability_score(self, redrob_signals: Dict) -> float:
        """
        Calculate reliability based on interview and offer history
        """
        
        interview_completion = redrob_signals.get('interview_completion_rate', 0.5) * 100
        offer_acceptance = redrob_signals.get('offer_acceptance_rate', 0.5) * 100
        
        # Both rates should be high for reliable candidate
        reliability_score = (interview_completion + offer_acceptance) / 2
        
        # Bonus for very high reliability
        if interview_completion >= 85 and offer_acceptance >= 80:
            reliability_score += 10.0
        
        return max(min(reliability_score, 100.0), 0.0)
    
    def _calculate_profile_quality_score(self, redrob_signals: Dict) -> float:
        """
        Calculate profile completeness and quality score
        """
        
        profile_completeness = redrob_signals.get('profile_completeness_percentage', 0.5) * 100
        
        # Profile completeness is the main factor
        quality_score = profile_completeness
        
        # Bonus thresholds
        if profile_completeness >= 90:
            quality_score += 10.0  # Exceptional profile
        elif profile_completeness >= 75:
            quality_score += 5.0   # Good profile
        
        return max(min(quality_score, 100.0), 0.0)
    
    def _calculate_overall_behavioral_score(self, availability: float, engagement: float,
                                          github: float, reliability: float,
                                          profile_quality: float) -> float:
        """
        Calculate overall behavioral score using weighted combination
        """
        
        # Base weights
        weights = {
            'availability': 0.25,
            'engagement': 0.30,
            'github': 0.20,
            'reliability': 0.15,
            'profile_quality': 0.10
        }
        
        # Adjust weights based on job requirements
        if self.job_requirements.requires_github:
            # Increase GitHub weight if required
            weights['github'] = 0.30
            weights['engagement'] = 0.25
        
        if self.job_requirements.max_notice_period_days:
            # Increase availability weight if specific requirement
            weights['availability'] = 0.35
            weights['engagement'] = 0.25
        
        # Calculate weighted score
        behavioral_score = (
            availability * weights['availability'] +
            engagement * weights['engagement'] +
            github * weights['github'] +
            reliability * weights['reliability'] +
            profile_quality * weights['profile_quality']
        )
        
        return max(min(behavioral_score, 100.0), 0.0)

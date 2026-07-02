#!/usr/bin/env python3
"""
Trajectory Features Extractor - Analyzes career progression and growth patterns
Measures promotions, job stability, career consistency, and growth trajectory
"""

from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path
from datetime import datetime, timedelta
import statistics

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class TrajectoryFeaturesExtractor:
    """
    Analyzes candidate career trajectory and progression patterns.
    Identifies growth trends, stability, and career development quality.
    """
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
        
        # Career progression keywords (configurable patterns)
        self.seniority_levels = {
            'intern': 1,
            'trainee': 1,
            'associate': 2,
            'junior': 2,
            'engineer': 3,
            'developer': 3,
            'analyst': 3,
            'specialist': 3,
            'senior': 4,
            'lead': 5,
            'principal': 5,
            'staff': 5,
            'manager': 6,
            'director': 7,
            'vp': 8,
            'vice president': 8,
            'cto': 9,
            'ceo': 9
        }
    
    def extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract trajectory features for a candidate
        
        Args:
            candidate: Candidate data dictionary
            
        Returns:
            Dictionary with career trajectory feature scores
        """
        
        career_history = candidate.get('career_history', [])
        profile = candidate.get('profile', {})
        
        if not career_history:
            return self._get_default_trajectory_features()
        
        # Calculate different trajectory aspects
        promotion_score = self._calculate_promotion_score(career_history)
        stability_score = self._calculate_stability_score(career_history)
        consistency_score = self._calculate_consistency_score(career_history)
        growth_velocity_score = self._calculate_growth_velocity(career_history, profile)
        salary_progression_score = self._calculate_salary_progression(career_history)
        
        # Overall trajectory score
        trajectory_score = self._calculate_overall_trajectory_score(
            promotion_score, stability_score, consistency_score,
            growth_velocity_score, salary_progression_score
        )
        
        return {
            'trajectory_score': round(trajectory_score, 2),
            'promotion_score': round(promotion_score, 2),
            'stability_score': round(stability_score, 2),
            'consistency_score': round(consistency_score, 2),
            'growth_velocity_score': round(growth_velocity_score, 2),
            'salary_progression_score': round(salary_progression_score, 2),
            'average_job_duration_months': self._calculate_average_job_duration(career_history),
            'total_promotions': self._count_promotions(career_history),
            'career_gaps_months': self._calculate_career_gaps(career_history),
            'is_career_changer': self._is_career_changer(career_history)
        }
    
    def _get_default_trajectory_features(self) -> Dict[str, Any]:
        """Return neutral scores for candidates with no career history"""
        return {
            'trajectory_score': 50.0,
            'promotion_score': 50.0,
            'stability_score': 50.0,
            'consistency_score': 50.0,
            'growth_velocity_score': 50.0,
            'salary_progression_score': 50.0,
            'average_job_duration_months': 0,
            'total_promotions': 0,
            'career_gaps_months': 0,
            'is_career_changer': False
        }
    
    def _calculate_promotion_score(self, career_history: List[Dict]) -> float:
        """
        Calculate promotion score based on title progression
        """
        
        if len(career_history) < 2:
            return 75.0  # Neutral for single job
        
        # Extract seniority levels for each job
        job_levels = []
        for job in career_history:
            title = job.get('title', '').lower()
            level = self._extract_seniority_level(title)
            job_levels.append(level)
        
        # Calculate promotion patterns
        promotions = 0
        demotions = 0
        lateral_moves = 0
        
        for i in range(1, len(job_levels)):
            if job_levels[i] > job_levels[i-1]:
                promotions += 1
            elif job_levels[i] < job_levels[i-1]:
                demotions += 1
            else:
                lateral_moves += 1
        
        total_moves = len(job_levels) - 1
        
        if total_moves == 0:
            return 75.0
        
        # Score based on promotion ratio
        promotion_ratio = promotions / total_moves
        demotion_ratio = demotions / total_moves
        
        # Base score from promotions
        promotion_score = promotion_ratio * 100
        
        # Penalty for demotions
        promotion_score -= demotion_ratio * 50
        
        # Bonus for consistent upward movement
        if promotions > 0 and demotions == 0:
            promotion_score += 20
        
        # Adjust for career stage expectations
        total_experience = sum(job.get('duration_months', 0) for job in career_history) / 12
        expected_promotions = max(1, int(total_experience / 3))  # Expect promotion every 3 years
        
        if promotions >= expected_promotions:
            promotion_score += 10
        
        return max(min(promotion_score, 100.0), 0.0)
    
    def _extract_seniority_level(self, title: str) -> int:
        """Extract numeric seniority level from job title"""
        
        # Default level
        level = 3
        
        # Check for seniority keywords
        for keyword, seniority in self.seniority_levels.items():
            if keyword in title:
                level = max(level, seniority)  # Take highest level found
        
        return level
    
    def _calculate_stability_score(self, career_history: List[Dict]) -> float:
        """
        Calculate job stability score based on tenure patterns
        """
        
        durations = [job.get('duration_months', 0) for job in career_history]
        
        if not durations:
            return 50.0
        
        avg_duration = statistics.mean(durations)
        
        # Count short tenures (job hopping indicator)
        short_jobs = sum(1 for d in durations if d < 12)  # Less than 1 year
        very_short_jobs = sum(1 for d in durations if d < 6)  # Less than 6 months
        
        total_jobs = len(durations)
        
        # Base stability score from average duration
        if avg_duration >= 36:  # 3+ years
            base_score = 90.0
        elif avg_duration >= 24:  # 2+ years
            base_score = 80.0
        elif avg_duration >= 18:  # 1.5+ years
            base_score = 70.0
        elif avg_duration >= 12:  # 1+ year
            base_score = 60.0
        else:
            base_score = 40.0
        
        # Penalties for job hopping
        short_job_penalty = (short_jobs / total_jobs) * 30
        very_short_job_penalty = (very_short_jobs / total_jobs) * 50
        
        stability_score = base_score - short_job_penalty - very_short_job_penalty
        
        # Bonus for trend improvement (recent jobs longer than earlier ones)
        if len(durations) >= 3:
            recent_avg = statistics.mean(durations[-2:])  # Last 2 jobs
            earlier_avg = statistics.mean(durations[:-2])  # Earlier jobs
            
            if recent_avg > earlier_avg * 1.2:  # 20% improvement
                stability_score += 15.0
        
        return max(min(stability_score, 100.0), 0.0)
    
    def _calculate_consistency_score(self, career_history: List[Dict]) -> float:
        """
        Calculate career consistency based on industry/domain focus
        """
        
        if len(career_history) < 2:
            return 80.0  # Neutral for single job
        
        # Analyze industry consistency
        industries = [job.get('industry', '').lower() for job in career_history]
        unique_industries = len(set(ind for ind in industries if ind))
        
        # Analyze role type consistency  
        role_types = []
        for job in career_history:
            title = job.get('title', '').lower()
            role_type = self._classify_role_type(title)
            role_types.append(role_type)
        
        unique_roles = len(set(role_types))
        
        total_jobs = len(career_history)
        
        # Industry consistency score
        if unique_industries <= 2:
            industry_score = 90.0
        elif unique_industries <= 3:
            industry_score = 70.0
        else:
            industry_score = 50.0
        
        # Role consistency score
        if unique_roles == 1:
            role_score = 100.0
        elif unique_roles == 2:
            role_score = 80.0
        elif unique_roles == 3:
            role_score = 60.0
        else:
            role_score = 40.0
        
        # Combined consistency score
        consistency_score = (industry_score + role_score) / 2
        
        # Bonus for deliberate career pivots vs random changes
        if self._is_strategic_pivot(career_history):
            consistency_score += 15.0
        
        return max(min(consistency_score, 100.0), 0.0)
    
    def _classify_role_type(self, title: str) -> str:
        """Classify job title into role type categories"""
        
        role_patterns = {
            'engineering': ['engineer', 'developer', 'programmer', 'architect'],
            'management': ['manager', 'director', 'head', 'lead', 'supervisor'],
            'sales': ['sales', 'account', 'business development'],
            'marketing': ['marketing', 'brand', 'campaign', 'content'],
            'operations': ['operations', 'ops', 'process'],
            'analyst': ['analyst', 'research'],
            'consultant': ['consultant', 'advisory']
        }
        
        for role_type, keywords in role_patterns.items():
            if any(keyword in title for keyword in keywords):
                return role_type
        
        return 'other'
    
    def _is_strategic_pivot(self, career_history: List[Dict]) -> bool:
        """
        Determine if career changes appear strategic vs random
        """
        
        if len(career_history) < 3:
            return False
        
        # Look for progression in related fields or clear upward movement
        job_levels = [self._extract_seniority_level(job.get('title', '').lower()) 
                     for job in career_history]
        
        # Strategic if overall upward trend despite role changes
        return job_levels[-1] > job_levels[0]
    
    def _calculate_growth_velocity(self, career_history: List[Dict], profile: Dict) -> float:
        """
        Calculate how fast the candidate has grown in their career
        """
        
        total_experience_years = profile.get('years_of_experience', 0)
        
        if total_experience_years == 0:
            return 50.0
        
        # Calculate seniority achieved vs experience
        current_title = profile.get('current_title', '').lower()
        current_level = self._extract_seniority_level(current_title)
        
        # Expected level based on experience
        if total_experience_years <= 2:
            expected_level = 2
        elif total_experience_years <= 5:
            expected_level = 3
        elif total_experience_years <= 8:
            expected_level = 4
        elif total_experience_years <= 12:
            expected_level = 5
        else:
            expected_level = 6
        
        # Score based on level vs expectation
        level_ratio = current_level / expected_level
        
        if level_ratio >= 1.2:  # 20% ahead
            velocity_score = 90.0
        elif level_ratio >= 1.0:  # On track
            velocity_score = 75.0
        elif level_ratio >= 0.8:  # Slightly behind
            velocity_score = 60.0
        else:  # Behind expectations
            velocity_score = 40.0
        
        # Bonus for rapid recent promotions
        promotions = self._count_promotions(career_history)
        promotion_velocity = promotions / max(total_experience_years, 1)
        
        if promotion_velocity > 0.5:  # More than 1 promotion per 2 years
            velocity_score += 15.0
        
        return max(min(velocity_score, 100.0), 0.0)
    
    def _count_promotions(self, career_history: List[Dict]) -> int:
        """Count total number of promotions in career"""
        
        if len(career_history) < 2:
            return 0
        
        promotions = 0
        job_levels = [self._extract_seniority_level(job.get('title', '').lower()) 
                     for job in career_history]
        
        for i in range(1, len(job_levels)):
            if job_levels[i] > job_levels[i-1]:
                promotions += 1
        
        return promotions
    
    def _calculate_salary_progression(self, career_history: List[Dict]) -> float:
        """
        Calculate salary progression if salary data is available
        """
        
        # Most datasets won't have salary data, return neutral
        # This is a placeholder for when salary data becomes available
        return 75.0
    
    def _calculate_average_job_duration(self, career_history: List[Dict]) -> float:
        """Calculate average job duration in months"""
        
        if not career_history:
            return 0.0
        
        durations = [job.get('duration_months', 0) for job in career_history]
        return statistics.mean(durations) if durations else 0.0
    
    def _calculate_career_gaps(self, career_history: List[Dict]) -> int:
        """Calculate total career gap periods in months"""
        
        # This would require start/end dates to calculate properly
        # Placeholder implementation
        return 0
    
    def _is_career_changer(self, career_history: List[Dict]) -> bool:
        """Determine if candidate has made significant career changes"""
        
        if len(career_history) < 2:
            return False
        
        role_types = [self._classify_role_type(job.get('title', '').lower()) 
                     for job in career_history]
        
        unique_role_types = len(set(role_types))
        return unique_role_types >= 3  # 3+ different role types indicates career change
    
    def _calculate_overall_trajectory_score(self, promotion: float, stability: float,
                                          consistency: float, velocity: float,
                                          salary: float) -> float:
        """
        Calculate overall trajectory score using weighted combination
        """
        
        # Base weights
        weights = {
            'promotion': 0.25,
            'stability': 0.25,
            'consistency': 0.20,
            'velocity': 0.20,
            'salary': 0.10
        }
        
        # Adjust weights based on job requirements
        if self.job_requirements.role_level == 'senior' or self.job_requirements.role_level == 'lead':
            # For senior roles, weight promotion and velocity more
            weights['promotion'] = 0.30
            weights['velocity'] = 0.25
            weights['stability'] = 0.20
        
        trajectory_score = (
            promotion * weights['promotion'] +
            stability * weights['stability'] +
            consistency * weights['consistency'] +
            velocity * weights['velocity'] +
            salary * weights['salary']
        )
        
        return max(min(trajectory_score, 100.0), 0.0)

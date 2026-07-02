#!/usr/bin/env python3
"""
Career Features Extractor - Analyzes career history and experience fit
Measures experience level, industry relevance, and career quality against job requirements
"""

from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path
from datetime import datetime, date

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class CareerFeaturesExtractor:
    """
    Analyzes candidate career history against job requirements.
   
    """
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
        
        # Company type patterns (configurable, not hardcoded business rules)
        self.company_patterns = {
            'product': ['swiggy', 'flipkart', 'zomato', 'razorpay', 'cred', 'paytm', 'ola', 'uber'],
            'consulting': ['tcs', 'infosys', 'wipro', 'accenture', 'capgemini', 'cognizant', 'deloitte'],
            'startup': ['startup', 'early stage', 'seed', 'series a'],
            'enterprise': ['microsoft', 'google', 'amazon', 'apple', 'facebook', 'meta', 'netflix']
        }
    
    def extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract career features for a candidate against job requirements
        
        Args:
            candidate: Candidate data dictionary
            
        Returns:
            Dictionary with career feature scores
        """
        
        profile = candidate.get('profile', {})
        career_history = candidate.get('career_history', [])
        
        # Calculate experience fit
        experience_score = self._calculate_experience_fit(profile)
        
        # Calculate industry relevance
        industry_score = self._calculate_industry_relevance(career_history)
        
        # Calculate company type fit
        company_type_score = self._calculate_company_type_fit(career_history)
        
        # Calculate role progression quality
        role_progression_score = self._calculate_role_progression(career_history)
        
        # Calculate current role relevance
        current_role_score = self._calculate_current_role_relevance(profile)
        
        # Overall career score
        career_score = self._calculate_overall_career_score(
            experience_score, industry_score, company_type_score,
            role_progression_score, current_role_score
        )
        
        return {
            'career_score': round(career_score, 2),
            'experience_fit_score': round(experience_score, 2),
            'industry_relevance_score': round(industry_score, 2),
            'company_type_score': round(company_type_score, 2),
            'role_progression_score': round(role_progression_score, 2),
            'current_role_score': round(current_role_score, 2),
            'total_years_experience': profile.get('years_of_experience', 0),
            'product_experience_ratio': self._calculate_product_experience_ratio(career_history),
            'current_industry': profile.get('current_industry', '').lower()
        }
    
    def _calculate_experience_fit(self, profile: Dict[str, Any]) -> float:
        """
        Calculate how well candidate's experience level fits job requirements
        """
        years_experience = profile.get('years_of_experience', 0)
        
        # If no specific experience requirements, use general scoring
        if not self.job_requirements.min_years_experience and not self.job_requirements.max_years_experience:
            # General experience scoring based on market standards
            if years_experience < 1:
                return 30.0  # Very junior
            elif years_experience < 3:
                return 60.0  # Junior
            elif years_experience < 8:
                return 90.0  # Sweet spot
            elif years_experience < 12:
                return 85.0  # Senior
            else:
                return 70.0  # Very senior (may be overqualified for some roles)
        
        # Score based on job requirements
        min_exp = self.job_requirements.min_years_experience or 0
        max_exp = self.job_requirements.max_years_experience or float('inf')
        
        if min_exp <= years_experience <= max_exp:
            # Perfect fit
            return 100.0
        elif years_experience < min_exp:
            # Under-qualified, but not necessarily disqualified
            gap = min_exp - years_experience
            penalty = min(gap * 15, 50)  # Up to 50 point penalty
            return max(50.0 - penalty, 10.0)
        else:
            # Over-qualified
            excess = years_experience - max_exp
            penalty = min(excess * 5, 30)  # Smaller penalty for over-qualification
            return max(100.0 - penalty, 70.0)
    
    def _calculate_industry_relevance(self, career_history: List[Dict]) -> float:
        """
        Calculate relevance of candidate's industry experience to job requirements
        """
        if not self.job_requirements.preferred_industries:
            return 75.0  # Neutral score if no specific industry preferences
        
        if not career_history:
            return 0.0
        
        total_months = 0
        relevant_months = 0
        
        for job in career_history:
            job_industry = job.get('industry', '').lower()
            duration = job.get('duration_months', 0)
            total_months += duration
            
            # Check if industry matches job preferences
            for preferred_industry in self.job_requirements.preferred_industries:
                if preferred_industry.lower() in job_industry or job_industry in preferred_industry.lower():
                    relevant_months += duration
                    break
        
        if total_months == 0:
            return 50.0
        
        relevance_ratio = relevant_months / total_months
        return min(relevance_ratio * 100, 100.0)
    
    def _calculate_company_type_fit(self, career_history: List[Dict]) -> float:
        """
        Calculate fit based on company types (product vs consulting, startup vs enterprise)
        """
        if not career_history:
            return 50.0
        
        company_type_scores = []
        
        for job in career_history:
            company_name = job.get('company', '').lower()
            duration = job.get('duration_months', 0)
            
            # Determine company type
            company_type = self._classify_company_type(company_name)
            
            # Score based on job requirements and company type
            type_score = self._score_company_type(company_type, duration)
            company_type_scores.append(type_score)
        
        if not company_type_scores:
            return 50.0
        
        # Weight by recency (more recent experience matters more)
        weighted_scores = []
        for i, score in enumerate(company_type_scores):
            # Most recent job gets full weight, older jobs get less weight
            weight = 1.0 / (i + 1)
            weighted_scores.append(score * weight)
        
        return sum(weighted_scores) / sum([1.0 / (i + 1) for i in range(len(company_type_scores))])
    
    def _classify_company_type(self, company_name: str) -> str:
        """Classify company into type categories"""
        
        for company_type, patterns in self.company_patterns.items():
            for pattern in patterns:
                if pattern in company_name:
                    return company_type
        
        return 'unknown'
    
    def _score_company_type(self, company_type: str, duration_months: int) -> float:
        """Score company type based on job requirements"""
        
        base_scores = {
            'product': 85.0,      # Generally preferred for most tech roles
            'enterprise': 90.0,   # High quality experience
            'startup': 80.0,      # Good for innovation roles
            'consulting': 60.0,   # Lower preference but not disqualifying
            'unknown': 70.0       # Neutral
        }
        
        base_score = base_scores.get(company_type, 70.0)
        
        # Bonus for longer tenure (shows stability)
        if duration_months >= 24:  # 2+ years
            base_score += 10.0
        elif duration_months >= 12:  # 1+ years
            base_score += 5.0
        
        # Adjust based on job requirements if specified
        if self.job_requirements.company_types:
            if company_type in self.job_requirements.company_types:
                base_score += 15.0  # Bonus for matching preference
        
        return min(base_score, 100.0)
    
    def _calculate_role_progression(self, career_history: List[Dict]) -> float:
        """
        Analyze career progression quality (promotions, growth, stability)
        """
        if not career_history or len(career_history) < 2:
            return 75.0  # Neutral for single job or no history
        
        progression_score = 0.0
        factors_count = 0
        
        # 1. Role level progression
        level_progression = self._analyze_level_progression(career_history)
        progression_score += level_progression
        factors_count += 1
        
        # 2. Job stability (not too many short jobs)
        stability_score = self._analyze_job_stability(career_history)
        progression_score += stability_score
        factors_count += 1
        
        # 3. Skill/domain consistency
        consistency_score = self._analyze_domain_consistency(career_history)
        progression_score += consistency_score
        factors_count += 1
        
        return progression_score / factors_count if factors_count > 0 else 75.0
    
    def _analyze_level_progression(self, career_history: List[Dict]) -> float:
        """Analyze if there's positive career progression"""
        
        # Keywords that indicate seniority levels
        level_keywords = {
            1: ['intern', 'trainee', 'associate', 'junior'],
            2: ['engineer', 'analyst', 'developer', 'executive'],
            3: ['senior', 'lead', 'principal'],
            4: ['manager', 'director', 'head'],
            5: ['vp', 'vice president', 'chief', 'cto', 'ceo']
        }
        
        job_levels = []
        for job in career_history:
            title = job.get('title', '').lower()
            level = 2  # Default level
            
            for level_num, keywords in level_keywords.items():
                if any(keyword in title for keyword in keywords):
                    level = level_num
                    break
            
            job_levels.append(level)
        
        # Check for progression
        if len(job_levels) < 2:
            return 75.0
        
        # Calculate progression trend
        progressions = 0
        regressions = 0
        
        for i in range(1, len(job_levels)):
            if job_levels[i] > job_levels[i-1]:
                progressions += 1
            elif job_levels[i] < job_levels[i-1]:
                regressions += 1
        
        # Score based on progression pattern
        if progressions > regressions:
            return 85.0  # Good progression
        elif progressions == regressions:
            return 70.0  # Neutral
        else:
            return 55.0  # Some concerns about career direction
    
    def _analyze_job_stability(self, career_history: List[Dict]) -> float:
        """Analyze job tenure patterns"""
        
        if not career_history:
            return 75.0
        
        durations = [job.get('duration_months', 0) for job in career_history]
        avg_duration = sum(durations) / len(durations)
        
        # Score based on average tenure
        if avg_duration >= 36:  # 3+ years average
            return 90.0
        elif avg_duration >= 24:  # 2+ years average
            return 80.0
        elif avg_duration >= 12:  # 1+ years average
            return 70.0
        else:
            return 50.0  # Job hopping concerns
    
    def _analyze_domain_consistency(self, career_history: List[Dict]) -> float:
        """Analyze consistency in domains/industries"""
        
        if not career_history:
            return 75.0
        
        industries = [job.get('industry', '').lower() for job in career_history]
        unique_industries = len(set([ind for ind in industries if ind]))
        total_jobs = len(career_history)
        
        # Score based on industry consistency
        if total_jobs <= 2:
            return 80.0  # Too few jobs to judge
        
        consistency_ratio = 1 - (unique_industries / total_jobs)
        return min(consistency_ratio * 100 + 50, 100.0)
    
    def _calculate_current_role_relevance(self, profile: Dict[str, Any]) -> float:
        """
        Score relevance of current role to job requirements
        """
        current_title = profile.get('current_title', '').lower()
        
        if not current_title:
            return 50.0
        
        # Check relevance to job role type
        job_role_type = self.job_requirements.role_type.lower()
        
        relevance_score = 50.0  # Base score
        
        # Direct role type match
        if job_role_type in current_title or any(word in current_title for word in job_role_type.split()):
            relevance_score += 30.0
        
        # Level appropriateness
        job_level = self.job_requirements.role_level.lower()
        if job_level in current_title:
            relevance_score += 20.0
        
        return min(relevance_score, 100.0)
    
    def _calculate_product_experience_ratio(self, career_history: List[Dict]) -> float:
        """Calculate ratio of product vs consulting experience"""
        
        if not career_history:
            return 0.0
        
        product_months = 0
        total_months = 0
        
        for job in career_history:
            company_name = job.get('company', '').lower()
            duration = job.get('duration_months', 0)
            total_months += duration
            
            company_type = self._classify_company_type(company_name)
            if company_type in ['product', 'startup', 'enterprise']:
                product_months += duration
        
        if total_months == 0:
            return 0.0
        
        return product_months / total_months
    
    def _calculate_overall_career_score(self, experience_score: float, industry_score: float,
                                      company_type_score: float, role_progression_score: float,
                                      current_role_score: float) -> float:
        """
        Calculate overall career fit score using weighted combination
        """
        
        # Dynamic weighting based on job requirements
        if self.job_requirements.min_years_experience or self.job_requirements.max_years_experience:
            # If specific experience requirements, weight experience heavily
            weights = {
                'experience': 0.30,
                'industry': 0.20,
                'company_type': 0.20,
                'role_progression': 0.15,
                'current_role': 0.15
            }
        else:
            # Balanced weighting
            weights = {
                'experience': 0.20,
                'industry': 0.25,
                'company_type': 0.25,
                'role_progression': 0.15,
                'current_role': 0.15
            }
        
        overall_score = (
            experience_score * weights['experience'] +
            industry_score * weights['industry'] +
            company_type_score * weights['company_type'] +
            role_progression_score * weights['role_progression'] +
            current_role_score * weights['current_role']
        )
        
        return min(overall_score, 100.0)
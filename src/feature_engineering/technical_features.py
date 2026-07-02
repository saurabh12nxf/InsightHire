#!/usr/bin/env python3
"""
Technical Features Extractor - Job-driven technical skill analysis
Measures how well a candidate's technical skills match specific job requirements
"""

from typing import Dict, List, Any, Optional
import re
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class TechnicalFeaturesExtractor:
    """
    Analyzes candidate technical skills against job requirements.
    No hardcoded skill lists - everything driven by job description.
    """
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
        
        # Proficiency level weights (not hardcoded business logic)
        self.proficiency_weights = {
            'expert': 1.0,
            'advanced': 0.8, 
            'intermediate': 0.6,
            'beginner': 0.3
        }
    
    def extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract technical features for a candidate against job requirements
        
        Args:
            candidate: Candidate data dictionary
            
        Returns:
            Dictionary with technical feature scores
        """
        
        # Get candidate skills
        candidate_skills = candidate.get('skills', [])
        
        # Calculate different aspects of technical fit
        skill_match_score = self._calculate_skill_match_score(candidate_skills)
        required_skills_coverage = self._calculate_required_skills_coverage(candidate_skills)
        preferred_skills_coverage = self._calculate_preferred_skills_coverage(candidate_skills)
        skill_depth_score = self._calculate_skill_depth_score(candidate_skills)
        technical_text_match = self._analyze_technical_text_match(candidate)
        
        # Overall technical fit (weighted combination)
        technical_fit_score = self._calculate_overall_technical_fit(
            skill_match_score,
            required_skills_coverage, 
            preferred_skills_coverage,
            skill_depth_score,
            technical_text_match
        )
        
        return {
            'technical_fit_score': round(technical_fit_score, 2),
            'skill_match_score': round(skill_match_score, 2),
            'required_skills_coverage': round(required_skills_coverage, 2),
            'preferred_skills_coverage': round(preferred_skills_coverage, 2),
            'skill_depth_score': round(skill_depth_score, 2),
            'technical_text_match': round(technical_text_match, 2),
            'total_relevant_skills': self._count_relevant_skills(candidate_skills),
            'has_github_mention': self._has_github_activity(candidate)
        }
    
    def _calculate_skill_match_score(self, candidate_skills: List[Dict]) -> float:
        """Calculate how well candidate skills match job requirements"""
        
        if not candidate_skills:
            return 0.0
            
        # Convert candidate skills to normalized format
        candidate_skill_dict = {}
        for skill in candidate_skills:
            skill_name = skill['name'].lower().strip()
            proficiency = skill.get('proficiency', 'intermediate').lower()
            candidate_skill_dict[skill_name] = proficiency
        
        total_score = 0.0
        max_possible_score = 0.0
        
        # Score required skills (higher weight)
        for required_skill in self.job_requirements.required_skills:
            max_possible_score += 10.0  # Required skills worth 10 points each
            
            if required_skill in candidate_skill_dict:
                proficiency = candidate_skill_dict[required_skill]
                proficiency_multiplier = self.proficiency_weights.get(proficiency, 0.5)
                total_score += 10.0 * proficiency_multiplier
        
        # Score preferred skills (lower weight)
        for preferred_skill in self.job_requirements.preferred_skills:
            max_possible_score += 5.0  # Preferred skills worth 5 points each
            
            if preferred_skill in candidate_skill_dict:
                proficiency = candidate_skill_dict[preferred_skill] 
                proficiency_multiplier = self.proficiency_weights.get(proficiency, 0.5)
                total_score += 5.0 * proficiency_multiplier
        
        # Convert to 0-100 scale
        if max_possible_score == 0:
            return 50.0  # Neutral score if no specific requirements
            
        return min((total_score / max_possible_score) * 100, 100.0)
    
    def _calculate_required_skills_coverage(self, candidate_skills: List[Dict]) -> float:
        """Calculate what percentage of required skills the candidate has"""
        
        if not self.job_requirements.required_skills:
            return 100.0  # If no required skills specified, give full score
            
        candidate_skill_names = [skill['name'].lower().strip() for skill in candidate_skills]
        
        covered_skills = 0
        for required_skill in self.job_requirements.required_skills:
            if required_skill in candidate_skill_names:
                covered_skills += 1
        
        coverage_percentage = (covered_skills / len(self.job_requirements.required_skills)) * 100
        return coverage_percentage
    
    def _calculate_preferred_skills_coverage(self, candidate_skills: List[Dict]) -> float:
        """Calculate what percentage of preferred skills the candidate has"""
        
        if not self.job_requirements.preferred_skills:
            return 0.0  # No bonus if no preferred skills specified
            
        candidate_skill_names = [skill['name'].lower().strip() for skill in candidate_skills]
        
        covered_skills = 0
        for preferred_skill in self.job_requirements.preferred_skills:
            if preferred_skill in candidate_skill_names:
                covered_skills += 1
        
        coverage_percentage = (covered_skills / len(self.job_requirements.preferred_skills)) * 100
        return coverage_percentage
    
    def _calculate_skill_depth_score(self, candidate_skills: List[Dict]) -> float:
        """
        Calculate skill depth based on proficiency levels and experience duration
        Rewards deep expertise over shallow knowledge
        """
        
        relevant_skills = []
        all_job_skills = self.job_requirements.get_all_relevant_skills()
        
        for skill in candidate_skills:
            skill_name = skill['name'].lower().strip()
            if skill_name in all_job_skills:
                relevant_skills.append(skill)
        
        if not relevant_skills:
            return 0.0
        
        total_depth_score = 0.0
        
        for skill in relevant_skills:
            proficiency = skill.get('proficiency', 'intermediate').lower()
            duration_months = skill.get('duration_months', 12)
            
            # Base proficiency score
            proficiency_score = self.proficiency_weights.get(proficiency, 0.5)
            
            # Duration bonus (up to 2x multiplier for 3+ years experience)
            duration_multiplier = min(1.0 + (duration_months / 36), 2.0)
            
            skill_depth = proficiency_score * duration_multiplier
            total_depth_score += skill_depth
        
        # Average depth score, converted to 0-100 scale
        average_depth = total_depth_score / len(relevant_skills)
        return min(average_depth * 50, 100.0)  # Scale to 0-100
    
    def _analyze_technical_text_match(self, candidate: Dict[str, Any]) -> float:
        """
        Analyze technical content in headline, summary, and job descriptions
        for job-relevant technical terms
        """
        
        # Combine all text content
        text_content = []
        
        profile = candidate.get('profile', {})
        text_content.append(profile.get('headline', ''))
        text_content.append(profile.get('summary', ''))
        
        # Add job descriptions
        for job in candidate.get('career_history', []):
            text_content.append(job.get('description', ''))
        
        full_text = ' '.join(text_content).lower()
        
        # Count mentions of job-relevant skills
        all_job_skills = self.job_requirements.get_all_relevant_skills()
        skill_mentions = 0
        
        for skill in all_job_skills:
            # Count occurrences of this skill in text
            mentions = len(re.findall(r'\b' + re.escape(skill) + r'\b', full_text))
            skill_mentions += min(mentions, 3)  # Cap at 3 mentions per skill
        
        # Convert to score (more mentions = higher score)
        if not all_job_skills:
            return 50.0  # Neutral if no specific skills to look for
            
        max_possible_mentions = len(all_job_skills) * 3
        text_match_score = (skill_mentions / max_possible_mentions) * 100
        
        return min(text_match_score, 100.0)
    
    def _calculate_overall_technical_fit(self, skill_match: float, required_coverage: float,
                                       preferred_coverage: float, skill_depth: float,
                                       text_match: float) -> float:
        """
        Calculate overall technical fit score using weighted combination
        Weights can be adjusted based on job requirements
        """
        
        # Dynamic weighting based on job requirements
        if self.job_requirements.required_skills:
            # If there are specific required skills, weight them heavily
            weights = {
                'skill_match': 0.35,
                'required_coverage': 0.30, 
                'preferred_coverage': 0.15,
                'skill_depth': 0.15,
                'text_match': 0.05
            }
        else:
            # If no specific requirements, balance across factors
            weights = {
                'skill_match': 0.25,
                'required_coverage': 0.20,
                'preferred_coverage': 0.20, 
                'skill_depth': 0.25,
                'text_match': 0.10
            }
        
        overall_score = (
            skill_match * weights['skill_match'] +
            required_coverage * weights['required_coverage'] +
            preferred_coverage * weights['preferred_coverage'] +
            skill_depth * weights['skill_depth'] +
            text_match * weights['text_match']
        )
        
        return min(overall_score, 100.0)
    
    def _count_relevant_skills(self, candidate_skills: List[Dict]) -> int:
        """Count how many job-relevant skills the candidate has"""
        
        candidate_skill_names = [skill['name'].lower().strip() for skill in candidate_skills]
        all_job_skills = self.job_requirements.get_all_relevant_skills()
        
        relevant_count = 0
        for skill_name in candidate_skill_names:
            if skill_name in all_job_skills:
                relevant_count += 1
                
        return relevant_count
    
    def _has_github_activity(self, candidate: Dict[str, Any]) -> bool:
        """Check if candidate has GitHub activity (if required by job)"""
        
        if not self.job_requirements.requires_github:
            return True  # Not required, so don't penalize
            
        github_score = candidate.get('redrob_signals', {}).get('github_activity_score', -1)
        return github_score > 0  # Has some GitHub activity
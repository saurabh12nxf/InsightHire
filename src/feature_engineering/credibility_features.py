#!/usr/bin/env python3
"""
Credibility Features Extractor - Detects inconsistencies and validates candidate claims
Identifies honeypots, skill-experience mismatches, and credibility red flags
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import sys
from pathlib import Path
import re
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class CredibilityFeaturesExtractor:
    """
    Analyzes candidate data for credibility and consistency.
    Detects honeypots, impossible claims, and skill-experience mismatches.
    """
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
        
        # Credibility validation rules (configurable)
        self.min_skill_experience_months = 6  # Minimum months to claim skill proficiency
        self.max_skill_experience_ratio = 2.0  # Max skill experience vs total experience
        
        # Suspicious patterns
        self.honeypot_indicators = [
            'test', 'fake', 'dummy', 'sample', 'example', 'placeholder'
        ]
        
        # High-skill keywords that need validation
        self.expert_skills = [
            'expert', 'advanced', 'architect', 'lead', 'senior', 'principal'
        ]
    
    def extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract credibility features for a candidate
        
        Args:
            candidate: Candidate data dictionary
            
        Returns:
            Dictionary with credibility feature scores and flags
        """
        
        profile = candidate.get('profile', {})
        skills = candidate.get('skills', [])
        career_history = candidate.get('career_history', [])
        redrob_signals = candidate.get('redrob_signals', {})
        
        # Run credibility checks
        skill_credibility = self._check_skill_credibility(skills, profile)
        experience_consistency = self._check_experience_consistency(career_history, profile)
        github_credibility = self._check_github_credibility(skills, redrob_signals)
        profile_credibility = self._check_profile_credibility(profile)
        honeypot_score = self._check_honeypot_indicators(candidate)
        
        # Overall credibility score
        credibility_score = self._calculate_overall_credibility(
            skill_credibility, experience_consistency, github_credibility,
            profile_credibility, honeypot_score
        )
        
        return {
            'credibility_score': round(credibility_score, 2),
            'skill_credibility_score': round(skill_credibility, 2),
            'experience_consistency_score': round(experience_consistency, 2),
            'github_credibility_score': round(github_credibility, 2),
            'profile_credibility_score': round(profile_credibility, 2),
            'honeypot_risk_score': round(100 - honeypot_score, 2),  # Invert for risk
            'has_credibility_flags': credibility_score < 70.0,
            'suspicious_skill_claims': self._count_suspicious_skills(skills, career_history),
            'impossible_timelines': self._detect_impossible_timelines(career_history)
        }
    
    def _check_skill_credibility(self, skills: List[Dict], profile: Dict) -> float:
        """
        Check if skill claims are credible based on experience and duration
        """
        
        if not skills:
            return 80.0  # Neutral if no skills listed
        
        total_experience_months = profile.get('years_of_experience', 0) * 12
        credible_skills = 0
        suspicious_skills = 0
        
        for skill in skills:
            skill_name = skill.get('name', '').lower()
            proficiency = skill.get('proficiency', 'intermediate').lower()
            duration_months = skill.get('duration_months', 12)
            
            # Check 1: Skill duration vs total experience
            if duration_months > total_experience_months + 6:  # 6 month buffer
                suspicious_skills += 1
                continue
            
            # Check 2: Expert claims need minimum experience
            if proficiency == 'expert' and duration_months < 24:  # 2+ years for expert
                suspicious_skills += 1
                continue
                
            # Check 3: Advanced claims need reasonable experience
            if proficiency == 'advanced' and duration_months < 12:  # 1+ year for advanced
                suspicious_skills += 1
                continue
            
            # Check 4: Very short experience with high claims
            if duration_months < self.min_skill_experience_months and proficiency in ['expert', 'advanced']:
                suspicious_skills += 1
                continue
            
            credible_skills += 1
        
        total_skills = len(skills)
        credibility_ratio = credible_skills / total_skills if total_skills > 0 else 1.0
        
        # Convert to score (higher credibility = higher score)
        credibility_score = credibility_ratio * 100
        
        # Penalty for high suspicious skill ratio
        if suspicious_skills / total_skills > 0.3:  # More than 30% suspicious
            credibility_score -= 20
        
        return max(min(credibility_score, 100.0), 0.0)
    
    def _check_experience_consistency(self, career_history: List[Dict], profile: Dict) -> float:
        """
        Check consistency between career history and profile claims
        """
        
        if not career_history:
            return 75.0  # Neutral if no career history
        
        # Calculate total experience from career history
        calculated_experience_months = sum(job.get('duration_months', 0) for job in career_history)
        calculated_years = calculated_experience_months / 12
        
        # Get claimed experience
        claimed_years = profile.get('years_of_experience', 0)
        
        # Check consistency
        if claimed_years == 0:
            return 50.0  # No claimed experience
        
        experience_ratio = calculated_years / claimed_years
        
        # Score based on consistency
        if 0.8 <= experience_ratio <= 1.2:  # Within 20% margin
            consistency_score = 90.0
        elif 0.6 <= experience_ratio <= 1.4:  # Within 40% margin
            consistency_score = 70.0
        elif 0.4 <= experience_ratio <= 1.6:  # Within 60% margin
            consistency_score = 50.0
        else:  # Major inconsistency
            consistency_score = 20.0
        
        # Check for impossible overlaps in job timeline
        overlap_penalty = self._check_job_overlaps(career_history)
        consistency_score -= overlap_penalty
        
        return max(min(consistency_score, 100.0), 0.0)
    
    def _check_job_overlaps(self, career_history: List[Dict]) -> float:
        """Check for impossible job overlaps (working multiple full-time jobs simultaneously)"""
        
        # This is a simplified check - in real implementation, would need actual dates
        # For now, assume jobs are listed chronologically and check for unrealistic short gaps
        
        if len(career_history) < 2:
            return 0.0
        
        penalty = 0.0
        
        # Look for patterns that suggest impossible timelines
        for i in range(len(career_history) - 1):
            current_job = career_history[i]
            next_job = career_history[i + 1]
            
            current_duration = current_job.get('duration_months', 0)
            next_duration = next_job.get('duration_months', 0)
            
            # Flag if someone claims very short tenures across multiple jobs
            if current_duration < 3 and next_duration < 3:  # Both less than 3 months
                penalty += 5.0
        
        return min(penalty, 30.0)  # Cap penalty at 30 points
    
    def _check_github_credibility(self, skills: List[Dict], redrob_signals: Dict) -> float:
        """
        Cross-validate GitHub activity with technical skill claims
        """
        
        github_score = redrob_signals.get('github_activity_score', -1)
        
        # Count technical skills that would typically show up on GitHub
        technical_skills = [
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 
            'vue', 'nodejs', 'django', 'flask', 'spring', 'docker', 'kubernetes'
        ]
        
        candidate_technical_skills = 0
        high_proficiency_technical = 0
        
        for skill in skills:
            skill_name = skill.get('name', '').lower()
            proficiency = skill.get('proficiency', '').lower()
            
            if any(tech_skill in skill_name for tech_skill in technical_skills):
                candidate_technical_skills += 1
                if proficiency in ['expert', 'advanced']:
                    high_proficiency_technical += 1
        
        # If no technical skills, GitHub not relevant
        if candidate_technical_skills == 0:
            return 80.0  # Neutral
        
        # If no GitHub data available
        if github_score == -1:
            # Penalty increases with technical skill claims
            if high_proficiency_technical >= 3:
                return 40.0  # Suspicious for expert technical person
            elif candidate_technical_skills >= 5:
                return 60.0  # Some concern
            else:
                return 75.0  # Less concern
        
        # Cross-validate GitHub score with technical skills
        if candidate_technical_skills >= 5 and github_score < 10:
            return 30.0  # Many technical skills but no GitHub activity
        elif candidate_technical_skills >= 3 and github_score < 5:
            return 50.0  # Some technical skills but very low GitHub
        elif high_proficiency_technical >= 2 and github_score < 20:
            return 45.0  # Claims expertise but low GitHub activity
        else:
            return 85.0  # Reasonable consistency
    
    def _check_profile_credibility(self, profile: Dict) -> float:
        """
        Check profile data for credibility red flags
        """
        
        credibility_score = 90.0  # Start high
        
        # Check for suspicious profile elements
        headline = profile.get('headline', '').lower()
        summary = profile.get('summary', '').lower()
        current_title = profile.get('current_title', '').lower()
        
        # Check for honeypot indicators in text
        for text_field in [headline, summary, current_title]:
            for indicator in self.honeypot_indicators:
                if indicator in text_field:
                    credibility_score -= 15.0
        
        # Check for unrealistic claims
        years_exp = profile.get('years_of_experience', 0)
        
        # Flag unrealistic experience for age (assume minimum age 22 at graduation)
        if years_exp > 40:  # More than 40 years experience
            credibility_score -= 20.0
        
        # Check for empty/minimal profiles
        if not headline and not summary:
            credibility_score -= 25.0
        elif len(headline) < 10 and len(summary) < 50:
            credibility_score -= 15.0
        
        # Check current industry consistency
        current_industry = profile.get('current_industry', '').lower()
        if current_industry and any(indicator in current_industry for indicator in self.honeypot_indicators):
            credibility_score -= 20.0
        
        return max(min(credibility_score, 100.0), 0.0)
    
    def _check_honeypot_indicators(self, candidate: Dict[str, Any]) -> float:
        """
        Check entire candidate profile for honeypot/fake indicators
        """
        
        honeypot_score = 90.0  # Start assuming legitimate
        
        # Check all text fields for suspicious patterns
        profile = candidate.get('profile', {})
        
        # Collect all text content
        text_fields = [
            profile.get('headline', ''),
            profile.get('summary', ''),
            profile.get('current_title', ''),
            profile.get('current_industry', '')
        ]
        
        # Add career history text
        for job in candidate.get('career_history', []):
            text_fields.extend([
                job.get('title', ''),
                job.get('company', ''),
                job.get('description', ''),
                job.get('industry', '')
            ])
        
        # Check each text field
        total_honeypot_flags = 0
        total_text_fields = len([field for field in text_fields if field.strip()])
        
        for text in text_fields:
            text_lower = text.lower()
            for indicator in self.honeypot_indicators:
                if indicator in text_lower:
                    total_honeypot_flags += 1
                    break  # One flag per text field max
        
        if total_text_fields > 0:
            honeypot_ratio = total_honeypot_flags / total_text_fields
            honeypot_score -= honeypot_ratio * 60  # Up to 60 point penalty
        
        # Additional suspicious patterns
        candidate_id = candidate.get('candidate_id', '')
        if 'test' in candidate_id.lower() or 'fake' in candidate_id.lower():
            honeypot_score -= 30.0
        
        # Check for too-perfect profiles (all fields filled with unrealistic data)
        if self._is_too_perfect_profile(candidate):
            honeypot_score -= 20.0
        
        return max(min(honeypot_score, 100.0), 0.0)
    
    def _is_too_perfect_profile(self, candidate: Dict[str, Any]) -> bool:
        """
        Detect profiles that are suspiciously complete/perfect
        """
        
        profile = candidate.get('profile', {})
        
        # Check if all optional fields are filled (suspicious for real profiles)
        optional_fields = ['headline', 'summary', 'current_industry', 'current_title']
        filled_fields = sum(1 for field in optional_fields if profile.get(field, '').strip())
        
        # Check if skills list is suspiciously long
        skills_count = len(candidate.get('skills', []))
        
        # Check if career history is suspiciously detailed
        career_count = len(candidate.get('career_history', []))
        
        # Too perfect if everything is filled and numbers are unrealistic
        return (filled_fields == len(optional_fields) and 
                skills_count > 25 and  # More than 25 skills is suspicious
                career_count > 8)      # More than 8 jobs is unusual
    
    def _count_suspicious_skills(self, skills: List[Dict], career_history: List[Dict]) -> int:
        """Count skills that have credibility issues"""
        
        suspicious_count = 0
        total_experience_years = sum(job.get('duration_months', 0) for job in career_history) / 12
        
        for skill in skills:
            proficiency = skill.get('proficiency', '').lower()
            duration_months = skill.get('duration_months', 12)
            duration_years = duration_months / 12
            
            # Suspicious if skill experience exceeds total experience
            if duration_years > total_experience_years + 1:  # 1 year buffer
                suspicious_count += 1
            # Suspicious if claiming expertise with minimal experience
            elif proficiency == 'expert' and duration_months < 18:
                suspicious_count += 1
        
        return suspicious_count
    
    def _detect_impossible_timelines(self, career_history: List[Dict]) -> int:
        """Detect impossible timeline claims"""
        
        impossible_count = 0
        
        # Check for jobs that are too short to be credible
        for job in career_history:
            duration_months = job.get('duration_months', 0)
            if duration_months < 1:  # Less than 1 month jobs are suspicious
                impossible_count += 1
        
        return impossible_count
    
    def _calculate_overall_credibility(self, skill_cred: float, exp_consistency: float,
                                     github_cred: float, profile_cred: float,
                                     honeypot_score: float) -> float:
        """
        Calculate overall credibility score using weighted combination
        """
        
        # Weights for different credibility factors
        weights = {
            'skill_credibility': 0.25,
            'experience_consistency': 0.25,
            'github_credibility': 0.20,
            'profile_credibility': 0.15,
            'honeypot_score': 0.15
        }
        
        # If job requires GitHub, weight GitHub credibility more heavily
        if self.job_requirements.requires_github:
            weights['github_credibility'] = 0.30
            weights['skill_credibility'] = 0.20
        
        credibility_score = (
            skill_cred * weights['skill_credibility'] +
            exp_consistency * weights['experience_consistency'] +
            github_cred * weights['github_credibility'] +
            profile_cred * weights['profile_credibility'] +
            honeypot_score * weights['honeypot_score']
        )
        
        return max(min(credibility_score, 100.0), 0.0)
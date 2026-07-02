#!/usr/bin/env python3
"""
Job Description Parser - Creates structured JD profiles from job requirements
Extracts comprehensive job matching criteria from job descriptions
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements


class JDProfileParser:
    """
    Creates comprehensive job description profiles for intelligent matching.
    Goes beyond basic requirements to capture nuanced matching criteria.
    """
    
    def __init__(self):
        # Industry-specific negative signals (configurable)
        self.common_negative_signals = {
            'technical': [
                'consulting only', 'non-technical', 'marketing background', 
                'sales only', 'research only', 'academic only'
            ],
            'sales': [
                'technical only', 'engineering background', 'introverted',
                'no customer interaction', 'backend only'
            ],
            'marketing': [
                'technical only', 'no creative experience', 'backend focused',
                'no customer-facing experience'
            ],
            'management': [
                'individual contributor only', 'no team experience', 
                'technical specialist only', 'junior level only'
            ]
        }
        
        # Positive signals by role type
        self.positive_signals = {
            'technical': [
                'open source contributions', 'github activity', 'technical blog',
                'hackathons', 'side projects', 'continuous learning'
            ],
            'sales': [
                'customer facing', 'quota achievement', 'relationship building',
                'negotiation skills', 'presentation skills'
            ],
            'marketing': [
                'campaign management', 'creative projects', 'brand building',
                'content creation', 'social media', 'analytics driven'
            ],
            'management': [
                'team leadership', 'mentoring', 'project delivery',
                'stakeholder management', 'strategic thinking'
            ]
        }
    
    def create_jd_profile(self, job_requirements: JobRequirements, 
                         additional_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create comprehensive JD profile from job requirements
        
        Args:
            job_requirements: JobRequirements object
            additional_preferences: Additional matching preferences
            
        Returns:
            Complete JD profile for intelligent matching
        """
        
        # Base profile from job requirements
        jd_profile = {
            'job_info': {
                'role_title': job_requirements.role_title,
                'role_level': job_requirements.role_level,
                'role_type': job_requirements.role_type,
                'created_at': datetime.now().isoformat()
            },
            
            # Core requirements
            'required_skills': job_requirements.required_skills.copy(),
            'preferred_skills': job_requirements.preferred_skills.copy(),
            
            # Experience requirements
            'experience_range': [
                job_requirements.min_years_experience or 0,
                job_requirements.max_years_experience or 20
            ],
            
            # Location and work preferences
            'preferred_locations': job_requirements.locations.copy(),
            'work_modes': job_requirements.work_modes.copy(),
            
            # Industry and company preferences
            'preferred_industries': job_requirements.preferred_industries.copy(),
            'company_types': job_requirements.company_types.copy(),
            
            # Behavioral requirements
            'behavioral_requirements': {
                'requires_github': job_requirements.requires_github,
                'max_notice_period_days': job_requirements.max_notice_period_days,
                'min_response_rate': job_requirements.min_response_rate
            }
        }
        
        # Add intelligent matching criteria
        jd_profile.update(self._generate_intelligent_criteria(job_requirements))
        
        # Add additional preferences if provided
        if additional_preferences:
            jd_profile.update(additional_preferences)
        
        return jd_profile
    
    def _generate_intelligent_criteria(self, job_requirements: JobRequirements) -> Dict[str, Any]:
        """Generate intelligent matching criteria based on role type and level"""
        
        role_type = job_requirements.role_type.lower()
        role_level = job_requirements.role_level.lower()
        
        # Generate positive signals
        positive_signals = []
        if role_type in self.positive_signals:
            positive_signals.extend(self.positive_signals[role_type])
        
        # Add level-specific positive signals
        if role_level in ['senior', 'lead', 'principal']:
            positive_signals.extend(['mentoring', 'leadership', 'architecture', 'strategic thinking'])
        elif role_level in ['junior', 'entry']:
            positive_signals.extend(['learning agility', 'fresh perspective', 'adaptability'])
        
        # Generate negative signals
        negative_signals = []
        if role_type in self.common_negative_signals:
            negative_signals.extend(self.common_negative_signals[role_type])
        
        # Add role-specific negative signals
        if job_requirements.requires_github and 'no github activity' not in negative_signals:
            negative_signals.append('no github activity')
        
        # Company type preferences
        company_preferences = self._generate_company_preferences(job_requirements)
        
        # Skill importance weighting
        skill_weights = self._generate_skill_weights(job_requirements)
        
        return {
            'positive_signals': list(set(positive_signals)),
            'negative_signals': list(set(negative_signals)),
            'company_preferences': company_preferences,
            'skill_weights': skill_weights,
            'matching_strategy': self._determine_matching_strategy(job_requirements)
        }
    
    def _generate_company_preferences(self, job_requirements: JobRequirements) -> Dict[str, float]:
        """Generate company type preferences with weights"""
        
        preferences = {
            'product_companies': 0.8,      # Generally preferred
            'startups': 0.7,               # Good for innovation
            'enterprise': 0.6,             # Stable experience
            'consulting': 0.4,             # Lower preference
            'unknown': 0.5                 # Neutral
        }
        
        # Adjust based on job requirements
        if 'startup' in job_requirements.company_types:
            preferences['startups'] = 0.9
        if 'enterprise' in job_requirements.company_types:
            preferences['enterprise'] = 0.9
        if 'consulting' in job_requirements.company_types:
            preferences['consulting'] = 0.7
        
        return preferences
    
    def _generate_skill_weights(self, job_requirements: JobRequirements) -> Dict[str, float]:
        """Generate skill importance weights"""
        
        skill_weights = {}
        
        # Required skills get higher weights
        for skill in job_requirements.required_skills:
            skill_weights[skill] = 1.0
        
        # Preferred skills get medium weights  
        for skill in job_requirements.preferred_skills:
            skill_weights[skill] = 0.6
        
        # Add domain-specific skill weights
        if job_requirements.role_type == 'technical':
            # Technical roles - programming languages are critical
            programming_langs = ['python', 'java', 'javascript', 'typescript', 'go', 'rust']
            for lang in programming_langs:
                if lang in skill_weights:
                    skill_weights[lang] *= 1.2  # 20% boost
        
        return skill_weights
    
    def _determine_matching_strategy(self, job_requirements: JobRequirements) -> Dict[str, float]:
        """Determine matching strategy weights for different scoring components"""
        
        # Base strategy
        strategy = {
            'technical_weight': 0.25,
            'career_weight': 0.25, 
            'behavioral_weight': 0.25,
            'trajectory_weight': 0.15,
            'credibility_weight': 0.10
        }
        
        # Adjust based on role type
        if job_requirements.role_type == 'technical':
            strategy['technical_weight'] = 0.35
            strategy['behavioral_weight'] = 0.20
        elif job_requirements.role_type in ['sales', 'marketing']:
            strategy['behavioral_weight'] = 0.35
            strategy['technical_weight'] = 0.15
        elif job_requirements.role_type == 'management':
            strategy['trajectory_weight'] = 0.25
            strategy['behavioral_weight'] = 0.30
        
        # Adjust based on role level
        if job_requirements.role_level in ['senior', 'lead', 'principal']:
            strategy['trajectory_weight'] += 0.10
            strategy['credibility_weight'] += 0.05
        
        # Normalize weights to sum to 1.0
        total_weight = sum(strategy.values())
        for key in strategy:
            strategy[key] = strategy[key] / total_weight
        
        return strategy
    
    def save_jd_profile(self, jd_profile: Dict[str, Any], 
                       output_path: str = 'Data/processed/jd_profile.json'):
        """Save JD profile to file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(jd_profile, f, indent=2, ensure_ascii=False)
        
        print(f"JD Profile saved to: {output_file}")
        return output_file
    
    def load_jd_profile(self, profile_path: str = 'Data/processed/jd_profile.json') -> Dict[str, Any]:
        """Load JD profile from file"""
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            jd_profile = json.load(f)
        
        return jd_profile
    
    def print_jd_profile_summary(self, jd_profile: Dict[str, Any]):
        """Print human-readable summary of JD profile"""
        
        print("=" * 60)
        print("JOB DESCRIPTION PROFILE SUMMARY") 
        print("=" * 60)
        
        job_info = jd_profile.get('job_info', {})
        print(f"Role: {job_info.get('role_title', 'Unknown')}")
        print(f"Level: {job_info.get('role_level', 'Unknown')}")
        print(f"Type: {job_info.get('role_type', 'Unknown')}")
        
        print(f"\n🔧 REQUIRED SKILLS:")
        for skill in jd_profile.get('required_skills', []):
            weight = jd_profile.get('skill_weights', {}).get(skill, 1.0)
            print(f"   • {skill} (weight: {weight:.1f})")
        
        print(f"\n⭐ PREFERRED SKILLS:")
        for skill in jd_profile.get('preferred_skills', []):
            weight = jd_profile.get('skill_weights', {}).get(skill, 0.6)
            print(f"   • {skill} (weight: {weight:.1f})")
        
        exp_range = jd_profile.get('experience_range', [0, 20])
        print(f"\n📅 EXPERIENCE: {exp_range[0]}-{exp_range[1]} years")
        
        print(f"\n✅ POSITIVE SIGNALS:")
        for signal in jd_profile.get('positive_signals', []):
            print(f"   • {signal}")
        
        print(f"\n❌ NEGATIVE SIGNALS:")
        for signal in jd_profile.get('negative_signals', []):
            print(f"   • {signal}")
        
        print(f"\n🏢 COMPANY PREFERENCES:")
        for company_type, weight in jd_profile.get('company_preferences', {}).items():
            print(f"   • {company_type}: {weight:.1f}")
        
        strategy = jd_profile.get('matching_strategy', {})
        print(f"\n⚖️  MATCHING STRATEGY:")
        for component, weight in strategy.items():
            print(f"   • {component}: {weight:.2f}")
        
        behavioral = jd_profile.get('behavioral_requirements', {})
        print(f"\n👤 BEHAVIORAL REQUIREMENTS:")
        print(f"   • GitHub Required: {behavioral.get('requires_github', False)}")
        print(f"   • Max Notice Period: {behavioral.get('max_notice_period_days', 'N/A')} days")
        print(f"   • Min Response Rate: {behavioral.get('min_response_rate', 'N/A')}")
        
        print("=" * 60)


# Factory functions for common job types
def create_ai_engineer_profile() -> Dict[str, Any]:
    """Create AI Engineer JD profile"""
    
    from src.job_requirements.job_schema import get_ai_engineer_template
    
    job_requirements = get_ai_engineer_template()
    parser = JDProfileParser()
    
    # Add AI-specific preferences
    ai_preferences = {
        'positive_signals': [
            'open source contributions', 'research papers', 'kaggle competitions',
            'machine learning projects', 'ai/ml blog posts', 'conference talks'
        ],
        'negative_signals': [
            'no technical background', 'consulting only', 'marketing focus',
            'no programming experience', 'research only without implementation'
        ],
        'skill_priorities': {
            'python': 1.0,
            'machine learning': 1.0, 
            'tensorflow': 0.8,
            'pytorch': 0.8,
            'docker': 0.6,
            'aws': 0.6
        }
    }
    
    jd_profile = parser.create_jd_profile(job_requirements, ai_preferences)
    return jd_profile


def create_business_analyst_profile() -> Dict[str, Any]:
    """Create Business Analyst JD profile"""
    
    from src.job_requirements.job_schema import JobRequirements
    
    job_requirements = JobRequirements(
        role_title="Business Analyst",
        role_level="mid",
        role_type="analyst",
        required_skills=["analysis", "excel", "reporting", "sql"],
        preferred_skills=["tableau", "power bi", "python", "project management"],
        min_years_experience=2,
        max_years_experience=8,
        preferred_industries=["technology", "finance", "consulting"]
    )
    
    parser = JDProfileParser()
    
    ba_preferences = {
        'positive_signals': [
            'stakeholder management', 'requirements gathering', 'process improvement',
            'data visualization', 'business intelligence', 'cross-functional collaboration'
        ],
        'negative_signals': [
            'purely technical background', 'no business interaction', 'backend only',
            'no analytical experience'
        ]
    }
    
    jd_profile = parser.create_jd_profile(job_requirements, ba_preferences)
    return jd_profile


if __name__ == "__main__":
    # Test JD Profile creation
    print("Testing JD Profile Parser...")
    
    # Create AI Engineer profile
    ai_profile = create_ai_engineer_profile()
    
    # Initialize parser  
    parser = JDProfileParser()
    
    # Save profile
    profile_path = parser.save_jd_profile(ai_profile)
    
    # Print summary
    parser.print_jd_profile_summary(ai_profile)
    
    print(f"\n✅ JD Profile created and saved to: {profile_path}")
    print("\nCheckpoint 1: Review the profile above.")
    print("Ask yourself: If I were a recruiter, is this exactly what I want?")
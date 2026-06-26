#!/usr/bin/env python3
"""
Job Requirements Schema - Universal job description data structure
This defines how we store job requirements in a flexible, non-hardcoded way
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json


@dataclass
class JobRequirements:
    """
    Universal job requirements structure that works for ANY role.
    No hardcoded skills or assumptions about job type.
    """
    
    # Core Requirements
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    
    # Experience Requirements  
    min_years_experience: Optional[int] = None
    max_years_experience: Optional[int] = None
    
    # Role Information
    role_title: str = ""
    role_level: str = ""  # Junior, Mid, Senior, Lead, Principal
    role_type: str = ""   # Technical, Management, Sales, Marketing, etc.
    
    # Industry & Company Preferences
    preferred_industries: List[str] = field(default_factory=list)
    company_types: List[str] = field(default_factory=list)  # startup, enterprise, consulting
    
    # Location & Work Preferences
    locations: List[str] = field(default_factory=list)
    work_modes: List[str] = field(default_factory=list)  # remote, hybrid, onsite
    
    # Behavioral Preferences
    max_notice_period_days: Optional[int] = None
    requires_github: bool = False
    min_response_rate: Optional[float] = None
    
    def __post_init__(self):
        """Validate and normalize the job requirements after creation"""
        # Normalize skills to lowercase for consistent matching
        self.required_skills = [skill.lower().strip() for skill in self.required_skills]
        self.preferred_skills = [skill.lower().strip() for skill in self.preferred_skills]
        
        # Normalize role level
        if self.role_level:
            self.role_level = self.role_level.lower().strip()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobRequirements':
        """
        Create JobRequirements from dictionary (e.g., loaded from JSON/YAML)
        
        Example usage:
        job_data = {
            'required_skills': ['python', 'machine learning'],
            'min_years_experience': 3,
            'role_level': 'senior'
        }
        requirements = JobRequirements.from_dict(job_data)
        """
        return cls(**data)
    
    @classmethod  
    def from_json_file(cls, filepath: str) -> 'JobRequirements':
        """Load job requirements from JSON configuration file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving/serialization"""
        return {
            'required_skills': self.required_skills,
            'preferred_skills': self.preferred_skills,
            'min_years_experience': self.min_years_experience,
            'max_years_experience': self.max_years_experience,
            'role_title': self.role_title,
            'role_level': self.role_level,
            'role_type': self.role_type,
            'preferred_industries': self.preferred_industries,
            'company_types': self.company_types,
            'locations': self.locations,
            'work_modes': self.work_modes,
            'max_notice_period_days': self.max_notice_period_days,
            'requires_github': self.requires_github,
            'min_response_rate': self.min_response_rate
        }
    
    def get_all_relevant_skills(self) -> List[str]:
        """Get combined list of required + preferred skills"""
        return list(set(self.required_skills + self.preferred_skills))
    
    def is_skill_required(self, skill: str) -> bool:
        """Check if a skill is in the required list"""
        return skill.lower().strip() in self.required_skills
    
    def is_skill_preferred(self, skill: str) -> bool:
        """Check if a skill is in the preferred list"""
        return skill.lower().strip() in self.preferred_skills
    
    def is_experience_in_range(self, years: float) -> bool:
        """Check if candidate experience falls within job requirements"""
        if self.min_years_experience and years < self.min_years_experience:
            return False
        if self.max_years_experience and years > self.max_years_experience:
            return False
        return True


# Predefined job configurations for common roles
# These are NOT hardcoded - they're just starting templates that can be modified

def get_ai_engineer_template() -> JobRequirements:
    """
    Template for AI Engineer role - can be customized for specific positions
    This is a starting point, not a hardcoded requirement
    """
    return JobRequirements(
        role_title="AI Engineer",
        role_level="mid",
        role_type="technical", 
        required_skills=["python", "machine learning"],
        preferred_skills=["tensorflow", "pytorch", "docker", "aws"],
        min_years_experience=3,
        max_years_experience=8,
        preferred_industries=["technology", "software", "ai"],
        requires_github=True,
        max_notice_period_days=60
    )

def get_marketing_manager_template() -> JobRequirements:
    """Template for Marketing Manager role"""
    return JobRequirements(
        role_title="Marketing Manager",
        role_level="mid",
        role_type="management",
        required_skills=["marketing strategy", "campaign management", "analytics"],
        preferred_skills=["google analytics", "facebook ads", "content marketing"],
        min_years_experience=4,
        max_years_experience=10,
        preferred_industries=["technology", "e-commerce", "saas"],
        requires_github=False
    )

def get_sales_executive_template() -> JobRequirements:
    """Template for Sales Executive role"""
    return JobRequirements(
        role_title="Sales Executive", 
        role_level="mid",
        role_type="sales",
        required_skills=["sales", "crm", "lead generation"],
        preferred_skills=["salesforce", "hubspot", "cold calling"],
        min_years_experience=2,
        max_years_experience=6,
        preferred_industries=["software", "saas", "technology"],
        requires_github=False,
        min_response_rate=0.6  # Sales people should be responsive
    )
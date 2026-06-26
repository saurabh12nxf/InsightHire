#!/usr/bin/env python3
"""
Job Description Parser - Extract requirements from free-form job descriptions
This module converts unstructured job descriptions into structured JobRequirements
"""

import re
from typing import List, Dict, Optional, Tuple
from .job_schema import JobRequirements


class JobDescriptionParser:
    """
    Parses free-form job description text into structured JobRequirements.
    Uses pattern matching and keyword extraction - no hardcoded assumptions about job types.
    """
    
    def __init__(self):
        # Common patterns for extracting information
        self.experience_patterns = [
            r'(\d+)[\+\-\s]*(?:to\s+(\d+))?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'(\d+)[\+\-\s]*(?:to\s+(\d+))?\s*(?:years?|yrs?)',
            r'minimum\s+(\d+)\s+(?:years?|yrs?)',
            r'at least\s+(\d+)\s+(?:years?|yrs?)'
        ]
        
        self.level_keywords = {
            'junior': ['junior', 'entry level', 'associate', 'trainee', 'graduate'],
            'mid': ['mid level', 'experienced', 'professional'], 
            'senior': ['senior', 'lead', 'principal', 'staff'],
            'executive': ['director', 'vp', 'head of', 'chief', 'manager']
        }
        
        self.role_type_keywords = {
            'technical': ['engineer', 'developer', 'programmer', 'architect', 'scientist'],
            'management': ['manager', 'director', 'head', 'lead', 'supervisor'],
            'sales': ['sales', 'account', 'business development', 'revenue'],
            'marketing': ['marketing', 'brand', 'campaign', 'content', 'social media'],
            'operations': ['operations', 'ops', 'process', 'logistics'],
            'hr': ['human resources', 'hr', 'recruitment', 'talent']
        }
        
    def parse(self, job_description: str, role_title: str = "") -> JobRequirements:
        """
        Parse a job description text into structured requirements
        
        Args:
            job_description: Free-form job description text
            role_title: Job title (if known)
            
        Returns:
            JobRequirements object with extracted information
        """
        
        # Clean the text
        text = self._clean_text(job_description)
        
        # Extract different components
        requirements = JobRequirements()
        requirements.role_title = role_title
        
        # Extract skills
        requirements.required_skills, requirements.preferred_skills = self._extract_skills(text)
        
        # Extract experience requirements
        requirements.min_years_experience, requirements.max_years_experience = self._extract_experience(text)
        
        # Extract role information
        requirements.role_level = self._extract_role_level(text, role_title)
        requirements.role_type = self._extract_role_type(text, role_title)
        
        # Extract preferences
        requirements.preferred_industries = self._extract_industries(text)
        requirements.company_types = self._extract_company_types(text)
        requirements.work_modes = self._extract_work_modes(text)
        
        # Extract behavioral requirements
        requirements.requires_github = self._requires_github(text)
        requirements.max_notice_period_days = self._extract_notice_period(text)
        
        return requirements
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        # Convert to lowercase for consistent matching
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\+\(\)]', ' ', text)
        
        return text.strip()
    
    def _extract_skills(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract required and preferred skills from job description
        Uses pattern matching to identify skill sections and requirements
        """
        required_skills = []
        preferred_skills = []
        
        # Look for skill sections
        skill_sections = [
            'requirements:', 'required skills:', 'must have:',
            'qualifications:', 'essential skills:', 'technical skills:'
        ]
        
        preferred_sections = [
            'preferred:', 'nice to have:', 'bonus:', 'plus:', 
            'additional:', 'desired:', 'preferred skills:'
        ]
        
        # Split text into sections
        lines = text.split('\n')
        current_section = 'general'
        
        for line in lines:
            line = line.strip()
            
            # Check if this line indicates a new section
            line_lower = line.lower()
            if any(section in line_lower for section in skill_sections):
                current_section = 'required'
                continue
            elif any(section in line_lower for section in preferred_sections):
                current_section = 'preferred'
                continue
                
            # Extract skills from the line
            skills_in_line = self._extract_skills_from_line(line)
            
            if current_section == 'required':
                required_skills.extend(skills_in_line)
            elif current_section == 'preferred':
                preferred_skills.extend(skills_in_line)
            else:
                # If no specific section, treat as required if it looks important
                if self._looks_like_requirement(line):
                    required_skills.extend(skills_in_line)
                else:
                    preferred_skills.extend(skills_in_line)
        
        # Also look for skills mentioned throughout the text
        all_text_skills = self._extract_skills_from_text(text)
        
        # Merge and deduplicate
        all_required = list(set(required_skills + all_text_skills[:5]))  # Top mentioned skills as required
        all_preferred = list(set(preferred_skills + all_text_skills[5:]))  # Others as preferred
        
        return all_required[:10], all_preferred[:15]  # Limit to reasonable numbers
    
    def _extract_skills_from_line(self, line: str) -> List[str]:
        """Extract individual skills from a line of text"""
        skills = []
        
        # Common skill patterns
        skill_patterns = [
            # Programming languages
            r'\b(?:python|java|javascript|typescript|c\+\+|c#|go|rust|scala)\b',
            # Frameworks and libraries
            r'\b(?:react|angular|vue|django|flask|spring|tensorflow|pytorch|pandas|numpy)\b',
            # Databases
            r'\b(?:mysql|postgresql|mongodb|redis|elasticsearch|sql)\b',
            # Cloud and DevOps
            r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|git)\b',
            # Data and ML
            r'\b(?:machine learning|deep learning|nlp|computer vision|data science|analytics)\b',
            # General business skills
            r'\b(?:project management|agile|scrum|leadership|communication|marketing|sales)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            skills.extend([match.lower() for match in matches])
            
        return list(set(skills))
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills mentioned throughout the entire text"""
        # This would use more sophisticated NLP in a production system
        # For now, use pattern matching for common skills
        
        skills_found = []
        
        # Comprehensive skill dictionary (this could be loaded from a file)
        skill_dictionary = [
            # Programming
            'python', 'java', 'javascript', 'typescript', 'sql', 'c++', 'c#',
            # ML/AI
            'machine learning', 'deep learning', 'nlp', 'computer vision', 
            'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
            # Web
            'react', 'angular', 'vue', 'nodejs', 'django', 'flask',
            # Cloud
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            # Data
            'sql', 'mongodb', 'redis', 'elasticsearch', 'bigquery',
            # Business
            'project management', 'agile', 'scrum', 'marketing', 'sales'
        ]
        
        for skill in skill_dictionary:
            if skill in text:
                skills_found.append(skill)
                
        return skills_found
    
    def _looks_like_requirement(self, line: str) -> bool:
        """Determine if a line contains a requirement vs preference"""
        requirement_indicators = ['must', 'required', 'need', 'essential', 'minimum']
        return any(indicator in line.lower() for indicator in requirement_indicators)
    
    def _extract_experience(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract experience requirements from text"""
        min_exp = None
        max_exp = None
        
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Range pattern (e.g., "3-5 years")
                    if match[0]:
                        min_exp = int(match[0])
                    if match[1]:
                        max_exp = int(match[1])
                else:
                    # Single number (e.g., "3+ years")
                    min_exp = int(match)
                    
                break  # Use first match found
                
        return min_exp, max_exp
    
    def _extract_role_level(self, text: str, role_title: str) -> str:
        """Extract role level (junior, mid, senior, etc.)"""
        full_text = f"{role_title} {text}".lower()
        
        for level, keywords in self.level_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                return level
                
        # If no explicit level found, infer from experience
        min_exp, _ = self._extract_experience(text)
        if min_exp:
            if min_exp <= 2:
                return 'junior'
            elif min_exp <= 5:
                return 'mid' 
            else:
                return 'senior'
                
        return 'mid'  # Default
    
    def _extract_role_type(self, text: str, role_title: str) -> str:
        """Extract role type (technical, management, sales, etc.)"""
        full_text = f"{role_title} {text}".lower()
        
        for role_type, keywords in self.role_type_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                return role_type
                
        return 'general'  # Default
    
    def _extract_industries(self, text: str) -> List[str]:
        """Extract preferred industries"""
        industries = []
        
        industry_keywords = {
            'technology': ['tech', 'technology', 'software', 'it'],
            'finance': ['finance', 'fintech', 'banking', 'investment'],
            'healthcare': ['healthcare', 'medical', 'pharma', 'biotech'],
            'e-commerce': ['e-commerce', 'ecommerce', 'retail', 'marketplace'],
            'consulting': ['consulting', 'advisory', 'professional services']
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                industries.append(industry)
                
        return industries
    
    def _extract_company_types(self, text: str) -> List[str]:
        """Extract company type preferences"""
        company_types = []
        
        if any(word in text for word in ['startup', 'early stage', 'fast-paced']):
            company_types.append('startup')
        if any(word in text for word in ['enterprise', 'large company', 'corporation']):
            company_types.append('enterprise')
        if any(word in text for word in ['consulting', 'client work', 'projects']):
            company_types.append('consulting')
            
        return company_types
    
    def _extract_work_modes(self, text: str) -> List[str]:
        """Extract work mode preferences"""
        work_modes = []
        
        if 'remote' in text:
            work_modes.append('remote')
        if 'hybrid' in text:
            work_modes.append('hybrid')  
        if any(word in text for word in ['onsite', 'office', 'in-person']):
            work_modes.append('onsite')
            
        return work_modes
    
    def _requires_github(self, text: str) -> bool:
        """Check if GitHub is required"""
        github_indicators = ['github', 'git', 'portfolio', 'code samples']
        return any(indicator in text for indicator in github_indicators)
    
    def _extract_notice_period(self, text: str) -> Optional[int]:
        """Extract maximum acceptable notice period"""
        # Look for notice period requirements
        notice_patterns = [
            r'(?:notice period|availability).*?(\d+)\s*(?:days?|weeks?|months?)',
            r'(?:join|start).*?(\d+)\s*(?:days?|weeks?)',
            r'immediate.*?(?:joiner|start|availability)'
        ]
        
        for pattern in notice_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                if 'immediate' in matches.group(0).lower():
                    return 0
                else:
                    days = int(matches.group(1))
                    # Convert to days if needed
                    if 'week' in matches.group(0).lower():
                        days *= 7
                    elif 'month' in matches.group(0).lower():
                        days *= 30
                    return days
                    
        return None  # No specific requirement found
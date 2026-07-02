#!/usr/bin/env python3
"""
Test script for the feature engineering pipeline
This script demonstrates how to use the feature extraction system
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements, get_ai_engineer_template
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.data_loader.candidate_loader import CandidateLoader

def test_single_candidate():
    """Test feature extraction on a single candidate"""
    
    print("="*60)
    print("TESTING SINGLE CANDIDATE FEATURE EXTRACTION")
    print("="*60)
    
    # Create job requirements
    job_requirements = get_ai_engineer_template()
    print(f"Job: {job_requirements.role_title} - {job_requirements.role_level}")
    print(f"Required skills: {job_requirements.required_skills}")
    
    # Initialize pipeline
    pipeline = FeaturePipeline(job_requirements, include_embeddings=False)  # Skip embeddings for speed
    
    # Create a sample candidate
    sample_candidate = {
        "candidate_id": "TEST_CAND_001",
        "profile": {
            "headline": "Senior AI Engineer with 5 years experience in ML and Python",
            "summary": "Experienced AI engineer specializing in machine learning, deep learning, and Python development. Built recommendation systems and computer vision applications.",
            "years_of_experience": 5.2,
            "current_title": "Senior AI Engineer",
            "current_industry": "Technology"
        },
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 48},
            {"name": "Machine Learning", "proficiency": "advanced", "duration_months": 36},
            {"name": "TensorFlow", "proficiency": "advanced", "duration_months": 24},
            {"name": "Docker", "proficiency": "intermediate", "duration_months": 18},
            {"name": "AWS", "proficiency": "intermediate", "duration_months": 12}
        ],
        "career_history": [
            {
                "title": "Senior AI Engineer",
                "company": "TechCorp",
                "duration_months": 18,
                "industry": "Technology",
                "description": "Led development of recommendation systems using Python and TensorFlow"
            },
            {
                "title": "AI Engineer", 
                "company": "DataStart",
                "duration_months": 24,
                "industry": "Technology",
                "description": "Built computer vision models and ML pipelines"
            },
            {
                "title": "Software Engineer",
                "company": "WebCorp", 
                "duration_months": 20,
                "industry": "Technology",
                "description": "Full-stack development with focus on backend systems"
            }
        ],
        "redrob_signals": {
            "open_to_work": True,
            "notice_period_days": 30,
            "last_active_days": 5,
            "recruiter_response_rate": 0.75,
            "profile_views_last_month": 25,
            "saved_by_recruiters": 8,
            "github_activity_score": 65,
            "interview_completion_rate": 0.85,
            "offer_acceptance_rate": 0.80,
            "profile_completeness_percentage": 0.90
        }
    }
    
    # Extract features
    print("\nExtracting features...")
    features = pipeline.extract_candidate_features(sample_candidate)
    
    # Display results
    print("\n" + "="*40)
    print("FEATURE EXTRACTION RESULTS")
    print("="*40)
    
    print(f"Candidate ID: {features['candidate_id']}")
    print(f"\nMAIN SCORES:")
    print(f"  Overall Score:        {features['overall_candidate_score']:6.1f}")
    print(f"  Technical Fit:        {features['technical_fit_score']:6.1f}")
    print(f"  Career Score:         {features['career_score']:6.1f}")
    print(f"  Behavioral Score:     {features['behavioral_score']:6.1f}")
    print(f"  Trajectory Score:     {features['trajectory_score']:6.1f}")
    print(f"  Credibility Score:    {features['credibility_score']:6.1f}")
    
    print(f"\nCOMPOSITE SCORES:")
    print(f"  Job Fit Score:        {features['job_fit_score']:6.1f}")
    print(f"  Hiring Likelihood:    {features['hiring_likelihood_score']:6.1f}")
    print(f"  Quality Score:        {features['candidate_quality_score']:6.1f}")
    
    print(f"\nDETAILS:")
    print(f"  Required Skills Coverage:    {features['required_skills_coverage']:6.1f}%")
    print(f"  GitHub Activity:             {features['github_activity_score']:6.1f}")
    print(f"  Availability Score:          {features['availability_score']:6.1f}")
    print(f"  Years Experience:            {features['total_years_experience']:6.1f}")
    print(f"  Average Job Duration:        {features['average_job_duration_months']:6.1f} months")

def test_multiple_candidates():
    """Test with multiple candidates from the dataset"""
    
    print("\n" + "="*60)
    print("TESTING MULTIPLE CANDIDATES FROM DATASET")
    print("="*60)
    
    # Load some candidates
    try:
        loader = CandidateLoader('Data/raw/candidates.jsonl')
        candidates = loader.load_candidates(limit=50)  # Load first 50 for testing
        print(f"Loaded {len(candidates)} candidates for testing")
    except Exception as e:
        print(f"Could not load candidates: {e}")
        return
    
    # Create job requirements
    job_requirements = get_ai_engineer_template()
    
    # Initialize pipeline
    pipeline = FeaturePipeline(job_requirements, include_embeddings=False)
    
    # Process candidates
    print("Processing candidates...")
    features_df = pipeline.process_candidates_batch(candidates, batch_size=25)
    
    print(f"\nProcessed {len(features_df)} candidates successfully")
    
    # Show top candidates
    print("\nTop 10 candidates by overall score:")
    top_10 = features_df.nlargest(10, 'overall_candidate_score')
    
    for i, (_, candidate) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {candidate['candidate_id']:15s} | Overall: {candidate['overall_candidate_score']:5.1f} | "
              f"Technical: {candidate['technical_fit_score']:5.1f} | Behavioral: {candidate['behavioral_score']:5.1f}")

def test_job_flexibility():
    """Test with different job types to show flexibility"""
    
    print("\n" + "="*60)
    print("TESTING JOB TYPE FLEXIBILITY")
    print("="*60)
    
    # Test different job types
    job_types = [
        JobRequirements(
            role_title="Marketing Manager",
            role_type="marketing",
            role_level="mid",
            required_skills=["marketing strategy", "campaign management", "analytics"],
            preferred_skills=["google analytics", "facebook ads", "content marketing"],
            min_years_experience=3,
            max_years_experience=8
        ),
        JobRequirements(
            role_title="Sales Executive", 
            role_type="sales",
            role_level="mid",
            required_skills=["sales", "crm", "lead generation"],
            preferred_skills=["salesforce", "hubspot", "cold calling"],
            min_years_experience=2,
            max_years_experience=6,
            min_response_rate=0.6
        )
    ]
    
    # Sample candidate that might fit different roles
    versatile_candidate = {
        "candidate_id": "VERSATILE_001",
        "profile": {
            "headline": "Marketing Professional with Sales Experience",
            "summary": "Marketing manager with strong sales background and analytics skills",
            "years_of_experience": 4.5,
            "current_title": "Marketing Manager",
            "current_industry": "Technology"
        },
        "skills": [
            {"name": "Marketing Strategy", "proficiency": "advanced", "duration_months": 30},
            {"name": "Sales", "proficiency": "intermediate", "duration_months": 24},
            {"name": "Google Analytics", "proficiency": "intermediate", "duration_months": 18},
            {"name": "CRM", "proficiency": "intermediate", "duration_months": 20}
        ],
        "career_history": [
            {
                "title": "Marketing Manager",
                "company": "TechStart",
                "duration_months": 18,
                "industry": "Technology",
                "description": "Led digital marketing campaigns and analytics"
            }
        ],
        "redrob_signals": {
            "open_to_work": True,
            "notice_period_days": 45,
            "recruiter_response_rate": 0.70,
            "github_activity_score": -1,  # No GitHub (appropriate for marketing)
            "profile_completeness_percentage": 0.85
        }
    }
    
    # Test against each job type
    for job_req in job_types:
        print(f"\nTesting against: {job_req.role_title}")
        pipeline = FeaturePipeline(job_req, include_embeddings=False)
        features = pipeline.extract_candidate_features(versatile_candidate)
        
        print(f"  Overall Score:     {features['overall_candidate_score']:6.1f}")
        print(f"  Technical Fit:     {features['technical_fit_score']:6.1f}")
        print(f"  Career Score:      {features['career_score']:6.1f}")
        print(f"  Skills Coverage:   {features['required_skills_coverage']:6.1f}%")

if __name__ == "__main__":
    print("FEATURE ENGINEERING PIPELINE TEST")
    print("This script tests the job-driven feature extraction system")
    
    # Test 1: Single candidate
    test_single_candidate()
    
    # Test 2: Multiple candidates (if dataset available)
    test_multiple_candidates()
    
    # Test 3: Different job types
    test_job_flexibility()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print("\nTo use the system:")
    print("1. Define your job requirements using JobRequirements class")
    print("2. Initialize FeaturePipeline with your requirements")
    print("3. Use extract_candidate_features() for single candidates")
    print("4. Use process_candidates_batch() for multiple candidates")
    print("5. Results are saved as Parquet files for further analysis")
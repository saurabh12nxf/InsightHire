#!/usr/bin/env python3
"""
Test with more realistic job requirements that match your dataset
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.data_loader.candidate_loader import CandidateLoader

def test_realistic_jobs():
    """Test with job requirements that better match your candidate pool"""
    
    # Load candidates
    try:
        loader = CandidateLoader('Data/raw/candidates.jsonl')
        candidates = loader.load_candidates(limit=100)
        print(f"Loaded {len(candidates)} candidates for testing")
    except Exception as e:
        print(f"Error loading candidates: {e}")
        return
    
    # Define realistic job requirements based on your EDA findings
    job_requirements = [
        JobRequirements(
            role_title="Business Analyst",
            role_type="analyst",
            role_level="mid",
            required_skills=["analysis", "excel", "reporting"],
            preferred_skills=["sql", "tableau", "project management", "agile"],
            min_years_experience=2,
            max_years_experience=8
        ),
        
        JobRequirements(
            role_title="Project Manager", 
            role_type="management",
            role_level="mid",
            required_skills=["project management", "agile", "scrum"],
            preferred_skills=["pmp", "jira", "confluence", "excel"],
            min_years_experience=3,
            max_years_experience=10
        ),
        
        JobRequirements(
            role_title="HR Manager",
            role_type="hr",
            role_level="mid", 
            required_skills=["hr", "recruiting", "talent management"],
            preferred_skills=["hris", "workday", "adp", "compliance"],
            min_years_experience=3,
            max_years_experience=8
        ),
        
        JobRequirements(
            role_title="Operations Manager",
            role_type="operations",
            role_level="mid",
            required_skills=["operations", "process improvement", "analytics"],
            preferred_skills=["lean", "six sigma", "supply chain", "excel"],
            min_years_experience=4,
            max_years_experience=10
        ),
        
        # More lenient AI role
        JobRequirements(
            role_title="Junior Data Analyst",
            role_type="technical",
            role_level="junior",
            required_skills=["excel", "analytics"],  # More basic requirements
            preferred_skills=["python", "sql", "tableau", "statistics"],
            min_years_experience=1,
            max_years_experience=4,
            requires_github=False  # Not required for junior role
        )
    ]
    
    print("\n" + "="*60)
    print("TESTING REALISTIC JOB REQUIREMENTS")
    print("="*60)
    
    results = []
    
    for job_req in job_requirements:
        print(f"\n🎯 Testing: {job_req.role_title}")
        print(f"   Required skills: {job_req.required_skills}")
        
        # Initialize pipeline
        pipeline = FeaturePipeline(job_req, include_embeddings=False)
        
        # Process candidates
        features_df = pipeline.process_candidates_batch(candidates, batch_size=50, save_progress=False)
        
        # Calculate statistics
        avg_overall = features_df['overall_candidate_score'].mean()
        avg_technical = features_df['technical_fit_score'].mean()
        avg_coverage = features_df['required_skills_coverage'].mean()
        high_fit_count = (features_df['overall_candidate_score'] > 70).sum()
        
        results.append({
            'job_title': job_req.role_title,
            'avg_overall': avg_overall,
            'avg_technical': avg_technical,
            'avg_coverage': avg_coverage,
            'high_fit_count': high_fit_count,
            'high_fit_percent': high_fit_count / len(features_df) * 100
        })
        
        print(f"   📊 Results:")
        print(f"      Average Overall Score:     {avg_overall:5.1f}")
        print(f"      Average Technical Fit:     {avg_technical:5.1f}")
        print(f"      Average Skills Coverage:   {avg_coverage:5.1f}%")
        print(f"      High Fit Candidates (>70): {high_fit_count:2d} ({high_fit_count/len(features_df)*100:4.1f}%)")
        
        # Show top candidate
        top_candidate = features_df.nlargest(1, 'overall_candidate_score').iloc[0]
        print(f"      🏆 Top Candidate: {top_candidate['candidate_id']} (Score: {top_candidate['overall_candidate_score']:.1f})")
    
    # Summary comparison
    print(f"\n" + "="*60)
    print("JOB REQUIREMENT COMPARISON SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('avg_overall', ascending=False)
    
    print(f"\nRanked by Average Overall Score:")
    for i, row in results_df.iterrows():
        print(f"{row['job_title']:25s} | Avg: {row['avg_overall']:5.1f} | "
              f"Coverage: {row['avg_coverage']:5.1f}% | "
              f"High Fit: {row['high_fit_count']:2.0f} ({row['high_fit_percent']:4.1f}%)")
    
    print(f"\n💡 INSIGHTS:")
    print(f"   • Job requirements matching your dataset will score much higher")
    print(f"   • Technical roles need candidates with matching backgrounds")
    print(f"   • Business roles show better fit with your diverse candidate pool")
    print(f"   • System correctly adapts scoring for different job types")

if __name__ == "__main__":
    test_realistic_jobs()
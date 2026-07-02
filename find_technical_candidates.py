#!/usr/bin/env python3
"""
Find more technical candidates from the dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_loader.candidate_loader import CandidateLoader
from src.job_requirements.job_schema import JobRequirements, get_ai_engineer_template
from src.feature_engineering.feature_pipeline import FeaturePipeline

def find_technical_candidates():
    """Find candidates with technical backgrounds"""
    
    try:
        # Load more candidates
        loader = CandidateLoader('Data/raw/candidates.jsonl')
        candidates = loader.load_candidates(limit=500)  # Load more to find technical ones
        print(f"Loaded {len(candidates)} candidates to search")
        
        # Technical keywords to look for
        technical_keywords = [
            'engineer', 'developer', 'programmer', 'software', 'python', 'java',
            'data scientist', 'ml', 'ai', 'machine learning', 'deep learning',
            'tensorflow', 'pytorch', 'react', 'angular', 'node', 'aws', 'cloud'
        ]
        
        technical_candidates = []
        
        for candidate in candidates:
            # Check profile for technical keywords
            profile = candidate.get('profile', {})
            headline = profile.get('headline', '').lower()
            summary = profile.get('summary', '').lower()
            current_title = profile.get('current_title', '').lower()
            
            # Check skills for technical skills
            skills = candidate.get('skills', [])
            skill_names = [skill.get('name', '').lower() for skill in skills]
            
            # Check career history for technical roles
            career_history = candidate.get('career_history', [])
            job_titles = [job.get('title', '').lower() for job in career_history]
            
            # Combine all text to search
            all_text = ' '.join([headline, summary, current_title] + skill_names + job_titles)
            
            # Count technical keyword matches
            technical_matches = sum(1 for keyword in technical_keywords if keyword in all_text)
            
            if technical_matches >= 2:  # At least 2 technical keywords
                technical_candidates.append((candidate, technical_matches))
        
        print(f"Found {len(technical_candidates)} candidates with technical backgrounds")
        
        # Sort by technical keyword count
        technical_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 50 most technical candidates
        top_technical = [cand[0] for cand in technical_candidates[:50]]
        
        if top_technical:
            print(f"\nTesting top {len(top_technical)} technical candidates...")
            
            # Run feature extraction on technical candidates
            job_requirements = get_ai_engineer_template()
            pipeline = FeaturePipeline(job_requirements, include_embeddings=False)
            
            features_df = pipeline.process_candidates_batch(top_technical)
            
            print(f"\n🎯 RESULTS WITH TECHNICAL CANDIDATES:")
            print(f"Average technical fit: {features_df['technical_fit_score'].mean():.1f}")
            print(f"Average overall score: {features_df['overall_candidate_score'].mean():.1f}")
            print(f"Candidates with >50 technical fit: {(features_df['technical_fit_score'] > 50).sum()}")
            
            # Show top results
            print(f"\nTop 5 Technical Candidates:")
            top_5 = features_df.nlargest(5, 'overall_candidate_score')
            
            for i, (_, candidate) in enumerate(top_5.iterrows(), 1):
                print(f"{i}. {candidate['candidate_id']:15s} | Overall: {candidate['overall_candidate_score']:5.1f} | "
                      f"Technical: {candidate['technical_fit_score']:5.1f} | Skills: {candidate['required_skills_coverage']:5.1f}%")
        
        else:
            print("No technical candidates found in the dataset")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_technical_candidates()
#!/usr/bin/env python3
"""
Test Phase 4 Ranking Pipeline
Simple test script for the complete candidate ranking system
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import get_ai_engineer_template
from src.ranking.generate_rankings import RankingPipeline

def test_complete_ranking():
    """Test the complete ranking pipeline"""
    
    print("🚀 TESTING PHASE 4 - RANKING PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Load candidate features
        features_path = 'Data/processed/candidate_features.parquet'
        print(f"Loading candidate features from: {features_path}")
        
        candidates_df = pd.read_parquet(features_path)
        print(f"✅ Loaded {len(candidates_df)} candidates with {len(candidates_df.columns)} features")
        
        # Step 2: Create job requirements
        job_requirements = get_ai_engineer_template()
        print(f"✅ Job Requirements: {job_requirements.role_title} ({job_requirements.role_level})")
        
        # Step 3: Initialize ranking pipeline (without embeddings for speed)
        pipeline = RankingPipeline(job_requirements, include_embeddings=False)
        print(f"✅ Ranking pipeline initialized")
        
        # Step 4: Run complete pipeline
        print(f"\n🔄 Running complete ranking pipeline...")
        final_rankings = pipeline.run_complete_pipeline(candidates_df, top_k=50)
        
        # Step 5: Show results
        print(f"\n📊 RANKING RESULTS:")
        print(f"Total candidates ranked: {len(final_rankings)}")
        
        if len(final_rankings) > 0:
            print(f"\nTop 10 Candidates:")
            for i, (_, candidate) in enumerate(final_rankings.head(10).iterrows(), 1):
                candidate_id = candidate['candidate_id']
                hybrid_score = candidate.get('hybrid_score', 0)
                rule_score = candidate.get('rule_based_score', 0)
                recommended = '✅' if candidate.get('is_recommended', False) else '❌'
                
                print(f"{i:2d}. {candidate_id:15s} | Hybrid: {hybrid_score:5.1f} | Rule: {rule_score:5.1f} | {recommended}")
        
        print(f"\n🎉 Phase 4 ranking pipeline completed successfully!")
        return final_rankings
        
    except Exception as e:
        print(f"❌ Error in ranking pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_complete_ranking()
#!/usr/bin/env python3
"""
Test Phase 5 Trust & Explainability Engine
Complete test of the trust and explainability system
"""

import sys
from pathlib import Path
import pandas as pd
import warnings
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import get_ai_engineer_template
from src.explainability.trust_engine import TrustEngine

def test_complete_trust_engine():
    """Test the complete Phase 5 trust and explainability pipeline"""
    
    print("🔒 TESTING PHASE 5 - TRUST & EXPLAINABILITY ENGINE")
    print("="*70)
    
    try:
        # Step 1: Load existing rankings from Phase 4
        print("📂 LOADING PHASE 4 RESULTS")
        print("-" * 40)
        
        rankings_path = 'Data/processed/final_rankings.csv'
        features_path = 'Data/processed/candidate_features.parquet'
        jd_profile_path = 'Data/processed/jd_profile.json'
        
        # Load data
        rankings_df = pd.read_csv(rankings_path)
        features_df = pd.read_parquet(features_path)
        
        with open(jd_profile_path, 'r') as f:
            jd_profile = json.load(f)
        
        print(f"✅ Loaded {len(rankings_df)} ranked candidates")
        print(f"✅ Loaded {len(features_df)} candidate features")
        print(f"✅ Loaded JD profile: {jd_profile['job_info']['role_title']}")
        
        # Merge rankings with features for complete dataset
        candidates_df = features_df.merge(
            rankings_df[['candidate_id', 'final_rank', 'is_recommended']], 
            on='candidate_id', 
            how='left'
        )
        
        print(f"✅ Merged dataset: {len(candidates_df)} candidates")
        
        # Step 2: Initialize Trust Engine
        print("\n🔒 INITIALIZING TRUST ENGINE")
        print("-" * 40)
        
        job_requirements = get_ai_engineer_template()
        trust_engine = TrustEngine(job_requirements, jd_profile)
        
        # Step 3: Run Trust Enhancement
        print("\n🚀 RUNNING TRUST ENHANCEMENT PIPELINE")
        print("-" * 40)
        
        enhanced_candidates = trust_engine.enhance_with_trust(candidates_df)
        
        # Step 4: Analyze Results
        print("\n📊 ANALYZING TRUST RESULTS")
        print("-" * 40)
        
        trust_engine.analyze_trust_results(enhanced_candidates, top_k=15)
        
        # Step 5: Export Results
        print("\n📁 EXPORTING TRUST RESULTS")
        print("-" * 40)
        
        output_paths = trust_engine.export_trust_results(enhanced_candidates)
        
        # Step 6: Run Evaluation
        print("\n✅ RUNNING EVALUATION CHECKLIST")
        print("-" * 40)
        
        trust_engine.run_evaluation_checklist(enhanced_candidates)
        
        # Step 7: Demo Explainable Output
        print("\n🎯 EXPLAINABLE OUTPUT DEMO")
        print("="*70)
        
        print("BEFORE Phase 5:")
        print("  Candidate A Score: 73.2")
        print("")
        print("AFTER Phase 5:")
        
        top_candidate = enhanced_candidates.nlargest(1, 'hybrid_score').iloc[0]
        explainable = top_candidate['explainable_output']
        
        print(f"  {explainable['candidate_id']}")
        print(f"  Match Score: {explainable['match_score']}")
        print(f"  Why: {explainable['summary']}")
        
        for strength in explainable['why_good']:
            print(f"    ✓ {strength}")
        
        for concern in explainable['concerns']:
            print(f"    • {concern}")
        
        print(f"  Confidence: {explainable['confidence']}")
        print(f"  Trust Level: {explainable['trust_level']}")
        print(f"  Verified: {'✓' if explainable['verified'] else '✗'}")
        print(f"  Recommendation: {explainable['recommendation']}")
        
        print(f"\n🎉 PHASE 5 COMPLETED SUCCESSFULLY!")
        
        return enhanced_candidates, output_paths
        
    except Exception as e:
        print(f"❌ Error in Phase 5 testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    candidates, paths = test_complete_trust_engine()
    
    if candidates is not None:
        print(f"\n" + "="*70)
        print("PHASE 5 SUMMARY")
        print("="*70)
        print("✅ Confidence Engine: Measures AI decision confidence")
        print("✅ Reasoning Engine: Generates human-readable explanations")
        print("✅ Honeypot Detector: Identifies fake/synthetic candidates")
        print("✅ Fairness Auditor: Analyzes system for bias")
        print("✅ Trust Scores: Overall trustworthiness assessment")
        print("✅ Explainable Output: Recruiter-friendly format")
        
        print(f"\n📁 OUTPUTS GENERATED:")
        if paths:
            for output_type, path in paths.items():
                print(f"   📄 {output_type}: {path.name}")
        
        print(f"\n🚀 AI IS NOW TRUSTWORTHY!")
        print("Recruiters will never ask 'Why did Candidate A rank above B?' again.")
        print("The system now explains every decision automatically.")
        
        print(f"\n🎯 READY FOR PHASE 6: Recruiter Copilot & Demo!")
    else:
        print("Phase 5 test failed. Please check the errors above.")
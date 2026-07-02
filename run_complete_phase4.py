#!/usr/bin/env python3
"""
Complete Phase 4 Ranking Pipeline
Final test of the complete candidate intelligence & ranking engine
"""

import sys
from pathlib import Path
import pandas as pd
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import get_ai_engineer_template
from src.ranking.jd_parser import JDProfileParser
from src.ranking.rule_matcher import RuleBasedMatcher
from src.ranking.hybrid_ranker import HybridRanker

def run_complete_phase4():
    """Run the complete Phase 4 pipeline"""
    
    print("🚀 PHASE 4 - CANDIDATE INTELLIGENCE & RANKING ENGINE")
    print("="*70)
    
    # STEP 1: Load candidate features
    print("\n📂 STEP 1: LOADING CANDIDATE FEATURES")
    print("-" * 50)
    
    candidates_df = pd.read_parquet('Data/processed/candidate_features.parquet')
    print(f"✅ Loaded {len(candidates_df)} candidates with {len(candidates_df.columns)} features")
    
    # STEP 2: Create job requirements and JD profile
    print("\n🎯 STEP 2: CREATING JD PROFILE")
    print("-" * 50)
    
    job_requirements = get_ai_engineer_template()
    print(f"Job: {job_requirements.role_title} ({job_requirements.role_level})")
    print(f"Required skills: {job_requirements.required_skills}")
    
    parser = JDProfileParser()
    jd_profile = parser.create_jd_profile(job_requirements)
    parser.save_jd_profile(jd_profile)
    print(f"✅ JD Profile created and saved")
    
    # STEP 3: Rule-based matching
    print("\n⚖️  STEP 3: RULE-BASED MATCHING")
    print("-" * 50)
    
    rule_matcher = RuleBasedMatcher(jd_profile, job_requirements)
    candidates_df = rule_matcher.calculate_rule_based_scores(candidates_df.copy())
    print(f"✅ Rule-based scoring completed")
    
    # Show rule-based results
    passed_constraints = candidates_df['hard_constraints_passed'].sum()
    avg_rule_score = candidates_df['rule_based_score'].mean()
    print(f"Candidates passing constraints: {passed_constraints}/{len(candidates_df)} ({passed_constraints/len(candidates_df)*100:.1f}%)")
    print(f"Average rule score: {avg_rule_score:.1f}")
    
    # STEP 4: Hybrid ranking
    print("\n🔀 STEP 4: HYBRID RANKING")
    print("-" * 50)
    
    hybrid_ranker = HybridRanker(jd_profile, job_requirements)
    candidates_df = hybrid_ranker.calculate_hybrid_scores(candidates_df)
    print(f"✅ Hybrid scoring completed")
    
    # STEP 5: Final rankings
    print("\n🏆 STEP 5: FINAL RANKINGS")
    print("-" * 50)
    
    final_rankings = hybrid_ranker.get_final_rankings(candidates_df, top_k=50)
    csv_path = hybrid_ranker.export_rankings(final_rankings)
    
    print(f"✅ Final rankings generated and exported")
    
    # STEP 6: Results analysis
    print("\n📊 STEP 6: RESULTS ANALYSIS")
    print("-" * 50)
    
    recommended_count = final_rankings['is_recommended'].sum()
    avg_hybrid_score = final_rankings['hybrid_score'].mean()
    
    print(f"📈 STATISTICS:")
    print(f"   Total ranked: {len(final_rankings)}")
    print(f"   Recommended: {recommended_count} ({recommended_count/len(final_rankings)*100:.1f}%)")
    print(f"   Average hybrid score: {avg_hybrid_score:.1f}")
    print(f"   Score range: {final_rankings['hybrid_score'].min():.1f} - {final_rankings['hybrid_score'].max():.1f}")
    
    print(f"\n🏅 TOP 15 CANDIDATES:")
    print(f"{'Rank':<4} {'Candidate ID':<15} {'Hybrid':<7} {'Rule':<6} {'Technical':<9} {'Career':<7} {'Behavioral':<10} {'Rec':<4}")
    print("-" * 75)
    
    for i, (_, candidate) in enumerate(final_rankings.head(15).iterrows(), 1):
        candidate_id = candidate['candidate_id']
        hybrid_score = candidate.get('hybrid_score', 0)
        rule_score = candidate.get('rule_based_score', 0) 
        technical_score = candidate.get('technical_fit_score', 0)
        career_score = candidate.get('career_score', 0)
        behavioral_score = candidate.get('behavioral_score', 0)
        recommended = '✅' if candidate.get('is_recommended', False) else '❌'
        
        print(f"{i:3d}. {candidate_id:<15} {hybrid_score:5.1f}   {rule_score:4.1f}   {technical_score:7.1f}   {career_score:5.1f}   {behavioral_score:8.1f}   {recommended}")
    
    # STEP 7: Evaluation checklist
    print(f"\n✅ STEP 7: EVALUATION CHECKLIST")
    print("-" * 50)
    
    print(f"CHECKPOINT QUESTIONS:")
    print(f"1. ❓ Would YOU hire these top 15 candidates for {job_requirements.role_title}?")
    print(f"2. ❓ Why did candidate 3 rank above candidate 4?")
    print(f"3. ❓ Does the ranking look human and logical?")
    print(f"4. ❓ Are high technical scores getting rewarded appropriately?")
    print(f"5. ❓ Are behavioral factors (availability, engagement) considered?")
    
    # Final deliverables
    print(f"\n📁 PHASE 4 DELIVERABLES:")
    print(f"✅ JD Profile: Data/processed/jd_profile.json")
    print(f"✅ Final Rankings: {csv_path}")
    print(f"✅ Candidate Intelligence Engine: Complete")
    print(f"✅ Evaluation Framework: Ready for review")
    
    print(f"\n🎉 PHASE 4 COMPLETED SUCCESSFULLY!")
    print(f"The ranking engine now answers: 'How well does this candidate fit THIS job?'")
    
    return final_rankings

if __name__ == "__main__":
    rankings = run_complete_phase4()
    
    print(f"\n" + "="*70)
    print("PHASE 4 SUMMARY")
    print("="*70)
    print("✅ Job Description Profile: Created comprehensive matching criteria")
    print("✅ Rule-Based Matching: Applied hard constraints and business logic") 
    print("✅ Hybrid Ranking: Combined all scoring approaches intelligently")
    print("✅ Final Rankings: Generated top candidate recommendations")
    print("✅ Export & Evaluation: CSV output with manual review checklist")
    print(f"\n🚀 Ready for Phase 5: Advanced Intelligence Features!")
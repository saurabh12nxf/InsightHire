#!/usr/bin/env python3
"""
Debug Phase 4 step by step
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        from src.job_requirements.job_schema import get_ai_engineer_template
        print("✅ job_schema imported")
    except Exception as e:
        print(f"❌ job_schema import failed: {e}")
        return False
    
    try:
        from src.ranking.jd_parser import JDProfileParser
        print("✅ jd_parser imported")
    except Exception as e:
        print(f"❌ jd_parser import failed: {e}")
        return False
    
    try:
        from src.ranking.rule_matcher import RuleBasedMatcher
        print("✅ rule_matcher imported")
    except Exception as e:
        print(f"❌ rule_matcher import failed: {e}")
        return False
    
    try:
        from src.ranking.hybrid_ranker import HybridRanker
        print("✅ hybrid_ranker imported")
    except Exception as e:
        print(f"❌ hybrid_ranker import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test data loading
        candidates_df = pd.read_parquet('Data/processed/candidate_features.parquet')
        print(f"✅ Loaded {len(candidates_df)} candidates")
        
        # Test job requirements
        from src.job_requirements.job_schema import get_ai_engineer_template
        job_req = get_ai_engineer_template()
        print(f"✅ Job requirements: {job_req.role_title}")
        
        # Test JD Profile creation
        from src.ranking.jd_parser import JDProfileParser
        parser = JDProfileParser()
        jd_profile = parser.create_jd_profile(job_req)
        print(f"✅ JD Profile created: {jd_profile['job_info']['role_title']}")
        
        # Test rule-based matching
        from src.ranking.rule_matcher import RuleBasedMatcher
        rule_matcher = RuleBasedMatcher(jd_profile, job_req)
        print(f"✅ Rule matcher initialized")
        
        # Test basic scoring on first candidate
        first_candidate = candidates_df.iloc[0]
        print(f"✅ Testing with candidate: {first_candidate['candidate_id']}")
        
        # Test rule-based scoring on small sample
        small_sample = candidates_df.head(5)
        scored_sample = rule_matcher.calculate_rule_based_scores(small_sample)
        print(f"✅ Rule-based scoring completed for {len(scored_sample)} candidates")
        
        # Show results
        print(f"\nSample Results:")
        for i, (_, candidate) in enumerate(scored_sample.iterrows(), 1):
            candidate_id = candidate['candidate_id']
            rule_score = candidate.get('rule_based_score', 0)
            constraints_passed = candidate.get('hard_constraints_passed', False)
            print(f"{i}. {candidate_id:15s} | Rule Score: {rule_score:5.1f} | Constraints: {'✅' if constraints_passed else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 DEBUGGING PHASE 4 COMPONENTS")
    print("="*50)
    
    if test_imports():
        print("All imports successful!")
        if test_basic_functionality():
            print("\n🎉 All basic functionality works!")
            print("Phase 4 components are ready to use.")
        else:
            print("\n❌ Basic functionality test failed.")
    else:
        print("Import test failed.")
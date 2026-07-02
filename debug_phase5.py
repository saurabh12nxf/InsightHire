#!/usr/bin/env python3
"""
Debug Phase 5 components step by step
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

def test_data_loading():
    """Test loading Phase 4 results"""
    print("🔍 Testing data loading...")
    
    try:
        # Check if files exist
        rankings_path = Path('Data/processed/final_rankings.csv')
        features_path = Path('Data/processed/candidate_features.parquet')
        jd_profile_path = Path('Data/processed/jd_profile.json')
        
        print(f"Rankings file exists: {rankings_path.exists()}")
        print(f"Features file exists: {features_path.exists()}")
        print(f"JD profile exists: {jd_profile_path.exists()}")
        
        if rankings_path.exists():
            rankings_df = pd.read_csv(rankings_path)
            print(f"✅ Rankings loaded: {len(rankings_df)} rows")
        
        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            print(f"✅ Features loaded: {len(features_df)} rows, {len(features_df.columns)} columns")
        
        if jd_profile_path.exists():
            with open(jd_profile_path, 'r') as f:
                jd_profile = json.load(f)
            print(f"✅ JD Profile loaded: {jd_profile['job_info']['role_title']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_imports():
    """Test Phase 5 imports"""
    print("\n🔍 Testing Phase 5 imports...")
    
    try:
        from src.job_requirements.job_schema import get_ai_engineer_template
        print("✅ job_schema imported")
        
        from src.explainability.confidence_engine import ConfidenceEngine
        print("✅ confidence_engine imported")
        
        from src.explainability.reasoning_engine import ReasoningEngine
        print("✅ reasoning_engine imported")
        
        from src.explainability.honeypot_detector import HoneypotDetector
        print("✅ honeypot_detector imported")
        
        from src.explainability.fairness_audit import FairnessAuditor
        print("✅ fairness_audit imported")
        
        from src.explainability.trust_engine import TrustEngine
        print("✅ trust_engine imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_confidence_engine():
    """Test confidence engine specifically"""
    print("\n🔍 Testing confidence engine...")
    
    try:
        from src.job_requirements.job_schema import get_ai_engineer_template
        from src.explainability.confidence_engine import ConfidenceEngine
        
        # Load sample data
        features_df = pd.read_parquet('Data/processed/candidate_features.parquet')
        job_requirements = get_ai_engineer_template()
        
        # Test confidence engine
        confidence_engine = ConfidenceEngine(job_requirements)
        
        # Test on small sample
        sample_df = features_df.head(5).copy()
        enhanced_df = confidence_engine.calculate_confidence_scores(sample_df)
        
        print(f"✅ Confidence scores calculated for {len(enhanced_df)} candidates")
        
        # Show results
        for i, (idx, candidate) in enumerate(enhanced_df.iterrows(), 1):
            candidate_id = candidate['candidate_id']
            confidence = candidate.get('confidence_score', 0)
            level = candidate.get('confidence_level', 'Unknown')
            print(f"   {i}. {candidate_id}: {confidence:.1f}% ({level})")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence engine error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reasoning_engine():
    """Test reasoning engine specifically"""
    print("\n🔍 Testing reasoning engine...")
    
    try:
        from src.job_requirements.job_schema import get_ai_engineer_template
        from src.explainability.reasoning_engine import ReasoningEngine
        
        # Load data
        features_df = pd.read_parquet('Data/processed/candidate_features.parquet')
        
        with open('Data/processed/jd_profile.json', 'r') as f:
            jd_profile = json.load(f)
        
        job_requirements = get_ai_engineer_template()
        
        # Test reasoning engine
        reasoning_engine = ReasoningEngine(job_requirements, jd_profile)
        
        # Test on small sample
        sample_df = features_df.head(3).copy()
        enhanced_df = reasoning_engine.generate_reasoning(sample_df)
        
        print(f"✅ Reasoning generated for {len(enhanced_df)} candidates")
        
        # Show sample reasoning
        for i, (idx, candidate) in enumerate(enhanced_df.iterrows(), 1):
            reasoning = candidate.get('reasoning_explanation', {})
            print(f"\n   {i}. {reasoning.get('candidate_id', 'Unknown')}")
            print(f"      Assessment: {reasoning.get('overall_assessment', 'N/A')}")
            
            strengths = reasoning.get('strengths', [])
            if strengths:
                print(f"      Strengths: {', '.join(strengths[:2])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning engine error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 DEBUGGING PHASE 5 COMPONENTS")
    print("="*50)
    
    # Test each component
    if test_data_loading():
        print("Data loading: ✅")
    else:
        print("Data loading: ❌")
        exit(1)
    
    if test_imports():
        print("Imports: ✅")
    else:
        print("Imports: ❌")
        exit(1)
    
    if test_confidence_engine():
        print("Confidence engine: ✅")
    else:
        print("Confidence engine: ❌")
        exit(1)
    
    if test_reasoning_engine():
        print("Reasoning engine: ✅")
    else:
        print("Reasoning engine: ❌")
        exit(1)
    
    print(f"\n🎉 All Phase 5 components working!")
    print("Ready to run the complete trust engine.")
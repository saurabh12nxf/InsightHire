#!/usr/bin/env python3
"""
Main script to load candidates dataset for TalentLens AI
Phase 1 checkpoint script
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader.candidate_loader import CandidateLoader


def main():
    """Load candidates and display count - Phase 1 checkpoint."""
    print("InsightHire- Loading Candidates Dataset")
    print("=" * 50)
    
    try:
        # Initialize the loader
        loader = CandidateLoader("Data/raw/candidates.jsonl")
        
        # Get total count first (faster than loading all)
        total_count = loader.get_candidate_count()
        
        if total_count > 0:
            print(f"Loaded {total_count} candidates")
            print("✅ Phase 1 complete")
            
            # Load a small sample to verify structure
            print("\nVerifying data structure with sample...")
            sample_candidates = loader.load_candidates(limit=5)
            
            if sample_candidates:
                first_candidate = sample_candidates[0]
                print(f"Sample candidate ID: {first_candidate.get('candidate_id')}")
                print(f"Sample candidate name: {first_candidate.get('profile', {}).get('anonymized_name')}")
                print(f"Sample candidate experience: {first_candidate.get('profile', {}).get('years_of_experience')} years")
                
                # Validate structure
                if loader.validate_candidate_structure(first_candidate):
                    print("✅ Data structure validation passed")
                else:
                    print(" Data structure validation failed")
            
        else:
            print(" No candidates found in dataset")
            return 1
            
    except FileNotFoundError:
        print(" Candidates file not found at Data/raw/candidates.jsonl")
        print("Make sure the dataset is in the correct location")
        return 1
    except Exception as e:
        print(f" Error loading candidates: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
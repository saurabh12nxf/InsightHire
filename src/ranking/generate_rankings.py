#!/usr/bin/env python3
"""
Generate Rankings - Master pipeline for complete candidate ranking process
Orchestrates the entire Phase 4 ranking pipeline from JD profile to final rankings
"""

import sys
from pathlib import Path
import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements, get_ai_engineer_template
from src.ranking.jd_parser import JDProfileParser, create_ai_engineer_profile
from src.ranking.semantic_matcher import SemanticMatcher
from src.ranking.rule_matcher import RuleBasedMatcher
from src.ranking.hybrid_ranker import HybridRanker
from src.ranking.score_combiner import ScoreCombiner


class RankingPipeline:
    """
    Master ranking pipeline that orchestrates the complete candidate ranking process.
    Combines JD parsing, semantic matching, rule-based matching, and hybrid ranking.
    """
    
    def __init__(self, job_requirements: JobRequirements, 
                 include_embeddings: bool = True,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the ranking pipeline
        
        Args:
            job_requirements: Job requirements object
            include_embeddings: Whether to use semantic matching
            embedding_model: Model for embeddings
        """
        
        self.job_requirements = job_requirements
        self.include_embeddings = include_embeddings
        self.embedding_model = embedding_model
        
        # Initialize components
        self.jd_parser = JDProfileParser()
        self.jd_profile = None
        self.semantic_matcher = None
        self.rule_matcher = None
        self.hybrid_ranker = None
        
        print(f"Initialized ranking pipeline for: {job_requirements.role_title}")
        print(f"Embeddings enabled: {include_embeddings}")
    
    def create_jd_profile(self, save_profile: bool = True) -> Dict[str, Any]:
        """
        Step 1: Create comprehensive JD profile
        
        Args:
            save_profile: Whether to save profile to file
            
        Returns:
            JD profile dictionary
        """
        
        print("\n" + "="*60)
        print("STEP 1: CREATING JD PROFILE")
        print("="*60)
        
        # Create JD profile
        self.jd_profile = self.jd_parser.create_jd_profile(self.job_requirements)
        
        # Save profile if requested
        if save_profile:
            self.jd_parser.save_jd_profile(self.jd_profile)
        
        # Print summary for review
        self.jd_parser.print_jd_profile_summary(self.jd_profile)
        
        return self.jd_profile
    
    def calculate_semantic_similarities(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Calculate semantic similarities using embeddings
        
        Args:
            candidates_df: DataFrame with candidate features and embeddings
            
        Returns:
            DataFrame with semantic similarity scores added
        """
        
        if not self.include_embeddings:
            print("\n🔍 STEP 2: SKIPPING SEMANTIC MATCHING (embeddings disabled)")
            candidates_df['semantic_similarity_score'] = 50.0  # Neutral score
            return candidates_df
        
        print("\n" + "="*60)
        print("STEP 2: CALCULATING SEMANTIC SIMILARITIES")
        print("="*60)
        
        # Initialize semantic matcher
        if not self.semantic_matcher:
            self.semantic_matcher = SemanticMatcher(self.embedding_model)
        
        # Create job embedding
        job_embedding = self.semantic_matcher.create_job_embedding(
            self.jd_profile, self.job_requirements
        )
        
        if job_embedding is not None:
            # Calculate similarities
            candidates_df = self.semantic_matcher.calculate_semantic_similarities(candidates_df)
            
            # Analyze top matches
            top_matches = self.semantic_matcher.get_top_semantic_matches(candidates_df, top_k=20)
            self.semantic_matcher.analyze_semantic_matches(top_matches)
            
            # Save results
            self.semantic_matcher.save_semantic_results(candidates_df)
        else:
            print("Warning: Could not create job embedding. Using neutral semantic scores.")
            candidates_df['semantic_similarity_score'] = 50.0
        
        return candidates_df
    
    def calculate_rule_based_scores(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Calculate rule-based scores and apply hard constraints
        
        Args:
            candidates_df: DataFrame with candidate features
            
        Returns:
            DataFrame with rule-based scores added
        """
        
        print("\n" + "="*60)
        print("STEP 3: CALCULATING RULE-BASED SCORES")
        print("="*60)
        
        # Initialize rule matcher
        if not self.rule_matcher:
            self.rule_matcher = RuleBasedMatcher(self.jd_profile, self.job_requirements)
        
        # Calculate rule-based scores
        candidates_df = self.rule_matcher.calculate_rule_based_scores(candidates_df)
        
        # Analyze results
        self.rule_matcher.analyze_rule_based_results(candidates_df)
        
        return candidates_df
    
    def calculate_hybrid_scores(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Calculate final hybrid scores combining all approaches
        
        Args:
            candidates_df: DataFrame with all previous scores
            
        Returns:
            DataFrame with final hybrid scores and rankings
        """
        
        print("\n" + "="*60)
        print("STEP 4: CALCULATING HYBRID SCORES")
        print("="*60)
        
        # Initialize hybrid ranker
        if not self.hybrid_ranker:
            self.hybrid_ranker = HybridRanker(self.jd_profile, self.job_requirements)
        
        # Calculate hybrid scores
        candidates_df = self.hybrid_ranker.calculate_hybrid_scores(candidates_df)
        
        # Analyze results
        top_candidates = self.hybrid_ranker.analyze_hybrid_results(candidates_df)
        
        return candidates_df
    
    def generate_final_rankings(self, candidates_df: pd.DataFrame, 
                               top_k: int = 100,
                               export_csv: bool = True) -> pd.DataFrame:
        """
        Step 5: Generate final candidate rankings
        
        Args:
            candidates_df: DataFrame with hybrid scores
            top_k: Number of top candidates to return
            export_csv: Whether to export rankings to CSV
            
        Returns:
            DataFrame with final rankings
        """
        
        print("\n" + "="*60)
        print("STEP 5: GENERATING FINAL RANKINGS")
        print("="*60)
        
        if not self.hybrid_ranker:
            print("Error: Hybrid ranker not initialized. Run calculate_hybrid_scores first.")
            return candidates_df.head(top_k)
        
        # Get final rankings
        final_rankings = self.hybrid_ranker.get_final_rankings(candidates_df, top_k)
        
        # Export to CSV if requested
        if export_csv:
            csv_path = self.hybrid_ranker.export_rankings(final_rankings, top_k=top_k)
            print(f"\n📊 Final rankings saved to: {csv_path}")
        
        # Print final summary
        self._print_final_summary(final_rankings)
        
        return final_rankings
    
    def run_complete_pipeline(self, candidates_df: pd.DataFrame, 
                            top_k: int = 100) -> pd.DataFrame:
        """
        Run the complete ranking pipeline
        
        Args:
            candidates_df: DataFrame with candidate features (from Phase 3)
            top_k: Number of top candidates to return
            
        Returns:
            DataFrame with final rankings
        """
        
        print("🚀 STARTING COMPLETE RANKING PIPELINE")
        print(f"Processing {len(candidates_df)} candidates for: {self.job_requirements.role_title}")
        
        start_time = time.time()
        
        try:
            # Step 1: Create JD Profile
            self.create_jd_profile()
            
            # Step 2: Calculate Semantic Similarities  
            candidates_df = self.calculate_semantic_similarities(candidates_df)
            
            # Step 3: Calculate Rule-based Scores
            candidates_df = self.calculate_rule_based_scores(candidates_df)
            
            # Step 4: Calculate Hybrid Scores
            candidates_df = self.calculate_hybrid_scores(candidates_df)
            
            # Step 5: Generate Final Rankings
            final_rankings = self.generate_final_rankings(candidates_df, top_k)
            
            # Pipeline completed successfully
            total_time = time.time() - start_time
            print(f"\n🎉 RANKING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total processing time: {total_time:.2f} seconds")
            print(f"📈 Final rankings generated for top {len(final_rankings)} candidates")
            
            return final_rankings
            
        except Exception as e:
            print(f"\n❌ ERROR in ranking pipeline: {e}")
            import traceback
            traceback.print_exc()
            return candidates_df.head(top_k)
    
    def _print_final_summary(self, final_rankings: pd.DataFrame):
        """Print final pipeline summary"""
        
        print(f"\n" + "="*60)
        print("FINAL RANKING SUMMARY")
        print("="*60)
        
        if len(final_rankings) == 0:
            print("No candidates in final rankings.")
            return
        
        # Top 10 summary
        print(f"\n🏆 TOP 10 FINAL RANKINGS:")
        print(f"{'Rank':<4} {'Candidate ID':<15} {'Hybrid Score':<12} {'Recommended':<12}")
        print("-" * 50)
        
        for i, (_, candidate) in enumerate(final_rankings.head(10).iterrows(), 1):
            candidate_id = candidate['candidate_id']
            hybrid_score = candidate.get('hybrid_score', 0)
            recommended = '✓' if candidate.get('is_recommended', False) else '✗'
            
            print(f"{i:3d}. {candidate_id:<15} {hybrid_score:10.1f}  {recommended:<12}")
        
        # Statistics
        avg_score = final_rankings['hybrid_score'].mean()
        recommended_count = final_rankings['is_recommended'].sum()
        
        print(f"\n📊 STATISTICS:")
        print(f"   Average Score: {avg_score:.1f}")
        print(f"   Recommended: {recommended_count}/{len(final_rankings)} ({recommended_count/len(final_rankings)*100:.1f}%)")
        print(f"   Score Range: {final_rankings['hybrid_score'].min():.1f} - {final_rankings['hybrid_score'].max():.1f}")
        
        print(f"\n✅ CHECKPOINT 5: Final rankings generated!")
        print(f"   📁 Check ranking.csv for complete results")
        print(f"   🔍 Review top candidates for hiring decisions")


def run_ranking_evaluation(final_rankings: pd.DataFrame, 
                          original_candidates: List[Dict[str, Any]] = None):
    """
    Step 6: Evaluation - Manual review of ranking quality
    
    Args:
        final_rankings: DataFrame with final rankings
        original_candidates: Original candidate data for detailed review
    """
    
    print("\n" + "="*60)
    print("STEP 6: RANKING EVALUATION")
    print("="*60)
    
    if len(final_rankings) == 0:
        print("No rankings to evaluate.")
        return
    
    print("🔍 MANUAL EVALUATION CHECKLIST:")
    print("\n1. TOP 20 CANDIDATES REVIEW:")
    
    top_20 = final_rankings.head(20)
    for i, (_, candidate) in enumerate(top_20.iterrows(), 1):
        print(f"   {i:2d}. {candidate['candidate_id']} (Score: {candidate.get('hybrid_score', 0):.1f})")
    
    print(f"\n📋 EVALUATION QUESTIONS:")
    print(f"   ❓ Question 1: Would YOU hire these top 20 candidates?")
    print(f"   ❓ Question 2: Why did Candidate 4 rank above Candidate 5?")
    print(f"   ❓ Question 3: Does the ranking look human and logical?")
    
    print(f"\n📊 COMPARISON ANALYSIS:")
    print(f"   📈 Top 20 (Rank 1-20): Avg Score = {top_20['hybrid_score'].mean():.1f}")
    
    if len(final_rankings) >= 40:
        bottom_20 = final_rankings.tail(20)
        print(f"   📉 Bottom 20: Avg Score = {bottom_20['hybrid_score'].mean():.1f}")
    
    if len(final_rankings) >= 60:
        middle_20_start = len(final_rankings) // 2 - 10
        middle_20_end = len(final_rankings) // 2 + 10
        middle_20 = final_rankings.iloc[middle_20_start:middle_20_end]
        print(f"   📊 Random Middle 20: Avg Score = {middle_20['hybrid_score'].mean():.1f}")
    
    print(f"\n✅ CHECKPOINT 6: Complete manual evaluation")
    print(f"   Everything should make intuitive sense!")
    print(f"   If not, adjust weights and scoring logic.")


# Main execution function
def main():
    """Main function to test the complete ranking pipeline"""
    
    print("PHASE 4 - CANDIDATE INTELLIGENCE & RANKING ENGINE")
    print("="*60)
    
    # Test with AI Engineer job
    job_requirements = get_ai_engineer_template()
    
    try:
        # Load candidate features from Phase 3
        features_path = Path('Data/processed/candidate_features.parquet')
        
        if features_path.exists():
            print(f"Loading candidate features from: {features_path}")
            candidates_df = pd.read_parquet(features_path)
            print(f"Loaded {len(candidates_df)} candidates with features")
        else:
            print(f"Error: Candidate features file not found at {features_path}")
            print("Please run the Phase 3 feature extraction pipeline first.")
            return
        
        # Initialize ranking pipeline
        pipeline = RankingPipeline(job_requirements, include_embeddings=False)  # Skip embeddings for speed
        
        # Run complete pipeline
        final_rankings = pipeline.run_complete_pipeline(candidates_df, top_k=100)
        
        # Run evaluation
        run_ranking_evaluation(final_rankings)
        
        print(f"\n🎉 PHASE 4 COMPLETE!")
        print(f"✅ JD Profile created and saved")
        print(f"✅ Semantic similarities calculated") 
        print(f"✅ Rule-based matching completed")
        print(f"✅ Hybrid scores generated")
        print(f"✅ Final rankings exported to CSV")
        print(f"✅ Manual evaluation checklist provided")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
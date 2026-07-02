#!/usr/bin/env python3
"""
Feature Pipeline - Master orchestrator for all feature extraction
Combines all feature extractors and creates final candidate feature vectors
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements
from src.feature_engineering.technical_features import TechnicalFeaturesExtractor
from src.feature_engineering.career_features import CareerFeaturesExtractor
from src.feature_engineering.behavioral_features import BehavioralFeaturesExtractor
from src.feature_engineering.trajectory_features import TrajectoryFeaturesExtractor
from src.feature_engineering.credibility_features import CredibilityFeaturesExtractor
from src.feature_engineering.embeddings_features import EmbeddingsFeaturesExtractor


class FeaturePipeline:
    """
    Master feature extraction pipeline that orchestrates all feature extractors.
    Processes candidates and creates comprehensive feature vectors for ranking.
    """
    
    def __init__(self, job_requirements: JobRequirements, 
                 include_embeddings: bool = True,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the feature pipeline
        
        Args:
            job_requirements: Job requirements configuration
            include_embeddings: Whether to generate text embeddings
            embedding_model: Model name for embeddings
        """
        
        self.job_requirements = job_requirements
        self.include_embeddings = include_embeddings
        
        # Initialize all feature extractors
        self.technical_extractor = TechnicalFeaturesExtractor(job_requirements)
        self.career_extractor = CareerFeaturesExtractor(job_requirements)
        self.behavioral_extractor = BehavioralFeaturesExtractor(job_requirements)
        self.trajectory_extractor = TrajectoryFeaturesExtractor(job_requirements)
        self.credibility_extractor = CredibilityFeaturesExtractor(job_requirements)
        
        # Initialize embeddings extractor if requested
        self.embeddings_extractor = None
        if include_embeddings:
            self.embeddings_extractor = EmbeddingsFeaturesExtractor(
                job_requirements, embedding_model
            )
    
    def extract_candidate_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features for a single candidate
        
        Args:
            candidate: Candidate data dictionary
            
        Returns:
            Complete feature dictionary for the candidate
        """
        
        candidate_id = candidate.get('candidate_id', 'unknown')
        
        try:
            # Extract features from each component
            technical_features = self.technical_extractor.extract_features(candidate)
            career_features = self.career_extractor.extract_features(candidate)
            behavioral_features = self.behavioral_extractor.extract_features(candidate)
            trajectory_features = self.trajectory_extractor.extract_features(candidate)
            credibility_features = self.credibility_extractor.extract_features(candidate)
            
            # Extract embeddings if enabled
            embeddings_features = {}
            if self.embeddings_extractor:
                embeddings_features = self.embeddings_extractor.extract_features(candidate)
            
            # Combine all features
            combined_features = {
                'candidate_id': candidate_id,
                'extraction_timestamp': datetime.now().isoformat(),
                
                # Technical features
                **technical_features,
                
                # Career features  
                **career_features,
                
                # Behavioral features
                **behavioral_features,
                
                # Trajectory features
                **trajectory_features,
                
                # Credibility features
                **credibility_features,
                
                # Embeddings features (if enabled)
                **embeddings_features
            }
            
            # Add overall composite scores
            combined_features.update(self._calculate_composite_scores(combined_features))
            
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features for candidate {candidate_id}: {e}")
            return self._get_fallback_features(candidate_id)
    
    def process_candidates_batch(self, candidates: List[Dict[str, Any]], 
                                batch_size: int = 100,
                                save_progress: bool = True,
                                output_dir: str = 'Data/processed') -> pd.DataFrame:
        """
        Process a batch of candidates and extract all features
        
        Args:
            candidates: List of candidate dictionaries
            batch_size: Number of candidates to process in each batch
            save_progress: Whether to save intermediate results
            output_dir: Directory to save results
            
        Returns:
            DataFrame with all candidate features
        """
        
        print(f"Starting feature extraction for {len(candidates)} candidates")
        print(f"Job requirements: {self.job_requirements.role_title} ({self.job_requirements.role_level})")
        
        all_features = []
        total_candidates = len(candidates)
        
        # Process in batches
        for batch_start in range(0, total_candidates, batch_size):
            batch_end = min(batch_start + batch_size, total_candidates)
            batch_candidates = candidates[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}: candidates {batch_start+1}-{batch_end}")
            
            batch_start_time = time.time()
            
            # Process each candidate in the batch
            for i, candidate in enumerate(batch_candidates):
                try:
                    features = self.extract_candidate_features(candidate)
                    all_features.append(features)
                    
                    # Progress indicator within batch
                    if (i + 1) % 50 == 0:
                        print(f"  Processed {i+1}/{len(batch_candidates)} in current batch")
                        
                except Exception as e:
                    candidate_id = candidate.get('candidate_id', f'batch_{batch_start}_{i}')
                    print(f"  Error processing candidate {candidate_id}: {e}")
                    fallback_features = self._get_fallback_features(candidate_id)
                    all_features.append(fallback_features)
            
            batch_time = time.time() - batch_start_time
            print(f"  Batch completed in {batch_time:.2f} seconds")
            
            # Save intermediate results
            if save_progress and len(all_features) > 0:
                self._save_intermediate_results(all_features, output_dir, batch_end)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Save final results
        output_path = Path(output_dir) / 'candidate_features.parquet'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(output_path, index=False)
        
        print(f"Feature extraction completed!")
        print(f"Results saved to: {output_path}")
        print(f"Total candidates processed: {len(features_df)}")
        
        # Print feature summary
        self._print_feature_summary(features_df)
        
        return features_df
    
    def _calculate_composite_scores(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate composite scores that combine multiple feature categories
        """
        
        # Overall candidate score (weighted combination of all main scores)
        main_scores = [
            features.get('technical_fit_score', 50.0),
            features.get('career_score', 50.0),
            features.get('behavioral_score', 50.0),
            features.get('trajectory_score', 50.0),
            features.get('credibility_score', 50.0)
        ]
        
        # Dynamic weighting based on job requirements
        if self.job_requirements.requires_github or len(self.job_requirements.required_skills) > 5:
            # Technical role - weight technical and credibility more
            weights = [0.30, 0.20, 0.20, 0.15, 0.15]  # Technical, Career, Behavioral, Trajectory, Credibility
        elif self.job_requirements.role_type in ['management', 'sales', 'marketing']:
            # Non-technical role - weight behavioral and career more
            weights = [0.15, 0.30, 0.30, 0.15, 0.10]
        else:
            # Balanced weighting
            weights = [0.25, 0.25, 0.25, 0.15, 0.10]
        
        overall_score = sum(score * weight for score, weight in zip(main_scores, weights))
        
        # Hiring likelihood score (behavioral + availability focus)
        hiring_likelihood = (
            features.get('behavioral_score', 50.0) * 0.40 +
            features.get('availability_score', 50.0) * 0.35 +
            features.get('credibility_score', 50.0) * 0.25
        )
        
        # Job fit score (technical + career focus)
        job_fit_score = (
            features.get('technical_fit_score', 50.0) * 0.45 +
            features.get('career_score', 50.0) * 0.35 +
            features.get('trajectory_score', 50.0) * 0.20
        )
        
        # Quality score (credibility + trajectory focus)
        quality_score = (
            features.get('credibility_score', 50.0) * 0.50 +
            features.get('trajectory_score', 50.0) * 0.30 +
            features.get('profile_quality_score', 50.0) * 0.20
        )
        
        return {
            'overall_candidate_score': round(overall_score, 2),
            'hiring_likelihood_score': round(hiring_likelihood, 2),
            'job_fit_score': round(job_fit_score, 2),
            'candidate_quality_score': round(quality_score, 2)
        }
    
    def _get_fallback_features(self, candidate_id: str) -> Dict[str, Any]:
        """
        Generate fallback features when extraction fails
        """
        
        return {
            'candidate_id': candidate_id,
            'extraction_timestamp': datetime.now().isoformat(),
            'error': True,
            
            # Technical features (neutral scores)
            'technical_fit_score': 50.0,
            'skill_match_score': 50.0,
            'required_skills_coverage': 0.0,
            'preferred_skills_coverage': 0.0,
            'skill_depth_score': 50.0,
            'technical_text_match': 50.0,
            'total_relevant_skills': 0,
            'has_github_mention': False,
            
            # Career features
            'career_score': 50.0,
            'experience_fit_score': 50.0,
            'industry_relevance_score': 50.0,
            'company_type_score': 50.0,
            'role_progression_score': 50.0,
            'current_role_score': 50.0,
            'total_years_experience': 0,
            'product_experience_ratio': 0.0,
            'current_industry': '',
            
            # Behavioral features
            'behavioral_score': 50.0,
            'availability_score': 50.0,
            'engagement_score': 50.0,
            'github_activity_score': 0.0,
            'reliability_score': 50.0,
            'profile_quality_score': 50.0,
            'is_open_to_work': False,
            'notice_period_days': None,
            'has_github_activity': False,
            
            # Trajectory features
            'trajectory_score': 50.0,
            'promotion_score': 50.0,
            'stability_score': 50.0,
            'consistency_score': 50.0,
            'growth_velocity_score': 50.0,
            'salary_progression_score': 50.0,
            'average_job_duration_months': 0,
            'total_promotions': 0,
            'career_gaps_months': 0,
            'is_career_changer': False,
            
            # Credibility features
            'credibility_score': 50.0,
            'skill_credibility_score': 50.0,
            'experience_consistency_score': 50.0,
            'github_credibility_score': 50.0,
            'profile_credibility_score': 50.0,
            'honeypot_risk_score': 50.0,
            'has_credibility_flags': False,
            'suspicious_skill_claims': 0,
            'impossible_timelines': 0,
            
            # Composite scores
            'overall_candidate_score': 50.0,
            'hiring_likelihood_score': 50.0,
            'job_fit_score': 50.0,
            'candidate_quality_score': 50.0
        }
    
    def _save_intermediate_results(self, features_list: List[Dict[str, Any]], 
                                  output_dir: str, batch_end: int):
        """Save intermediate results during processing"""
        
        try:
            temp_df = pd.DataFrame(features_list)
            temp_path = Path(output_dir) / f'candidate_features_temp_{batch_end}.parquet'
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_df.to_parquet(temp_path, index=False)
            print(f"  Saved intermediate results: {temp_path}")
        except Exception as e:
            print(f"  Warning: Could not save intermediate results: {e}")
    
    def _print_feature_summary(self, features_df: pd.DataFrame):
        """Print summary statistics of extracted features"""
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        
        # Main score statistics
        main_scores = [
            'overall_candidate_score', 'technical_fit_score', 'career_score',
            'behavioral_score', 'trajectory_score', 'credibility_score'
        ]
        
        print("\nMain Score Statistics:")
        for score in main_scores:
            if score in features_df.columns:
                mean_score = features_df[score].mean()
                std_score = features_df[score].std()
                min_score = features_df[score].min()
                max_score = features_df[score].max()
                print(f"  {score:25s}: μ={mean_score:5.1f}, σ={std_score:5.1f}, range=[{min_score:5.1f}, {max_score:5.1f}]")
        
        # Top candidates
        print(f"\nTop 10 Candidates by Overall Score:")
        top_candidates = features_df.nlargest(10, 'overall_candidate_score')[
            ['candidate_id', 'overall_candidate_score', 'technical_fit_score', 'behavioral_score']
        ]
        for idx, row in top_candidates.iterrows():
            print(f"  {row['candidate_id']:15s}: Overall={row['overall_candidate_score']:5.1f}, "
                  f"Technical={row['technical_fit_score']:5.1f}, Behavioral={row['behavioral_score']:5.1f}")
        
        # Feature availability
        print(f"\nFeature Availability:")
        feature_counts = {
            'Has GitHub Activity': (features_df['has_github_activity'] == True).sum(),
            'Open to Work': (features_df['is_open_to_work'] == True).sum(),
            'High Credibility (>80)': (features_df['credibility_score'] > 80).sum(),
            'High Technical Fit (>70)': (features_df['technical_fit_score'] > 70).sum(),
            'Career Changers': (features_df['is_career_changer'] == True).sum()
        }
        
        for feature, count in feature_counts.items():
            percentage = (count / len(features_df)) * 100
            print(f"  {feature:25s}: {count:5d} candidates ({percentage:5.1f}%)")
        
        print("\n" + "="*60)
    
    def create_job_summary_report(self, features_df: pd.DataFrame, 
                                 output_path: str = 'Data/processed/job_matching_report.md') -> str:
        """
        Create a summary report of how well candidates match the job
        
        Args:
            features_df: DataFrame with extracted features
            output_path: Path to save the report
            
        Returns:
            Report content as string
        """
        
        # Calculate job matching statistics
        high_fit_candidates = features_df[features_df['job_fit_score'] > 75]
        medium_fit_candidates = features_df[(features_df['job_fit_score'] >= 50) & (features_df['job_fit_score'] <= 75)]
        low_fit_candidates = features_df[features_df['job_fit_score'] < 50]
        
        report_content = f"""# Job Matching Report

## Job Requirements
- **Role**: {self.job_requirements.role_title}
- **Level**: {self.job_requirements.role_level}
- **Type**: {self.job_requirements.role_type}
- **Required Skills**: {', '.join(self.job_requirements.required_skills) if self.job_requirements.required_skills else 'None specified'}
- **Preferred Skills**: {', '.join(self.job_requirements.preferred_skills) if self.job_requirements.preferred_skills else 'None specified'}
- **Experience Range**: {self.job_requirements.min_years_experience or 'No minimum'} - {self.job_requirements.max_years_experience or 'No maximum'} years

## Candidate Pool Analysis
- **Total Candidates Analyzed**: {len(features_df):,}
- **High Job Fit (>75)**: {len(high_fit_candidates):,} candidates ({len(high_fit_candidates)/len(features_df)*100:.1f}%)
- **Medium Job Fit (50-75)**: {len(medium_fit_candidates):,} candidates ({len(medium_fit_candidates)/len(features_df)*100:.1f}%)
- **Low Job Fit (<50)**: {len(low_fit_candidates):,} candidates ({len(low_fit_candidates)/len(features_df)*100:.1f}%)

## Score Distribution
- **Average Overall Score**: {features_df['overall_candidate_score'].mean():.1f}
- **Average Technical Fit**: {features_df['technical_fit_score'].mean():.1f}
- **Average Career Fit**: {features_df['career_score'].mean():.1f}
- **Average Behavioral Score**: {features_df['behavioral_score'].mean():.1f}

## Top 20 Recommended Candidates
"""
        
        top_candidates = features_df.nlargest(20, 'overall_candidate_score')
        for i, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
            report_content += f"""
### {i}. {candidate['candidate_id']}
- **Overall Score**: {candidate['overall_candidate_score']:.1f}
- **Technical Fit**: {candidate['technical_fit_score']:.1f}
- **Career Score**: {candidate['career_score']:.1f}
- **Behavioral Score**: {candidate['behavioral_score']:.1f}
- **Hiring Likelihood**: {candidate['hiring_likelihood_score']:.1f}
"""
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Job matching report saved to: {output_path}")
        return report_content
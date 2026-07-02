#!/usr/bin/env python3
"""
Trust Engine - Master orchestrator for Phase 5 Trust & Explainability
Combines confidence, reasoning, honeypot detection, and fairness audit
"""

import sys
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.job_requirements.job_schema import JobRequirements
from src.explainability.confidence_engine import ConfidenceEngine
from src.explainability.reasoning_engine import ReasoningEngine
from src.explainability.honeypot_detector import HoneypotDetector
from src.explainability.fairness_audit import FairnessAuditor


class TrustEngine:
    """
    Master trust and explainability engine that orchestrates all Phase 5 components.
    Transforms opaque AI rankings into transparent, trustworthy decisions.
    """
    
    def __init__(self, job_requirements: JobRequirements, jd_profile: Dict[str, Any]):
        self.job_requirements = job_requirements
        self.jd_profile = jd_profile
        
        # Initialize all engines
        self.confidence_engine = ConfidenceEngine(job_requirements)
        self.reasoning_engine = ReasoningEngine(job_requirements, jd_profile)
        self.honeypot_detector = HoneypotDetector(job_requirements)
        self.fairness_auditor = FairnessAuditor(job_requirements)
        
        print(f"Trust Engine initialized for: {job_requirements.role_title}")
    
    def enhance_with_trust(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete trust enhancement pipeline
        
        Args:
            candidates_df: DataFrame with candidate rankings and scores
            
        Returns:
            DataFrame enhanced with trust and explainability features
        """
        
        print("\n🔒 TRUST & EXPLAINABILITY ENGINE")
        print("="*60)
        print(f"Enhancing {len(candidates_df)} candidates with trust features...")
        
        start_time = time.time()
        
        # Step 1: Calculate Confidence Scores
        print("\n🎯 STEP 1: CALCULATING CONFIDENCE SCORES")
        print("-" * 40)
        candidates_df = self.confidence_engine.calculate_confidence_scores(candidates_df)
        
        # Step 2: Generate Reasoning Explanations
        print("\n🧠 STEP 2: GENERATING REASONING EXPLANATIONS")
        print("-" * 40)
        candidates_df = self.reasoning_engine.generate_reasoning(candidates_df)
        
        # Step 3: Detect Honeypots
        print("\n🕵️ STEP 3: DETECTING HONEYPOTS")
        print("-" * 40)
        candidates_df = self.honeypot_detector.detect_honeypots(candidates_df)
        
        # Step 4: Conduct Fairness Audit
        print("\n⚖️  STEP 4: CONDUCTING FAIRNESS AUDIT")
        print("-" * 40)
        top_candidates = candidates_df.nlargest(100, 'hybrid_score')
        fairness_results = self.fairness_auditor.conduct_fairness_audit(candidates_df, top_candidates)
        
        # Add fairness results to DataFrame metadata
        candidates_df.attrs['fairness_audit'] = fairness_results
        
        # Step 5: Generate Trust Scores
        print("\n🛡️  STEP 5: CALCULATING TRUST SCORES")
        print("-" * 40)
        candidates_df = self._calculate_trust_scores(candidates_df)
        
        # Step 6: Create Explainable Output Format
        print("\n📋 STEP 6: CREATING EXPLAINABLE OUTPUT")
        print("-" * 40)
        candidates_df = self._create_explainable_output(candidates_df)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ TRUST ENHANCEMENT COMPLETED!")
        print(f"⏱️  Processing time: {total_time:.2f} seconds")
        print(f"🎯 Confidence scores: ✓")
        print(f"🧠 Reasoning explanations: ✓") 
        print(f"🕵️ Honeypot detection: ✓")
        print(f"⚖️  Fairness audit: ✓")
        print(f"🛡️  Trust scores: ✓")
        
        return candidates_df
    
    def _calculate_trust_scores(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall trust scores for each candidate"""
        
        trust_scores = []
        trust_levels = []
        
        for idx, candidate in candidates_df.iterrows():
            # Get component scores
            confidence_score = candidate.get('confidence_score', 50.0)
            credibility_score = candidate.get('enhanced_credibility_score', 
                                            candidate.get('credibility_score', 80.0))
            honeypot_risk = candidate.get('honeypot_risk_score', 0.0)
            
            # Calculate trust score
            # High confidence + high credibility + low honeypot risk = high trust
            base_trust = (confidence_score * 0.4 + credibility_score * 0.6)
            honeypot_penalty = honeypot_risk * 0.5  # Reduce trust based on honeypot risk
            
            trust_score = max(0.0, base_trust - honeypot_penalty)
            
            # Categorize trust level
            if trust_score >= 85.0:
                trust_level = 'Very High'
            elif trust_score >= 75.0:
                trust_level = 'High'
            elif trust_score >= 60.0:
                trust_level = 'Medium'
            elif trust_score >= 45.0:
                trust_level = 'Low'
            else:
                trust_level = 'Very Low'
            
            trust_scores.append(trust_score)
            trust_levels.append(trust_level)
        
        candidates_df['trust_score'] = trust_scores
        candidates_df['trust_level'] = trust_levels
        
        # Summary
        avg_trust = pd.Series(trust_scores).mean()
        high_trust_count = sum(1 for score in trust_scores if score >= 75.0)
        
        print(f"Average trust score: {avg_trust:.1f}")
        print(f"High trust candidates: {high_trust_count}/{len(candidates_df)} ({high_trust_count/len(candidates_df)*100:.1f}%)")
        
        return candidates_df
    
    def _create_explainable_output(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Create human-readable explainable output format"""
        
        explainable_outputs = []
        
        for idx, candidate in candidates_df.iterrows():
            # Get reasoning explanation
            reasoning = candidate.get('reasoning_explanation', {})
            
            # Create explainable format
            explainable_output = {
                'candidate_id': candidate['candidate_id'],
                'match_score': round(candidate.get('hybrid_score', 0), 1),
                'confidence': f"{candidate.get('confidence_score', 50):.0f}%",
                'trust_level': candidate.get('trust_level', 'Medium'),
                'recommendation': reasoning.get('recommendation', 'Needs review'),
                
                # Strengths and concerns
                'why_good': reasoning.get('strengths', [])[:3],  # Top 3 strengths
                'concerns': reasoning.get('concerns', [])[:2],   # Top 2 concerns
                
                # Key metrics
                'technical_fit': f"{candidate.get('technical_fit_score', 0):.0f}/100",
                'experience_match': f"{candidate.get('experience_fit_score', 50):.0f}/100",
                'availability': f"{candidate.get('availability_score', 50):.0f}/100",
                
                # Trust indicators
                'verified': not candidate.get('honeypot_flag', False),
                'data_quality': candidate.get('confidence_level', 'Medium'),
                
                # Summary
                'summary': reasoning.get('overall_assessment', 'Candidate assessment unavailable')
            }
            
            explainable_outputs.append(explainable_output)
        
        candidates_df['explainable_output'] = explainable_outputs
        
        return candidates_df
    
    def analyze_trust_results(self, candidates_df: pd.DataFrame, top_k: int = 20):
        """Analyze trust enhancement results"""
        
        print("\n" + "="*70)
        print("TRUST & EXPLAINABILITY ANALYSIS")
        print("="*70)
        
        # Overall statistics
        total_candidates = len(candidates_df)
        avg_confidence = candidates_df['confidence_score'].mean()
        avg_trust = candidates_df['trust_score'].mean()
        honeypots_detected = candidates_df['honeypot_flag'].sum()
        
        print(f"📊 TRUST STATISTICS:")
        print(f"   Total Candidates: {total_candidates:,}")
        print(f"   Average Confidence: {avg_confidence:.1f}%")
        print(f"   Average Trust Score: {avg_trust:.1f}%")
        print(f"   Honeypots Detected: {honeypots_detected} ({honeypots_detected/total_candidates*100:.1f}%)")
        
        # Trust level distribution
        trust_distribution = candidates_df['trust_level'].value_counts()
        print(f"\n🛡️  TRUST LEVEL DISTRIBUTION:")
        for level, count in trust_distribution.items():
            percentage = count / total_candidates * 100
            print(f"   {level:10s}: {count:3d} candidates ({percentage:5.1f}%)")
        
        # Top candidates with explainable output
        top_candidates = candidates_df.nlargest(top_k, 'hybrid_score')
        
        print(f"\n🏆 TOP {top_k} CANDIDATES WITH EXPLANATIONS:")
        print("="*70)
        
        for i, (idx, candidate) in enumerate(top_candidates.iterrows(), 1):
            output = candidate['explainable_output']
            
            print(f"\n{i:2d}. {output['candidate_id']} (Score: {output['match_score']})")
            print(f"    {output['summary']}")
            print(f"    📊 Confidence: {output['confidence']} | Trust: {output['trust_level']} | Verified: {'✓' if output['verified'] else '✗'}")
            
            if output['why_good']:
                print(f"    ✓ Strengths: {', '.join(output['why_good'][:2])}")
            
            if output['concerns']:
                print(f"    ⚠ Concerns: {', '.join(output['concerns'][:2])}")
            
            print(f"    💼 Recommendation: {output['recommendation']}")
        
        # Quality assessment
        high_confidence = (candidates_df['confidence_score'] >= 75).sum()
        high_trust = (candidates_df['trust_score'] >= 75).sum()
        verified = (~candidates_df['honeypot_flag']).sum()
        
        print(f"\n📈 QUALITY METRICS:")
        print(f"   High Confidence (≥75%): {high_confidence}/{total_candidates} ({high_confidence/total_candidates*100:.1f}%)")
        print(f"   High Trust (≥75%): {high_trust}/{total_candidates} ({high_trust/total_candidates*100:.1f}%)")
        print(f"   Verified Candidates: {verified}/{total_candidates} ({verified/total_candidates*100:.1f}%)")
        
        # Fairness summary
        if 'fairness_audit' in candidates_df.attrs:
            fairness = candidates_df.attrs['fairness_audit']['fairness_assessment']
            print(f"\n⚖️  FAIRNESS ASSESSMENT:")
            print(f"   Overall Fairness: {fairness['overall_fairness_score']:.1f}/100 ({fairness['fairness_level']})")
            
            if fairness['identified_issues']:
                print(f"   Issues: {len(fairness['identified_issues'])} identified")
            else:
                print(f"   Issues: None identified")
    
    def export_trust_results(self, candidates_df: pd.DataFrame, 
                           output_dir: str = 'Data/processed') -> Dict[str, Path]:
        """Export all trust and explainability results"""
        
        output_paths = {}
        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # 1. Export explainable rankings
        explainable_rankings = []
        for idx, candidate in candidates_df.nlargest(100, 'hybrid_score').iterrows():
            output = candidate['explainable_output']
            explainable_rankings.append({
                'rank': len(explainable_rankings) + 1,
                'candidate_id': output['candidate_id'],
                'match_score': output['match_score'],
                'summary': output['summary'],
                'strengths': '; '.join(output['why_good']),
                'concerns': '; '.join(output['concerns']),
                'confidence': output['confidence'],
                'trust_level': output['trust_level'],
                'verified': output['verified'],
                'recommendation': output['recommendation']
            })
        
        explainable_df = pd.DataFrame(explainable_rankings)
        explainable_path = output_directory / 'explainable_rankings.csv'
        explainable_df.to_csv(explainable_path, index=False)
        output_paths['explainable_rankings'] = explainable_path
        
        # 2. Export confidence analysis
        confidence_summary = self.confidence_engine.get_confidence_summary(candidates_df)
        confidence_path = output_directory / 'confidence_analysis.json'
        with open(confidence_path, 'w') as f:
            json.dump(confidence_summary, f, indent=2, default=str)
        output_paths['confidence_analysis'] = confidence_path
        
        # 3. Export reasoning report
        reasoning_path = self.reasoning_engine.export_reasoning_report(candidates_df)
        output_paths['reasoning_report'] = reasoning_path
        
        # 4. Export honeypot report
        honeypot_path = self.honeypot_detector.generate_honeypot_report(candidates_df)
        output_paths['honeypot_report'] = honeypot_path
        
        # 5. Export fairness report
        if 'fairness_audit' in candidates_df.attrs:
            fairness_path = self.fairness_auditor.generate_fairness_report(
                candidates_df.attrs['fairness_audit']
            )
            output_paths['fairness_report'] = fairness_path
        
        # 6. Export complete trust dataset
        trust_columns = [
            'candidate_id', 'hybrid_score', 'confidence_score', 'trust_score',
            'trust_level', 'honeypot_flag', 'honeypot_risk_score', 'reasoning_summary'
        ]
        
        available_columns = [col for col in trust_columns if col in candidates_df.columns]
        trust_df = candidates_df[available_columns].copy()
        
        trust_path = output_directory / 'trust_enhanced_candidates.parquet'
        trust_df.to_parquet(trust_path, index=False)
        output_paths['trust_dataset'] = trust_path
        
        print(f"\n📁 TRUST RESULTS EXPORTED:")
        for result_type, path in output_paths.items():
            print(f"   {result_type}: {path}")
        
        return output_paths
    
    def run_evaluation_checklist(self, candidates_df: pd.DataFrame):
        """Run Phase 5 evaluation checklist"""
        
        print("\n" + "="*60)
        print("PHASE 5 EVALUATION CHECKLIST")
        print("="*60)
        
        top_candidates = candidates_df.nlargest(10, 'hybrid_score')
        
        print(f"\n📝 EVALUATION QUESTIONS:")
        
        print(f"\n1. ❓ WHY RANK 1?")
        if len(top_candidates) > 0:
            rank1 = top_candidates.iloc[0]
            explanation = rank1.get('reasoning_explanation', {})
            print(f"   Candidate: {rank1['candidate_id']}")
            print(f"   Score: {rank1.get('hybrid_score', 0):.1f}")
            print(f"   Assessment: {explanation.get('overall_assessment', 'N/A')}")
            print(f"   Strengths: {', '.join(explanation.get('strengths', [])[:2])}")
            print(f"   Confidence: {rank1.get('confidence_score', 50):.0f}%")
        
        print(f"\n2. ❓ WHY RANK 2?")
        if len(top_candidates) > 1:
            rank2 = top_candidates.iloc[1]
            explanation = rank2.get('reasoning_explanation', {})
            print(f"   Candidate: {rank2['candidate_id']}")
            print(f"   Score: {rank2.get('hybrid_score', 0):.1f}")
            print(f"   Assessment: {explanation.get('overall_assessment', 'N/A')}")
            print(f"   Strengths: {', '.join(explanation.get('strengths', [])[:2])}")
            print(f"   Confidence: {rank2.get('confidence_score', 50):.0f}%")
        
        print(f"\n3. ❓ WHY RANK 3?")
        if len(top_candidates) > 2:
            rank3 = top_candidates.iloc[2]
            explanation = rank3.get('reasoning_explanation', {})
            print(f"   Candidate: {rank3['candidate_id']}")
            print(f"   Score: {rank3.get('hybrid_score', 0):.1f}")
            print(f"   Assessment: {explanation.get('overall_assessment', 'N/A')}")
            print(f"   Strengths: {', '.join(explanation.get('strengths', [])[:2])}")
            print(f"   Confidence: {rank3.get('confidence_score', 50):.0f}%")
        
        print(f"\n4. 🛡️  CAN YOU DEFEND THESE RANKINGS?")
        print(f"   Each ranking now comes with:")
        print(f"   ✓ Confidence score (how sure we are)")
        print(f"   ✓ Detailed reasoning (why this ranking)")
        print(f"   ✓ Trust verification (honeypot detection)")
        print(f"   ✓ Fairness audit (bias detection)")
        
        print(f"\n✅ PHASE 5 DELIVERABLES COMPLETE:")
        print(f"   🎯 confidence_score: Every candidate has confidence level")
        print(f"   🧠 reasoning: Human-readable explanations generated")
        print(f"   🕵️ credibility_score: Enhanced with honeypot detection")
        print(f"   🚩 honeypot_flag: Fake candidates identified")
        print(f"   ⚖️  fairness_report.md: Bias analysis completed")


if __name__ == "__main__":
    print("Trust Engine - Master Trust & Explainability Orchestrator")
    print("This module transforms opaque rankings into transparent, trustworthy decisions.")
    print("\nExample usage:")
    print("1. Load candidates with hybrid rankings")
    print("2. Load JD profile from Phase 4")
    print("3. engine = TrustEngine(job_requirements, jd_profile)")
    print("4. candidates_df = engine.enhance_with_trust(candidates_df)")
    print("5. engine.analyze_trust_results(candidates_df)")
    print("6. engine.export_trust_results(candidates_df)")
    print("7. engine.run_evaluation_checklist(candidates_df)")
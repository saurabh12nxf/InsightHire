# TalentLens AI

An intelligent candidate discovery and ranking system for AI/ML roles.

## Project Structure

```
talentlens-ai/
├── Data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Processed data files
├── notebooks/                  # Jupyter notebooks for analysis
├── src/
│   ├── data_loader/           # Data loading utilities
│   ├── feature_engineering/   # Feature extraction and processing
│   ├── embeddings/            # Embedding generation
│   ├── rankings/              # Candidate ranking algorithms
│   ├── reasoning/             # Reasoning and explanation generation
│   └── validation/            # Validation utilities
├── outputs/                   # Generated outputs and results
├── models/                    # Trained models and weights
└── README.md
```

## Phase 1 - Project Setup ✅

**Goal**: Make sure the dataset loads.

**Status**: Complete

### Checkpoint Test
```bash
python load_candidates.py
```

**Expected Output**:
```
Loaded 100000 candidates
✅ Phase 1 complete
```

### What was accomplished:
- ✅ Created complete project structure
- ✅ Implemented candidate data loader
- ✅ Validated dataset structure (100,000 candidates)
- ✅ Verified data integrity and format
- ✅ Set up modular architecture for future phases

## Getting Started

1. Ensure Python 3.8+ is installed
2. Run the Phase 1 checkpoint: `python load_candidates.py`
3. Verify you see "Loaded 100000 candidates" message

## Dataset Overview

The dataset contains 100,000 candidate profiles with:
- **Profile Information**: Name, headline, summary, location, experience
- **Career History**: Previous roles, companies, durations
- **Education**: Degrees, institutions, grades
- **Skills**: Technical skills with proficiency levels
- **Redrob Signals**: Platform engagement metrics

**Sample Candidate Fields**:
- candidate_id (CAND_XXXXXXX format)
- profile (basic info and current role)
- career_history (up to 10 previous positions)
- education (academic background)
- skills (technical competencies)
- redrob_signals (engagement metrics

## BY SAURABH SINGH 

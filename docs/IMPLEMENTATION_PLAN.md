# BenchDrift v2: Relevance Selection Implementation Plan

## Overview
This document outlines the minimum changes needed to add relevance-based variation selection to the BenchDrift pipeline.

## Goals
1. Select relevant transformations per problem (not apply all)
2. Score and rank transformations by relevance
3. Detect gaps in taxonomy coverage
4. Generate actionable output (breaks/fixes/works)

## Architecture

### Current Pipeline Flow
```
Problem → VariationGenerator (all transforms) → Validation → Response → Evaluation
```

### New Pipeline Flow
```
Problem → RelevanceSelector → Filtered Transforms → VariationGenerator → Validation → Response → OutcomeClassifier → Report
```

## Minimum Changes Required

### 1. NEW FILE: `src/benchdrift/pipeline/relevance_selector.py`
**Purpose**: Select relevant transformations based on problem analysis

**Components**:
- `TaxonomyIndex`: Load and embed transformation definitions
- `ProblemAnalyzer`: Extract features from input problem
- `RelevanceScorer`: Score transformations by relevance
- `GapDetector`: Identify when coverage is poor

**Key Functions**:
```python
class RelevanceSelector:
    def __init__(self, taxonomy_path: str)
    def select_transformations(self, problem: str, top_k: int = 10) -> List[RankedTransformation]
    def detect_gaps(self, problem: str, scores: List[float]) -> GapReport
```

### 2. MODIFY: `src/benchdrift/pipeline/unified_variation_engine_batched.py`
**Change**: Accept optional `transformations` parameter

**Before**:
```python
def generate_comprehensive_variations(self, problem: str, ...):
    # Always generates all variation types
```

**After**:
```python
def generate_comprehensive_variations(self, problem: str,
                                      transformations: List[str] = None,  # NEW
                                      ...):
    if transformations:
        # Only generate specified transformations
    else:
        # Existing behavior (all transformations)
```

### 3. NEW FILE: `src/benchdrift/pipeline/outcome_classifier.py`
**Purpose**: Classify results into actionable categories

**Functions**:
```python
def classify_outcomes(baseline_result, variation_results, ground_truth) -> OutcomeReport:
    """
    Returns:
        - breaks_it: Variations where model failed (baseline correct)
        - fixes_it: Variations where model succeeded (baseline wrong)
        - also_works: Robust variations
        - also_fails: Consistent failures
    """
```

### 4. OPTIONAL: Pipeline entry point wrapper
**File**: `scripts/run_with_relevance.py`
**Purpose**: End-to-end wrapper using new relevance selection

## Implementation Order

### Phase 1: Core Relevance Selector (This Session)
1. Create `relevance_selector.py` with:
   - Taxonomy loading
   - Feature extraction (regex-based)
   - Simple scoring (feature overlap + domain matching)
   - Basic gap detection

2. Test standalone on sample problems

### Phase 2: Pipeline Integration
1. Modify `unified_variation_engine_batched.py` to accept filters
2. Add entry point wrapper
3. Test end-to-end

### Phase 3: Outcome Classification
1. Create `outcome_classifier.py`
2. Integrate with pipeline output
3. Generate actionable reports

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `relevance_selector.py` | NEW | Core relevance selection logic |
| `outcome_classifier.py` | NEW | Outcome classification |
| `unified_variation_engine_batched.py` | MODIFY | Accept filtered transforms |
| `config/taxonomy/transformations.yaml` | NEW | Searchable taxonomy (DONE) |

## Risk Assessment

### Low Risk Changes
- Adding new files (no existing code affected)
- Taxonomy configuration

### Medium Risk Changes
- Modifying `unified_variation_engine_batched.py`
  - Mitigation: Keep existing behavior as default
  - Add `transformations=None` parameter (backward compatible)

### Testing Strategy
1. Unit test relevance selector standalone
2. Integration test with single problem
3. Full pipeline test with small benchmark

## Dependencies
- `sentence-transformers` (for embeddings) - already used in codebase
- `pyyaml` (for taxonomy loading) - standard
- Existing model clients (RITS, VLLM)

## Next Steps
1. Implement `relevance_selector.py`
2. Test with sample problems
3. Integrate with pipeline

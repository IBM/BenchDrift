# MATH-Hard Dataset

High-difficulty high school competition mathematics problems from the MATH dataset.

## Source

- **HuggingFace**: [lighteval/MATH-Hard](https://huggingface.co/datasets/lighteval/MATH-Hard)
- **Paper**: [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874) (Hendrycks et al., 2021)
- **Difficulty**: Level 5 only (hardest problems from the full MATH dataset)

## Dataset Statistics

### Test Split
- **Total**: 1,324 problems
- **All Level 5**: Hardest difficulty tier
- **7 Subjects**: Stratified sampling across math domains

#### Problems by Subject
- Algebra: 307 problems (23%)
- Intermediate Algebra: 280 problems (21%)
- Prealgebra: 193 problems (15%)
- Number Theory: 154 problems (12%)
- Precalculus: 135 problems (10%)
- Geometry: 132 problems (10%)
- Counting & Probability: 123 problems (9%)

### Train Split
- **Total**: 2,304 problems
- **All Level 5**: Hardest difficulty tier

## Format

JSONL format with the following fields:

```json
{
    "question": "There are numbers $A$ and $B$ for which...",
    "answer": "We can approach this problem by picking clever values...",
    "level": "Level 5",
    "subject": "Algebra"
}
```

### Fields
- **question** (required): The problem statement
- **answer** (required): Step-by-step solution with final answer in `\\boxed{}`
- **level** (metadata): Always "Level 5" (hardest)
- **subject** (metadata): One of 7 math subjects

## Files

- `test.jsonl`: Full test set (1,324 problems)
- `test_500.jsonl`: Stratified sample (500 problems)
- `train.jsonl`: Full training set (2,304 problems)

## Usage

### Load Test Set
```python
import json

problems = []
with open('math-hard/test_500.jsonl') as f:
    for line in f:
        problems.append(json.loads(line))

print(f"Loaded {len(problems)} problems")
# Output: Loaded 500 problems
```

### Use with Variation Pipeline
```bash
python unified_batched_pipeline.py \
    --input-problems evaluation_suite/data/math-hard/test_500.jsonl \
    --stage 0 \
    --output-dir results/math-hard
```

## Download Script

To re-download or create custom samples:

```bash
# Download full dataset
python download_math_hard.py --download

# Create 500-problem sample
python download_math_hard.py --download --sample --sample-size 500

# Create custom sample from existing download
python download_math_hard.py --sample --sample-size 300
```

## Comparison with Other Benchmarks

| Benchmark | Domain | Difficulty | Problems | Format |
|-----------|--------|------------|----------|--------|
| **MATH-Hard** | High school math | Level 5 (hardest) | 1,324 | Open-ended |
| GSM8K | Grade school math | Easy-Medium | 1,319 | Word problems |
| MMLU | Multi-subject | Varies | 14,042 | Multiple choice |
| TOT | Temporal arithmetic | Medium | 1,000+ | Open-ended |

## Key Characteristics

1. **High Difficulty**: Only Level 5 problems (5% hardest from full MATH)
2. **Competition Math**: Problems from AMC, AIME, etc.
3. **LaTeX Heavy**: Extensive use of mathematical notation
4. **Step-by-Step Solutions**: Detailed reasoning paths
5. **\\boxed{} Answers**: Final answers marked clearly

## Example Problem

**Question**:
```
What is the range of the function $y = \frac{x^2 + 3x + 2}{x+1}$?
```

**Answer**:
```
We can factor the numerator to get $y = \frac{(x+1)(x+2)}{x+1}$.
If we exclude the case where $x = -1$, we can cancel to get $y = x+2$.
But we still need to exclude $x = -1$, which means we exclude $y = 1$.
So the range is $\\boxed{(-\\infty, 1) \\cup (1, \\infty)}$.
```

## Citation

```bibtex
@article{hendrycks2021math,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2103.03874},
  year={2021}
}
```

#!/usr/bin/env python3
"""
Create sampled versions of MMLU and TOT-Arithmetic benchmarks.

- MMLU: Sample 1000 questions stratified across all subjects
- TOT-Arithmetic: Sample 1000 questions (all are temporal arithmetic)
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)  # For reproducibility


def sample_mmlu(input_file: Path, output_file: Path, n_samples: int = 1000):
    """Sample MMLU questions stratified by subject"""
    print(f"📥 Loading MMLU from {input_file}...")

    # Load all questions
    questions = []
    with open(input_file) as f:
        for line in f:
            questions.append(json.loads(line))

    total = len(questions)
    print(f"   Total questions: {total:,}")

    # Check if MMLU has subject field
    has_subject = 'subject' in questions[0] if questions else False

    if has_subject:
        # Group by subject for stratified sampling
        by_subject = {}
        for q in questions:
            subject = q.get('subject', 'unknown')
            if subject not in by_subject:
                by_subject[subject] = []
            by_subject[subject].append(q)

        print(f"   Found {len(by_subject)} subjects")

        # Sample proportionally from each subject
        sampled = []
        for subject, subject_qs in sorted(by_subject.items()):
            # Sample proportionally
            n_from_subject = max(1, int(n_samples * len(subject_qs) / total))
            subject_sample = random.sample(subject_qs, min(n_from_subject, len(subject_qs)))
            sampled.extend(subject_sample)
            print(f"     {subject}: {len(subject_qs)} → {len(subject_sample)}")

        # If we're over, trim randomly; if under, add more
        if len(sampled) > n_samples:
            sampled = random.sample(sampled, n_samples)
        elif len(sampled) < n_samples:
            remaining = [q for q in questions if q not in sampled]
            extra = random.sample(remaining, min(n_samples - len(sampled), len(remaining)))
            sampled.extend(extra)
    else:
        # No subject field, just random sample
        print("   No subject field found, random sampling...")
        sampled = random.sample(questions, min(n_samples, len(questions)))

    # Shuffle to mix subjects
    random.shuffle(sampled)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Saved {len(sampled)} MMLU questions to {output_file}")
    return len(sampled)


def sample_tot_arithmetic(input_file: Path, output_file: Path, n_samples: int = 1000):
    """Sample TOT-Arithmetic questions"""
    print(f"\n📥 Loading TOT-Arithmetic from {input_file}...")

    # Load all questions
    questions = []
    with open(input_file) as f:
        for line in f:
            questions.append(json.loads(line))

    total = len(questions)
    print(f"   Total questions: {total:,}")

    # Random sample
    sampled = random.sample(questions, min(n_samples, len(questions)))

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Saved {len(sampled)} TOT-Arithmetic questions to {output_file}")
    return len(sampled)


if __name__ == "__main__":
    print("🚀 Creating sampled benchmark datasets\n")
    print("=" * 60)

    data_dir = Path("data")

    # Sample MMLU
    mmlu_count = sample_mmlu(
        input_file=data_dir / "mmlu" / "test.jsonl",
        output_file=data_dir / "mmlu" / "test_sampled_1000.jsonl",
        n_samples=1000
    )

    # Sample TOT-Arithmetic
    tot_count = sample_tot_arithmetic(
        input_file=data_dir / "tot" / "arithmetic.jsonl",
        output_file=data_dir / "tot" / "test_sampled_1000.jsonl",
        n_samples=1000
    )

    print("\n" + "=" * 60)
    print("✅ All sampled datasets created!")
    print(f"\n📊 Summary:")
    print(f"  • MMLU (sampled): {mmlu_count:,} questions")
    print(f"  • TOT-Arithmetic (sampled): {tot_count:,} questions")
    print(f"\nTotal: {mmlu_count + tot_count:,} questions")
    print(f"\nFiles created:")
    print(f"  • data/mmlu/test_sampled_1000.jsonl")
    print(f"  • data/tot/test_sampled_1000.jsonl")

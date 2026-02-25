#!/usr/bin/env python3
"""
Download and format the MATH-Hard dataset from lighteval.

Source: https://huggingface.co/datasets/lighteval/MATH-Hard
MATH-Hard is a challenging subset of high school competition mathematics problems.
"""

import json
import random
from pathlib import Path
from collections import Counter

# Try to import datasets library
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  'datasets' library not found. Install with: pip install datasets")


def download_and_format_math_hard(output_dir: Path):
    """
    Download MATH-Hard dataset from HuggingFace and format to match our schema.

    Our format:
    {"question": "...", "answer": "..."}
    """

    if not HAS_DATASETS:
        print("❌ Cannot download without 'datasets' library")
        print("   Install with: pip install datasets")
        return False

    print("🚀 Downloading MATH-Hard dataset from HuggingFace...")
    print("   Source: lighteval/MATH-Hard")
    print("   This may take a few minutes on first run...")

    try:
        # Download dataset
        dataset = load_dataset("lighteval/MATH-Hard")

        print(f"\n📦 Available splits: {list(dataset.keys())}")

        # Process all available splits
        for split_name in dataset.keys():
            print(f"\n📥 Processing {split_name} split...")
            data = dataset[split_name]

            # Examine first item to understand structure
            if len(data) > 0:
                print(f"\n🔍 Dataset structure (first item):")
                first_item = data[0]
                print(f"   Keys: {list(first_item.keys())}")
                for key, value in first_item.items():
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                    else:
                        preview = str(value)
                    print(f"   • {key}: {preview}")

            # Count by subject/type if available
            subjects = Counter()
            levels = Counter()

            # Convert to our format
            formatted = []
            for item in data:
                # Try different possible field names
                problem = item.get('problem') or item.get('question') or item.get('input') or ""
                solution = item.get('solution') or item.get('answer') or item.get('output') or ""
                level = item.get('level', None)
                subject = item.get('type') or item.get('subject') or item.get('category', None)

                if subject:
                    subjects[subject] += 1
                if level:
                    levels[level] += 1

                # Format to match our schema
                formatted_item = {
                    'question': problem,
                    'answer': solution,
                }

                # Add metadata if available
                if level is not None:
                    formatted_item['level'] = level
                if subject is not None:
                    formatted_item['subject'] = subject

                formatted.append(formatted_item)

            # Save
            # Use 'test' as default name for consistency with other benchmarks
            if split_name == 'train':
                output_file = output_dir / "train.jsonl"
            else:
                output_file = output_dir / "test.jsonl"

            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                for item in formatted:
                    f.write(json.dumps(item) + '\n')

            print(f"✅ Saved {len(formatted):,} problems to {output_file}")

            # Print statistics
            print(f"\n📊 Statistics for {split_name}:")
            print(f"   Total problems: {len(formatted):,}")

            if subjects:
                print(f"\n   By subject:")
                for subject, count in sorted(subjects.items(), key=lambda x: -x[1]):
                    print(f"     • {subject}: {count:,}")

            if levels:
                print(f"\n   By difficulty level:")
                for level in sorted(levels.keys()):
                    print(f"     • Level {level}: {levels[level]:,}")

        return True

    except Exception as e:
        print(f"❌ Error downloading MATH-Hard dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def sample_math_hard_dataset(input_file: Path, output_file: Path, n_samples: int = 500):
    """
    Sample MATH-Hard problems stratified by subject and difficulty (if available).

    Strategy:
    - Sample proportionally from each subject (if available)
    - Otherwise, random sample
    """
    print(f"\n📥 Loading MATH-Hard from {input_file}...")

    # Load all questions
    questions = []
    with open(input_file) as f:
        for line in f:
            questions.append(json.loads(line))

    total = len(questions)
    print(f"   Total questions: {total:,}")

    # Check if we have subject/level metadata
    has_subject = 'subject' in questions[0] if questions else False
    has_level = 'level' in questions[0] if questions else False

    if has_subject or has_level:
        # Group by available metadata
        by_group = {}
        for q in questions:
            subject = q.get('subject', 'unknown')
            level = q.get('level', 'unknown')

            if has_subject and has_level:
                key = f"{subject}_L{level}"
            elif has_subject:
                key = subject
            else:
                key = f"L{level}"

            if key not in by_group:
                by_group[key] = []
            by_group[key].append(q)

        print(f"   Found {len(by_group)} groups")

        # Sample proportionally from each group
        sampled = []
        for key, group_qs in sorted(by_group.items()):
            # Sample proportionally
            n_from_group = max(1, int(n_samples * len(group_qs) / total))
            group_sample = random.sample(group_qs, min(n_from_group, len(group_qs)))
            sampled.extend(group_sample)
            print(f"     {key}: {len(group_qs)} → {len(group_sample)}")

        # Adjust to exact sample size
        if len(sampled) > n_samples:
            sampled = random.sample(sampled, n_samples)
        elif len(sampled) < n_samples:
            remaining = [q for q in questions if q not in sampled]
            extra = random.sample(remaining, min(n_samples - len(sampled), len(remaining)))
            sampled.extend(extra)
    else:
        # No metadata, just random sample
        print("   No metadata found, random sampling...")
        sampled = random.sample(questions, min(n_samples, len(questions)))

    # Shuffle to mix groups
    random.shuffle(sampled)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Saved {len(sampled)} MATH-Hard questions to {output_file}")

    # Print statistics of sample
    if has_subject:
        subjects = Counter(q.get('subject', 'unknown') for q in sampled)
        print(f"\n📊 Sample by subject:")
        for subject, count in sorted(subjects.items(), key=lambda x: -x[1]):
            print(f"     • {subject}: {count}")

    if has_level:
        levels = Counter(q.get('level', 'unknown') for q in sampled)
        print(f"\n   Sample by level:")
        for level in sorted(levels.keys()):
            print(f"     • Level {level}: {levels[level]}")

    return len(sampled)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and format MATH-Hard dataset")
    parser.add_argument('--download', action='store_true',
                       help='Download dataset from HuggingFace')
    parser.add_argument('--sample', action='store_true',
                       help='Create sampled version')
    parser.add_argument('--sample-size', type=int, default=500,
                       help='Number of problems to sample (default: 500)')

    args = parser.parse_args()

    random.seed(42)  # For reproducibility

    data_dir = Path(__file__).parent / "math-hard"

    print("=" * 80)
    print("📚 MATH-Hard Dataset Setup")
    print("   Source: https://huggingface.co/datasets/lighteval/MATH-Hard")
    print("=" * 80)

    if args.download:
        print("\n🔽 Downloading MATH-Hard dataset...")
        success = download_and_format_math_hard(data_dir)

        if not success:
            print("\n❌ Download failed. Please install dependencies:")
            print("   pip install datasets")
            exit(1)

    if args.sample:
        # Check if test.jsonl exists
        test_file = data_dir / "test.jsonl"
        if not test_file.exists():
            print(f"\n❌ {test_file} not found. Run with --download first.")
            exit(1)

        print(f"\n🎲 Creating sampled dataset ({args.sample_size} problems)...")
        sample_math_hard_dataset(
            input_file=test_file,
            output_file=data_dir / f"test_{args.sample_size}.jsonl",
            n_samples=args.sample_size
        )

    if not args.download and not args.sample:
        print("\n⚠️  No action specified. Use --download and/or --sample")
        print("\nExamples:")
        print("  # Download dataset")
        print("  python download_math_hard.py --download")
        print("\n  # Download and create 500-problem sample")
        print("  python download_math_hard.py --download --sample --sample-size 500")
        print("\n  # Just create sample (if already downloaded)")
        print("  python download_math_hard.py --sample --sample-size 500")
    else:
        print("\n" + "=" * 80)
        print("✅ MATH-Hard dataset setup complete!")
        print("=" * 80)

        if data_dir.exists():
            files = list(data_dir.glob("*.jsonl"))
            if files:
                print(f"\n📁 Files created in {data_dir}:")
                for f in sorted(files):
                    size = f.stat().st_size / 1024  # KB
                    num_lines = sum(1 for _ in open(f))
                    print(f"  • {f.name} ({num_lines:,} problems, {size:.1f} KB)")

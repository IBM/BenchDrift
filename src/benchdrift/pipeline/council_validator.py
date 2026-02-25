"""
Council-based Validation using VLLM (Sequential Multi-Judge Approach).

Designed to work with VLLM by running judges sequentially (one model at a time).

Flow:
1. Run validation with Judge 1 (VLLM) → store verdicts in separate verdicts file
2. Run validation with Judge 2 (VLLM) → store verdicts in separate verdicts file
3. Run validation with Judge 3 (VLLM) → store verdicts in separate verdicts file
4. Run council synthesis (VLLM) → read verdicts, make final decisions, update unified file

Judge verdicts are stored in a separate file (not in unified file).
Final output matches what current validation stage produces.
"""

import json
import logging
import os
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


def get_verdicts_file_path(unified_file: str) -> str:
    """Get the path for the verdicts file based on unified file path."""
    base, ext = os.path.splitext(unified_file)
    return f"{base}_council_verdicts.json"


def load_verdicts(verdicts_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load verdicts from file.

    Returns:
        Dict mapping variation_key -> {judge_id: verdict_info}
        variation_key is "{problem_id}_{var_idx}"
    """
    if os.path.exists(verdicts_file):
        with open(verdicts_file, 'r') as f:
            return json.load(f)
    return {}


def save_verdicts(verdicts_file: str, verdicts: Dict[str, Dict[str, Any]]) -> None:
    """Save verdicts to file."""
    with open(verdicts_file, 'w') as f:
        json.dump(verdicts, f, indent=2)


def get_judge_validation_prompt() -> str:
    """Get the system prompt for judge validation."""
    return """You are an expert judge validating whether a problem variation preserves the original answer.

Your task: Determine if the variation has the EXACT SAME numerical/logical answer as the original problem.

CRITICAL CHECKS:
- Numerical values must be equivalent (30 min = 0.5 hours = 1800 seconds)
- Mathematical operations must be preserved
- Key quantities cannot change (5 apples != 10 apples)
- Time points must be equivalent (8:00 AM = 0800 hours)

EXAMPLES:
VALID: "30 minutes" -> "half an hour" (equivalent)
VALID: "8:00 AM" -> "0800 hours" (equivalent)
INVALID: "30 minutes" -> "45 minutes" (different value)
INVALID: "5 apples" -> "10 apples" (different quantity)

Respond with ONLY one word: VALID or INVALID"""


def build_judge_user_prompt(original: str, variation: str, ground_truth: str = "") -> str:
    """Build user prompt for judge validation."""
    gt_section = f"\nGround Truth Answer: {ground_truth}" if ground_truth else ""

    return f"""Original Problem:
{original}
{gt_section}

Variation:
{variation}

Is this variation VALID (same answer) or INVALID (different answer)?"""


def parse_judge_response(response: str) -> str:
    """Parse judge response to extract verdict. Conservative: defaults to INVALID."""
    response_upper = response.upper().strip()
    if not response_upper:
        return "INVALID"
    if "VALID" in response_upper and "INVALID" not in response_upper:
        return "VALID"
    if "INVALID" in response_upper:
        return "INVALID"
    return "INVALID"


def run_judge_validation(
    unified_file: str,
    judge_id: str,
    model_client,
    batch_size: int = 5000
) -> Dict[str, Any]:
    """
    Run a single judge's validation pass on all variations.
    Verdicts are stored in a separate verdicts file.

    Args:
        unified_file: Path to unified JSON file
        judge_id: Identifier for this judge (e.g., "judge_1")
        model_client: VLLM model client (already initialized)
        batch_size: Batch size for VLLM calls

    Returns:
        Stats dict with counts
    """
    logger.info(f"🧑‍⚖️ Running {judge_id} validation...")

    # Load unified data
    with open(unified_file, 'r') as f:
        data = json.load(f)

    # Load existing verdicts
    verdicts_file = get_verdicts_file_path(unified_file)
    all_verdicts = load_verdicts(verdicts_file)

    model_name = getattr(model_client, 'model_name', str(type(model_client).__name__))
    logger.info(f"   Model: {model_name}")

    # Build baseline lookup
    baselines = {}
    for entry in data:
        if entry.get('is_baseline'):
            problem_id = entry.get('problem_id', '')
            baselines[problem_id] = {
                'original': entry.get('original_problem', entry.get('problem', '')),
                'ground_truth': entry.get('ground_truth_answer', entry.get('answer', ''))
            }

    # Collect variations to validate
    variations_to_validate = []  # (variation_key, original, variation, ground_truth)

    var_idx_by_problem = {}  # Track variation index per problem
    for entry in data:
        if entry.get('is_variant'):
            problem_id = entry.get('problem_id', '')

            # Get variation index for this problem
            if problem_id not in var_idx_by_problem:
                var_idx_by_problem[problem_id] = 0
            var_idx = var_idx_by_problem[problem_id]
            var_idx_by_problem[problem_id] += 1

            variation_key = f"{problem_id}_{var_idx}"

            baseline = baselines.get(problem_id, {})
            original = baseline.get('original', '')
            ground_truth = baseline.get('ground_truth', '')
            variation = entry.get('modified_problem', '')

            if original and variation:
                variations_to_validate.append((variation_key, original, variation, ground_truth))

    if not variations_to_validate:
        logger.info("   No variations to validate")
        return {'total': 0, 'valid': 0, 'invalid': 0}

    logger.info(f"   Validating {len(variations_to_validate)} variations in batches of {batch_size}...")

    # Build all prompts
    system_prompt = get_judge_validation_prompt()
    all_system_prompts = []
    all_user_prompts = []

    for _, original, variation, ground_truth in variations_to_validate:
        all_system_prompts.append(system_prompt)
        all_user_prompts.append(build_judge_user_prompt(original, variation, ground_truth))

    # Process in batches using VLLM client
    all_responses = []

    from tqdm import tqdm
    for batch_start in tqdm(range(0, len(variations_to_validate), batch_size),
                            desc=f'{judge_id} validating', unit='batch'):
        batch_end = min(batch_start + batch_size, len(variations_to_validate))
        batch_sys = all_system_prompts[batch_start:batch_end]
        batch_user = all_user_prompts[batch_start:batch_end]

        if hasattr(model_client, 'get_model_response'):
            responses = model_client.get_model_response(batch_sys, batch_user)
        else:
            responses = [model_client.get_single_response(s, u) for s, u in zip(batch_sys, batch_user)]

        all_responses.extend(responses)

    # Store verdicts
    valid_count = 0
    invalid_count = 0

    for (variation_key, _, _, _), response in zip(variations_to_validate, all_responses):
        verdict = parse_judge_response(str(response))

        # Initialize variation entry if needed
        if variation_key not in all_verdicts:
            all_verdicts[variation_key] = {}

        # Store this judge's verdict
        all_verdicts[variation_key][judge_id] = {
            'model': model_name,
            'verdict': verdict
        }

        if verdict == "VALID":
            valid_count += 1
        else:
            invalid_count += 1

    # Save verdicts to separate file
    save_verdicts(verdicts_file, all_verdicts)

    logger.info(f"   ✅ {judge_id} complete: {valid_count} VALID, {invalid_count} INVALID")
    logger.info(f"   Verdicts saved to: {verdicts_file}")

    return {
        'total': len(variations_to_validate),
        'valid': valid_count,
        'invalid': invalid_count
    }


def get_council_synthesis_prompt() -> str:
    """Get the system prompt for council synthesis."""
    return """You are the Chairman of a council of LLM judges. Your role is to synthesize individual judge verdicts into a final decision.

You will receive verdicts from multiple judges for a problem variation. Based on their input:
1. Make a final VALID or INVALID decision
2. Consider majority voting

Respond with ONLY one word: VALID or INVALID"""


def build_council_user_prompt(original: str, variation: str, ground_truth: str, verdicts: Dict) -> str:
    """Build user prompt for council synthesis."""
    verdicts_text = ""
    for judge_id, verdict_info in verdicts.items():
        verdicts_text += f"- {judge_id} ({verdict_info.get('model', 'unknown')}): {verdict_info.get('verdict', 'UNKNOWN')}\n"

    gt_section = f"\nGround Truth Answer: {ground_truth}" if ground_truth else ""

    return f"""Original Problem:
{original}
{gt_section}

Variation:
{variation}

Judge Verdicts:
{verdicts_text}

Based on the council's input, what is the final verdict? VALID or INVALID?"""


def get_rectification_prompt() -> str:
    """Get the system prompt for rectification."""
    return """You are an expert at fixing problem variations that have been marked as invalid.

Your task: Modify the variation to preserve the EXACT SAME answer as the original problem.
- Keep the variation's style and format
- Only change what's necessary to make the answer match
- Preserve numerical equivalence

Output ONLY the corrected variation text, nothing else."""


def build_rectification_user_prompt(original: str, variation: str, ground_truth: str) -> str:
    """Build user prompt for rectification."""
    gt_section = f"\nGround Truth Answer: {ground_truth}" if ground_truth else ""

    return f"""Original Problem:
{original}
{gt_section}

Invalid Variation (answer doesn't match):
{variation}

Provide a corrected version of this variation that preserves the original answer:"""


def run_council_synthesis(
    unified_file: str,
    model_client,
    batch_size: int = 5000,
    rectify_invalid: bool = False,
    min_judges_required: int = 2
) -> Dict[str, Any]:
    """
    Run council synthesis to aggregate judge verdicts and make final decisions.
    Updates the unified file (like current validation stage does).

    Args:
        unified_file: Path to unified JSON file
        model_client: VLLM model client for council (chairman)
        batch_size: Batch size for VLLM calls
        rectify_invalid: If True, rectify invalid variations; if False, drop them
        min_judges_required: Minimum judges needed to synthesize

    Returns:
        Stats dict with counts
    """
    logger.info(f"🏛️ Running council synthesis...")

    # Load unified data
    with open(unified_file, 'r') as f:
        data = json.load(f)

    # Load verdicts
    verdicts_file = get_verdicts_file_path(unified_file)
    all_verdicts = load_verdicts(verdicts_file)

    if not all_verdicts:
        logger.info("   No verdicts found. Run judges first.")
        return {'total': 0, 'valid': 0, 'invalid': 0, 'rectified': 0, 'dropped': 0}

    model_name = getattr(model_client, 'model_name', str(type(model_client).__name__))
    logger.info(f"   Chairman model: {model_name}")

    # Check how many judges have run
    judge_ids = set()
    for verdicts in all_verdicts.values():
        judge_ids.update(verdicts.keys())
    logger.info(f"   Judges found: {sorted(judge_ids)}")

    if len(judge_ids) < min_judges_required:
        logger.info(f"   Need at least {min_judges_required} judges, only {len(judge_ids)} have run")
        return {'total': 0, 'valid': 0, 'invalid': 0, 'rectified': 0, 'dropped': 0}

    # Build baseline lookup
    baselines = {}
    for entry in data:
        if entry.get('is_baseline'):
            problem_id = entry.get('problem_id', '')
            baselines[problem_id] = {
                'original': entry.get('original_problem', entry.get('problem', '')),
                'ground_truth': entry.get('ground_truth_answer', entry.get('answer', ''))
            }

    # Collect variations with their verdicts
    # Need to match data entries to verdict keys
    variations_for_council = []  # (data_idx, variation_key, original, variation, ground_truth, verdicts)

    var_idx_by_problem = {}
    for data_idx, entry in enumerate(data):
        if entry.get('is_variant'):
            problem_id = entry.get('problem_id', '')

            if problem_id not in var_idx_by_problem:
                var_idx_by_problem[problem_id] = 0
            var_idx = var_idx_by_problem[problem_id]
            var_idx_by_problem[problem_id] += 1

            variation_key = f"{problem_id}_{var_idx}"

            if variation_key not in all_verdicts:
                continue

            verdicts = all_verdicts[variation_key]
            if len(verdicts) < min_judges_required:
                continue

            baseline = baselines.get(problem_id, {})
            original = baseline.get('original', '')
            ground_truth = baseline.get('ground_truth', '')
            variation = entry.get('modified_problem', '')

            if original and variation:
                variations_for_council.append(
                    (data_idx, variation_key, original, variation, ground_truth, verdicts)
                )

    if not variations_for_council:
        logger.info("   No variations with sufficient judge verdicts")
        return {'total': 0, 'valid': 0, 'invalid': 0, 'rectified': 0, 'dropped': 0}

    logger.info(f"   Synthesizing {len(variations_for_council)} variations...")

    # Build synthesis prompts
    system_prompt = get_council_synthesis_prompt()
    all_system_prompts = []
    all_user_prompts = []

    for _, _, original, variation, ground_truth, verdicts in variations_for_council:
        all_system_prompts.append(system_prompt)
        all_user_prompts.append(build_council_user_prompt(original, variation, ground_truth, verdicts))

    # Process synthesis in batches
    all_responses = []

    from tqdm import tqdm
    for batch_start in tqdm(range(0, len(variations_for_council), batch_size),
                            desc='Council synthesizing', unit='batch'):
        batch_end = min(batch_start + batch_size, len(variations_for_council))
        batch_sys = all_system_prompts[batch_start:batch_end]
        batch_user = all_user_prompts[batch_start:batch_end]

        if hasattr(model_client, 'get_model_response'):
            responses = model_client.get_model_response(batch_sys, batch_user)
        else:
            responses = [model_client.get_single_response(s, u) for s, u in zip(batch_sys, batch_user)]

        all_responses.extend(responses)

    # Parse synthesis results and collect invalid indices
    valid_count = 0
    invalid_list = []  # (list_idx, data_idx, original, variation, ground_truth)

    for list_idx, ((data_idx, _, original, variation, ground_truth, _), response) in enumerate(
            zip(variations_for_council, all_responses)):
        verdict = parse_judge_response(str(response))

        if verdict == "VALID":
            valid_count += 1
        else:
            invalid_list.append((list_idx, data_idx, original, variation, ground_truth))

    invalid_count = len(invalid_list)
    logger.info(f"   Council decisions: {valid_count} VALID, {invalid_count} INVALID")

    # Handle invalid variations
    rectified_count = 0
    dropped_count = 0
    entries_to_remove = set()

    if invalid_list:
        if rectify_invalid:
            # Rectify invalid variations
            logger.info(f"   🔧 Rectifying {invalid_count} invalid variations...")

            rect_system = get_rectification_prompt()
            rect_system_prompts = []
            rect_user_prompts = []

            for _, _, original, variation, ground_truth in invalid_list:
                rect_system_prompts.append(rect_system)
                rect_user_prompts.append(build_rectification_user_prompt(original, variation, ground_truth))

            # Process rectification in batches
            rect_responses = []
            for batch_start in tqdm(range(0, len(invalid_list), batch_size),
                                    desc='Rectifying', unit='batch'):
                batch_end = min(batch_start + batch_size, len(invalid_list))
                batch_sys = rect_system_prompts[batch_start:batch_end]
                batch_user = rect_user_prompts[batch_start:batch_end]

                if hasattr(model_client, 'get_model_response'):
                    responses = model_client.get_model_response(batch_sys, batch_user)
                else:
                    responses = [model_client.get_single_response(s, u) for s, u in zip(batch_sys, batch_user)]

                rect_responses.extend(responses)

            # Apply rectifications
            for (_, data_idx, _, _, _), rectified in zip(invalid_list, rect_responses):
                rectified_text = str(rectified).strip()
                if rectified_text:
                    data[data_idx]['modified_problem'] = rectified_text
                    data[data_idx]['validation_corrected'] = True
                    rectified_count += 1

            logger.info(f"   ✅ Rectified {rectified_count} variations")
        else:
            # Mark entries for removal
            for _, data_idx, _, _, _ in invalid_list:
                entries_to_remove.add(data_idx)
            dropped_count = len(entries_to_remove)

            # Filter data
            data = [entry for idx, entry in enumerate(data) if idx not in entries_to_remove]

            logger.info(f"   ✅ Dropped {dropped_count} invalid variations")

    # Mark validation complete
    for entry in data:
        entry['validation_complete'] = True

    # Save updated unified file
    with open(unified_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"🏛️ Council synthesis complete!")

    return {
        'total': len(variations_for_council),
        'valid': valid_count,
        'invalid': invalid_count,
        'rectified': rectified_count,
        'dropped': dropped_count
    }


def get_judge_status(unified_file: str) -> Dict[str, Any]:
    """Get status of judges that have run."""
    verdicts_file = get_verdicts_file_path(unified_file)
    all_verdicts = load_verdicts(verdicts_file)

    if not all_verdicts:
        return {'judges': [], 'variations_count': 0}

    judge_ids = set()
    for verdicts in all_verdicts.values():
        judge_ids.update(verdicts.keys())

    return {
        'judges': sorted(list(judge_ids)),
        'variations_count': len(all_verdicts),
        'verdicts_file': verdicts_file
    }


def clear_verdicts(unified_file: str) -> None:
    """Clear the verdicts file for a fresh start."""
    verdicts_file = get_verdicts_file_path(unified_file)
    if os.path.exists(verdicts_file):
        os.remove(verdicts_file)
        logger.info(f"Cleared verdicts file: {verdicts_file}")


def council_validate_variations(
    problems_with_variations: List,
    ground_truths: List[str],
    api_key: str = None,
    council_models: List[str] = None,
    chairman_model: str = None,
    batch_size: int = 10,
    client_type: str = 'vllm',
    max_model_len: int = 8192,
    rectify_invalid: bool = False,
    min_judges: int = 2,
    unified_file: str = None,
) -> Dict[int, List[int]]:
    """
    In-process council validation for use from the unified pipeline.

    Runs multiple judges sequentially, then synthesizes verdicts.
    Returns invalid variation indices per problem, matching the interface
    expected by unified_batched_pipeline_semantic.py.

    NOTE: For large-scale runs, prefer using council_validator_batch.py via
    the shell script (run_all_experiments_optimized.sh) which manages GPU
    memory more efficiently by loading one model at a time.

    Args:
        problems_with_variations: List of (original_problem, [variation_texts])
        ground_truths: Ground truth answers per problem
        api_key: Not used (legacy OpenRouter param, kept for compatibility)
        council_models: List of judge model names for VLLM
        chairman_model: Chairman model name for synthesis
        batch_size: Batch size for validation
        client_type: Model client type ('vllm' or 'rits')
        max_model_len: Max model length for VLLM
        rectify_invalid: Whether to rectify or drop invalid variations
        min_judges: Minimum judges required for council synthesis
        unified_file: Path to unified JSON file (needed for verdict storage)

    Returns:
        Dict mapping problem_index -> [invalid_variation_indices]
    """
    import tempfile

    if not council_models or not chairman_model:
        logger.warning("Council models or chairman not specified, falling back to empty result")
        return {}

    if not unified_file:
        logger.warning("No unified_file specified for council validation — cannot store verdicts")
        return {}

    # Write temporary unified file with the problems/variations in the expected format
    # so run_judge_validation can process them
    temp_data = []
    for prob_idx, ((original, variations), gt) in enumerate(zip(problems_with_variations, ground_truths)):
        # Baseline entry
        temp_data.append({
            'problem_id': f'problem_{prob_idx}',
            'is_baseline': True,
            'original_problem': original,
            'problem': original,
            'ground_truth_answer': gt,
            'answer': gt,
        })
        # Variation entries
        for var_text in variations:
            temp_data.append({
                'problem_id': f'problem_{prob_idx}',
                'is_variant': True,
                'modified_problem': var_text,
            })

    # Use the unified_file path for verdict storage
    with open(unified_file, 'w') as f:
        json.dump(temp_data, f, indent=2)

    # Import model client factory
    from benchdrift.models.model_client import ModelClientFactory

    # Clear previous verdicts
    clear_verdicts(unified_file)

    # Run each judge sequentially
    for judge_idx, model_name in enumerate(council_models):
        judge_id = f'judge_{judge_idx + 1}'
        logger.info(f"Running {judge_id} with model {model_name}...")

        model_client = ModelClientFactory.create_client(
            client_type, model_name, max_model_len=max_model_len
        )
        run_judge_validation(
            unified_file=unified_file,
            judge_id=judge_id,
            model_client=model_client,
            batch_size=batch_size,
        )
        del model_client  # Release GPU memory

    # Run council synthesis with chairman
    logger.info(f"Running council synthesis with chairman {chairman_model}...")
    chairman_client = ModelClientFactory.create_client(
        client_type, chairman_model, max_model_len=max_model_len
    )
    run_council_synthesis(
        unified_file=unified_file,
        model_client=chairman_client,
        batch_size=batch_size,
        rectify_invalid=rectify_invalid,
        min_judges_required=min_judges,
    )
    del chairman_client

    # Read back the updated file to determine which variations were dropped/invalid
    with open(unified_file, 'r') as f:
        updated_data = json.load(f)

    # Build the invalid_by_problem map expected by the caller
    # Compare original variations with what survived
    surviving_variants = {}
    for entry in updated_data:
        if entry.get('is_variant'):
            pid = entry.get('problem_id', '')
            if pid not in surviving_variants:
                surviving_variants[pid] = set()
            surviving_variants[pid].add(entry.get('modified_problem', ''))

    invalid_by_problem = {}
    for prob_idx, (original, variations) in enumerate(problems_with_variations):
        pid = f'problem_{prob_idx}'
        surviving = surviving_variants.get(pid, set())
        invalid_indices = []
        for var_idx, var_text in enumerate(variations):
            if var_text not in surviving:
                invalid_indices.append(var_idx)
        if invalid_indices:
            invalid_by_problem[prob_idx] = invalid_indices

    return invalid_by_problem


# CLI interface
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Council-based validation for BenchDrift')
    parser.add_argument('--unified-file', required=True, help='Path to unified JSON file')
    parser.add_argument('--mode', required=True, choices=['judge', 'council', 'status', 'clear'],
                       help='Mode: judge (run single judge), council (synthesize), status (show judges), clear (reset)')
    parser.add_argument('--judge-id', help='Judge identifier (e.g., judge_1)')
    parser.add_argument('--model-name', help='Model name for VLLM')
    parser.add_argument('--client-type', default='vllm', choices=['vllm', 'rits', 'ollama', 'ollama_logits'],
                       help='Model client type')
    parser.add_argument('--batch-size', type=int, default=5000, help='Batch size')
    parser.add_argument('--max-model-len', type=int, default=8192, help='Max model length for VLLM')
    parser.add_argument('--rectify-invalid', action='store_true', help='Rectify invalid variations')
    parser.add_argument('--min-judges', type=int, default=2, help='Minimum judges for council')

    args = parser.parse_args()

    if args.mode == 'status':
        status = get_judge_status(args.unified_file)
        print(f"Judges: {status['judges'] if status['judges'] else 'None'}")
        print(f"Variations with verdicts: {status['variations_count']}")
        if status.get('verdicts_file'):
            print(f"Verdicts file: {status['verdicts_file']}")

    elif args.mode == 'clear':
        clear_verdicts(args.unified_file)
        print("Verdicts cleared")

    elif args.mode == 'judge':
        if not args.judge_id or not args.model_name:
            print("Error: --judge-id and --model-name required for judge mode")
            exit(1)

        from benchdrift.pipeline.comprehensive_variation_engine_v2 import create_model_client_for_variations
        model_client = create_model_client_for_variations(args.client_type, args.model_name, args.max_model_len)

        stats = run_judge_validation(
            unified_file=args.unified_file,
            judge_id=args.judge_id,
            model_client=model_client,
            batch_size=args.batch_size
        )
        print(f"Stats: {stats}")

    elif args.mode == 'council':
        if not args.model_name:
            print("Error: --model-name required for council mode")
            exit(1)

        status = get_judge_status(args.unified_file)
        if len(status['judges']) < args.min_judges:
            print(f"Error: Need {args.min_judges} judges, only {len(status['judges'])} have run")
            exit(1)

        from benchdrift.pipeline.comprehensive_variation_engine_v2 import create_model_client_for_variations
        model_client = create_model_client_for_variations(args.client_type, args.model_name, args.max_model_len)

        stats = run_council_synthesis(
            unified_file=args.unified_file,
            model_client=model_client,
            batch_size=args.batch_size,
            rectify_invalid=args.rectify_invalid,
            min_judges_required=args.min_judges
        )
        print(f"Stats: {stats}")

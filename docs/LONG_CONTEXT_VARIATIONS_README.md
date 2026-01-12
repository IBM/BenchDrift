# Long-Context Variation Engine

## Overview

The Long-Context Variation Engine adds specialized variations for prompts with extensive grounding context (e.g., tool documentation, rules, examples, available connections) followed by user queries.

This addresses the unique brittleness patterns in long-context prompts that differ from short math/temporal/NL problems.

## Research Backing

The variations are grounded in research from:
- **Reading comprehension & discourse processing** (Kintsch & van Dijk, 1978; Meyer, 1975)
- **Typographic processing** (surface-level format effects)
- **Coreference resolution** (Hobbs, 1979; Grosz et al., 1995)
- **Information theory** (Shannon - redundancy effects)
- **Speech acts & pragmatics** (Searle, 1975; Austin, 1962)
- **Memory effects** (primacy/recency effects in information processing)

## Variation Types

### 1. Document Structure Variations
- **Section reordering**: Move RULES before/after EXAMPLES
- **Heading reformulation**: "RULES:" → "Selection Guidelines:" → "Tool Selection Criteria:"
- **List structure changes**: Numbered lists (1. 2. 3.) → bullet lists (- - -)
- **Information chunking**: Combine/split sections

### 2. Reference/Formatting Variations (Deterministic)
- **Quote style**: `"text"` → `'text'` → `` `text` ``
- **Case variations**: `UPPERCASE_HEADINGS` → `Title_Case_Headings`
- **Whitespace**: Dense (single newline) → spacious (double newline)
- **Delimiters**: `<<< >>>` → `""" """` → `---`

### 3. Reference Expression Variations (LLM-based)
- **Full reference → pronoun**: "the tool" → "it"
- **Definite/indefinite articles**: "Select a tool" → "Select the tool"
- **Demonstratives**: "this query" → "the query" → "that user question"

### 4. Information Redundancy Variations
- **Minimal context**: Remove EXAMPLES section
- **Expanded context**: Add more examples/clarifications
- **Ellipsis → explicit**: "Run on DD1D" → "Run this tool on DD1D database instance"

### 5. Instruction Directness Variations (LLM-based)
- **Imperative**: "Select TOOL or NO_TOOL"
- **Interrogative**: "Should you select TOOL or NO_TOOL?"
- **Declarative**: "Your task is to determine whether to select..."

### 6. Context Positioning Variations
- **Rules-first vs examples-first**: Swap RULES and EXAMPLES sections
- **Query at beginning vs end**: Move USER_QUERY to start/end
- **Key info placement**: Critical constraints at start vs end

## Usage

### Command Line

Add the `--enable-long-context` flag to enable long-context variations:

```bash
python unified_batched_pipeline.py \
  --input long_context_prompts.jsonl \
  --unified-file results.json \
  --all-stages \
  --enable-long-context \
  --batch-size 50
```

### Python API

```python
from comprehensive_variation_engine_v2 import ComprehensiveVariationEngine
from long_context_variation_engine import LongContextVariationEngine

# Initialize with model client
engine = ComprehensiveVariationEngine(model_client=your_model_client)

# Generate variations with long-context mode enabled
variations = engine.generate_comprehensive_variations(
    problem=long_context_prompt,
    enable_long_context=True,
    max_variations=20
)

# Access long-context specific variations
for var in variations:
    if 'long_context' in var.get('generation_method', ''):
        print(f"Type: {var['transformation_type']}")
        print(f"Modified: {var['modified_problem'][:100]}...")
```

### Standalone Long-Context Engine

You can also use the long-context engine directly:

```python
from long_context_variation_engine import LongContextVariationEngine

# Initialize
lc_engine = LongContextVariationEngine(model_client=your_model_client)

# Detect structure (uses LLM or heuristics)
structure = lc_engine.detect_long_context_structure(prompt)
print(f"Found {len(structure['sections'])} sections")

# Generate variations
variations = lc_engine.generate_long_context_variations(
    problem=prompt,
    max_variations=20
)
```

## When to Use Long-Context Mode

Enable `--enable-long-context` when your prompts have:

1. **Extensive grounding context** (>500 tokens)
2. **Multiple sections** (rules, documentation, examples, connections)
3. **Clear context-query separation** (USER_QUERY at beginning or end)
4. **Tool selection/verification tasks** with detailed specifications
5. **Structured prompts** with headings, lists, and delimiters

Examples:
- Tool selection prompts with documentation
- Agentic prompts with rules and examples
- RAG prompts with retrieved context
- System prompts with extensive guidelines

## Don't Use Long-Context Mode For

- Short math problems ("What is 2+3?")
- Single-sentence questions
- Prompts without clear structure
- Conversational queries without grounding context

## Structure Detection

The engine automatically detects prompt structure using:

**With LLM** (recommended):
- Identifies section types (rules, docs, examples, query)
- Detects heading formats and list structures
- Determines query location (beginning/middle/end)
- Extracts delimiters and formatting patterns

**Without LLM** (fallback):
- Regex-based section detection
- Pattern matching for common headings
- Heuristic query location detection

## Performance Notes

**Deterministic variations** (Types 1, 2, 4):
- Fast, no LLM calls
- Predictable output
- Quote changes, case changes, whitespace, section removal

**LLM-based variations** (Types 3, 5):
- Requires model client
- Reference expression rewriting
- Instruction directness changes
- Uses same model as variation stage

**Hybrid variations** (Type 6):
- Mix of deterministic reordering and LLM reformulation
- Context positioning variations

## Output Format

Each variation is a dict with:

```python
{
    'modified_problem': str,        # The transformed prompt
    'transformation_type': str,     # e.g., 'heading_reformulation', 'quotes_double_to_single'
    'original_component': str,      # What was changed
    'new_component': str,          # What it was changed to
    'original_problem': str,        # Original prompt (added by pipeline)
    'detection_method': str,        # 'long_context_structure_analysis'
    'generation_method': str,       # 'long_context_variations'
    'confidence': str               # 'deterministic' or 'model_generated'
}
```

## Integration with Existing Pipeline

The long-context engine integrates seamlessly:

1. **Same model client** as variation stage
2. **Same output format** as other variation engines
3. **No pipeline changes** except adding `enable_long_context` flag
4. **Runs after generic variations**, before combination generation

Order of execution:
```
1. Math variations
2. Temporal variations
3. NL variations
4. Model-guided variations
5. Combination variations
6. Generic variations
7. Long-context variations  ← NEW (when enabled)
8. Ensure minimum combinations
9. Add metadata
```

## Example Variations

**Original prompt:**
```
RULES:
1. Check instance availability
2. Verify parameters
3. Match tool capability

USER_QUERY: Which indexes exist in DWY1?
```

**Variation 1 (Heading reformulation):**
```
Selection Guidelines:
1. Check instance availability
2. Verify parameters
3. Match tool capability

USER_QUERY: Which indexes exist in DWY1?
```

**Variation 2 (List structure):**
```
RULES:
- Check instance availability
- Verify parameters
- Match tool capability

USER_QUERY: Which indexes exist in DWY1?
```

**Variation 3 (Quote style):**
```
RULES:
1. Check instance availability
2. Verify parameters
3. Match tool capability

USER_QUERY: Which indexes exist in DWY1?
```
(Changes all `"` to `'` throughout)

**Variation 4 (Query positioning):**
```
USER_QUERY: Which indexes exist in DWY1?

RULES:
1. Check instance availability
2. Verify parameters
3. Match tool capability
```

**Variation 5 (Instruction directness):**
```
RULES:
1. Should you check instance availability?
2. Have you verified parameters?
3. Does the tool capability match?

USER_QUERY: Which indexes exist in DWY1?
```

## Troubleshooting

**"Long-context variation engine failed to initialize"**
- Check that `long_context_variation_engine.py` is in the same directory
- Ensure model client is properly configured
- Engine will fall back to other variations if initialization fails

**"LLM structure detection failed, falling back to heuristics"**
- LLM call timeout or error
- Will use regex-based fallback
- Variations will still be generated, just fewer LLM-based ones

**"Long-context variations requested but engine not available"**
- Engine failed to initialize but flag was set
- Check import errors and dependencies
- Pipeline will continue without long-context variations

## Files

- `long_context_variation_engine.py` - Main engine implementation
- `comprehensive_variation_engine_v2.py` - Integration point
- `complete_variation_pipeline.py` - Pipeline integration
- `unified_batched_pipeline.py` - CLI flag handling

## Future Extensions

Potential additions:
1. **Fact reordering** within sections (preserving dependencies)
2. **Paraphrase variations** of key rules/constraints
3. **Redundancy injection** (add obvious clarifications)
4. **Format standardization** (markdown → plain text)
5. **Compression variations** (aggressive summarization)
6. **Multi-language** heading styles

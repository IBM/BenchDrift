"""
Relevance Selector Module for BenchDrift v2

Two-stage relevance selection:
1. Embedding retrieval: Fast pre-filtering of candidate transformations
2. LLM selection: Accurate relevance assessment using judge model

Uses the same model client as the rest of the pipeline (judge model).
No hardcoded rules - all selection is semantic/LLM-based.
"""

import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Import prompts from centralized registry
from benchdrift.pipeline.prompts import (
    RELEVANCE_SELECTION_SYSTEM,
    RELEVANCE_SELECTION_USER,
    GAP_DETECTION_SYSTEM,
    GAP_DETECTION_USER,
    get_prompt
)

logger = logging.getLogger('BenchDrift.RelevanceSelector')

# Optional embedding support
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Using LLM-only selection.")


@dataclass
class RankedTransformation:
    """A transformation ranked by relevance"""
    name: str
    axis: str
    category: str
    score: float
    reasoning: str = ""


@dataclass
class GapReport:
    """Report of taxonomy coverage gaps"""
    has_gaps: bool
    coverage_score: float
    covered_axes: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RelevanceResult:
    """Complete result of relevance analysis"""
    problem: str
    ranked_transformations: List[RankedTransformation]
    gap_report: Optional[GapReport] = None


class TaxonomyIndex:
    """Loads and indexes the transformation taxonomy"""

    def __init__(self, taxonomy_path: str = None):
        if taxonomy_path is None:
            taxonomy_path = Path(__file__).parent.parent.parent.parent / "config" / "taxonomy" / "transformations.yaml"

        self.taxonomy_path = Path(taxonomy_path)
        self.transformations = {}
        self.axes = {}
        self.embeddings = {}
        self.embedding_model = None

        self._load_taxonomy()

    def _load_taxonomy(self):
        """Load taxonomy from YAML file"""
        if not self.taxonomy_path.exists():
            logger.warning(f"Taxonomy file not found: {self.taxonomy_path}")
            return

        with open(self.taxonomy_path, 'r') as f:
            data = yaml.safe_load(f)

        self.axes = data.get('axes', {})

        # Flatten transformations for indexing
        for axis_name, axis_data in self.axes.items():
            for trans_name, trans_data in axis_data.get('transformations', {}).items():
                self.transformations[trans_name] = {
                    'name': trans_name,
                    'axis': axis_name,
                    **trans_data
                }

        logger.info(f"Loaded {len(self.transformations)} transformations from taxonomy")

    def compute_embeddings(self, model_name: str = 'all-mpnet-base-v2'):
        """Pre-compute embeddings for all transformations"""
        if not EMBEDDINGS_AVAILABLE:
            return

        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

        for trans_name, trans_data in self.transformations.items():
            # Create rich text for embedding
            text_parts = [trans_data.get('description', '')]
            for example in trans_data.get('examples', []):
                if isinstance(example, dict):
                    text_parts.append(f"{example.get('before', '')} -> {example.get('after', '')}")

            embed_text = " ".join(text_parts)
            self.embeddings[trans_name] = self.embedding_model.encode(embed_text)

        logger.info(f"Computed embeddings for {len(self.embeddings)} transformations")

    def get_embedding(self, text: str):
        """Get embedding for text"""
        if self.embedding_model is None:
            return None
        return self.embedding_model.encode(text)


class RelevanceSelector:
    """
    Two-stage relevance selection:
    1. Embedding retrieval (fast candidate pool)
    2. LLM selection (accurate relevance with reasoning)

    Uses prompts from centralized prompts.py registry.
    """

    def __init__(self,
                 model_client=None,
                 taxonomy_path: str = None,
                 use_embeddings: bool = True,
                 embedding_model: str = 'all-mpnet-base-v2'):
        """
        Initialize relevance selector.

        Args:
            model_client: The model client for LLM selection (same as judge model)
            taxonomy_path: Path to transformations.yaml
            use_embeddings: Whether to use embeddings for pre-filtering
            embedding_model: Sentence transformer model name
        """
        self.model_client = model_client
        self.taxonomy = TaxonomyIndex(taxonomy_path)
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE

        if self.use_embeddings:
            self.taxonomy.compute_embeddings(embedding_model)

    def select(self,
              problem: str,
              top_k: int = 10,
              candidate_pool_size: int = 25,
              detect_gaps: bool = True) -> RelevanceResult:
        """
        Select relevant transformations for a problem.

        Args:
            problem: Input problem text
            top_k: Maximum number of transformations to return
            candidate_pool_size: Size of embedding-based candidate pool
            detect_gaps: Whether to run gap detection

        Returns:
            RelevanceResult with ranked transformations
        """
        # Stage 1: Embedding retrieval (fast pre-filter)
        if self.use_embeddings:
            candidates = self._embedding_retrieval(problem, candidate_pool_size)
        else:
            # No embeddings - use all transformations as candidates
            candidates = list(self.taxonomy.transformations.keys())

        # Stage 2: LLM selection (accurate relevance)
        if self.model_client:
            ranked = self._llm_selection(problem, candidates)
        else:
            # Fallback: return candidates with default scores
            ranked = [
                RankedTransformation(
                    name=name,
                    axis=self.taxonomy.transformations[name].get('axis', 'unknown'),
                    category=self.taxonomy.transformations[name].get('category', 'unknown'),
                    score=0.5,
                    reasoning="LLM selection not available"
                )
                for name in candidates[:top_k]
            ]

        # Filter to top_k
        ranked = ranked[:top_k]

        # Gap detection (optional)
        gap_report = None
        if detect_gaps and self.model_client and ranked:
            gap_report = self._detect_gaps(problem, ranked)

        return RelevanceResult(
            problem=problem,
            ranked_transformations=ranked,
            gap_report=gap_report
        )

    def _embedding_retrieval(self, problem: str, top_k: int) -> List[str]:
        """
        Stage 1: Fast embedding-based candidate retrieval.

        Returns transformation names sorted by similarity.
        """
        problem_embedding = self.taxonomy.get_embedding(problem)
        if problem_embedding is None:
            return list(self.taxonomy.transformations.keys())

        # Compute similarities
        similarities = []
        for trans_name, trans_embedding in self.taxonomy.embeddings.items():
            norm_product = np.linalg.norm(problem_embedding) * np.linalg.norm(trans_embedding)
            if norm_product == 0:
                sim = 0.0
            else:
                sim = np.dot(problem_embedding, trans_embedding) / norm_product
            similarities.append((trans_name, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in similarities[:top_k]]

    def _llm_selection(self, problem: str, candidates: List[str]) -> List[RankedTransformation]:
        """
        Stage 2: LLM-based relevance selection.

        Uses judge model to assess which candidates are actually relevant.
        """
        # Build candidate descriptions for LLM
        candidate_descriptions = []
        for i, name in enumerate(candidates, 1):
            trans = self.taxonomy.transformations.get(name, {})
            desc = trans.get('description', 'No description')
            examples = trans.get('examples', [])
            example_str = ""
            if examples and isinstance(examples[0], dict):
                ex = examples[0]
                example_str = f" (e.g., '{ex.get('before', '')}' -> '{ex.get('after', '')}')"

            candidate_descriptions.append(f"{i}. {name}: {desc}{example_str}")

        candidates_text = "\n".join(candidate_descriptions)

        user_prompt = f"""PROBLEM:
{problem}

CANDIDATE TRANSFORMATIONS:
{candidates_text}

For each transformation, determine if it is RELEVANT for testing this problem's robustness."""

        # Call LLM
        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response(
                    [RELEVANCE_SELECTION_SYSTEM],
                    [user_prompt]
                )
                response = responses[0] if responses else ""
            else:
                response = str(self.model_client.generate(
                    user_prompt,
                    RELEVANCE_SELECTION_SYSTEM
                ))
        except Exception as e:
            logger.error(f"LLM selection failed: {e}")
            # Fallback to all candidates with default scores
            return [
                RankedTransformation(
                    name=name,
                    axis=self.taxonomy.transformations.get(name, {}).get('axis', 'unknown'),
                    category=self.taxonomy.transformations.get(name, {}).get('category', 'unknown'),
                    score=0.5,
                    reasoning="LLM selection failed"
                )
                for name in candidates
            ]

        # Parse LLM response
        return self._parse_selection_response(response, candidates)

    def _parse_selection_response(self, response: str, candidates: List[str]) -> List[RankedTransformation]:
        """Parse LLM selection response into ranked transformations"""
        results = []

        # Parse response blocks
        blocks = response.split('---')

        parsed_names = set()
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Extract fields
            name = None
            verdict = None
            score = 0.5
            reasoning = ""

            for line in block.split('\n'):
                line = line.strip()
                if line.startswith('TRANSFORMATION:'):
                    name = line.replace('TRANSFORMATION:', '').strip()
                elif line.startswith('VERDICT:'):
                    verdict = line.replace('VERDICT:', '').strip().upper()
                elif line.startswith('SCORE:'):
                    try:
                        score = float(line.replace('SCORE:', '').strip())
                    except:
                        score = 0.5
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()

            # Match name to candidates (fuzzy matching)
            matched_name = None
            if name:
                for candidate in candidates:
                    if candidate.lower() == name.lower() or candidate in name or name in candidate:
                        matched_name = candidate
                        break

            if matched_name and matched_name not in parsed_names:
                parsed_names.add(matched_name)
                trans = self.taxonomy.transformations.get(matched_name, {})

                if verdict == 'RELEVANT':
                    results.append(RankedTransformation(
                        name=matched_name,
                        axis=trans.get('axis', 'unknown'),
                        category=trans.get('category', 'unknown'),
                        score=score,
                        reasoning=reasoning
                    ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _detect_gaps(self, problem: str, ranked: List[RankedTransformation]) -> GapReport:
        """
        Detect gaps in taxonomy coverage using LLM.

        Returns recommendations for new transformations (not hardcoded).
        """
        relevant_names = [t.name for t in ranked]
        relevant_text = "\n".join([
            f"- {t.name} ({t.axis}): {t.reasoning}"
            for t in ranked[:10]
        ])

        covered_axes = list(set(t.axis for t in ranked))

        user_prompt = f"""PROBLEM:
{problem}

RELEVANT TRANSFORMATIONS FOUND:
{relevant_text}

AXES COVERED: {', '.join(covered_axes)}

Analyze if there are any gaps in coverage for testing this problem's robustness.
If gaps exist, recommend new transformations that would address them."""

        try:
            if hasattr(self.model_client, 'get_model_response'):
                responses = self.model_client.get_model_response(
                    [GAP_DETECTION_SYSTEM],
                    [user_prompt]
                )
                response = responses[0] if responses else ""
            else:
                response = str(self.model_client.generate(
                    user_prompt,
                    GAP_DETECTION_SYSTEM
                ))
        except Exception as e:
            logger.error(f"Gap detection failed: {e}")
            return GapReport(
                has_gaps=False,
                coverage_score=len(ranked) / max(len(self.taxonomy.transformations), 1),
                covered_axes=covered_axes
            )

        # Parse gap response
        return self._parse_gap_response(response, covered_axes, len(ranked))

    def _parse_gap_response(self, response: str, covered_axes: List[str], num_relevant: int) -> GapReport:
        """Parse LLM gap detection response"""
        has_gaps = 'GAP_FOUND: YES' in response.upper()

        # Extract recommendations
        recommendations = []
        if 'RECOMMENDATION:' in response:
            rec_blocks = response.split('RECOMMENDATION:')[1:]
            for block in rec_blocks:
                rec = {}
                for line in block.split('\n'):
                    line = line.strip()
                    if line.startswith('NAME:'):
                        rec['name'] = line.replace('NAME:', '').strip()
                    elif line.startswith('AXIS:'):
                        rec['axis'] = line.replace('AXIS:', '').strip()
                    elif line.startswith('DESCRIPTION:'):
                        rec['description'] = line.replace('DESCRIPTION:', '').strip()
                    elif line.startswith('RATIONALE:'):
                        rec['rationale'] = line.replace('RATIONALE:', '').strip()

                if rec.get('name'):
                    recommendations.append(rec)

        coverage_score = min(1.0, num_relevant / 10.0)  # Normalize to 10 as good coverage

        return GapReport(
            has_gaps=has_gaps,
            coverage_score=coverage_score,
            covered_axes=covered_axes,
            recommendations=recommendations
        )

    def get_transformation_names(self, result: RelevanceResult) -> List[str]:
        """Convenience method to get just transformation names"""
        return [t.name for t in result.ranked_transformations]


# Convenience function for testing
def test_selector(problem: str, model_client=None):
    """Quick test of relevance selector"""
    selector = RelevanceSelector(
        model_client=model_client,
        use_embeddings=EMBEDDINGS_AVAILABLE
    )

    result = selector.select(problem, top_k=5, detect_gaps=True)

    print(f"\nProblem: {problem[:80]}...")
    print(f"\nRelevant transformations:")
    for t in result.ranked_transformations:
        print(f"  - {t.name} ({t.axis}): {t.score:.2f}")
        print(f"    Reasoning: {t.reasoning}")

    if result.gap_report:
        print(f"\nGap detection:")
        print(f"  Has gaps: {result.gap_report.has_gaps}")
        print(f"  Coverage: {result.gap_report.coverage_score:.2f}")
        if result.gap_report.recommendations:
            print(f"  Recommendations:")
            for rec in result.gap_report.recommendations:
                print(f"    - {rec.get('name')}: {rec.get('rationale', '')}")

    return result


if __name__ == "__main__":
    # Test without model client (embedding only)
    test_problems = [
        "What is 15 + 25?",
        "The meeting starts at 2:00 PM and lasts 90 minutes. When does it end?",
    ]

    for problem in test_problems:
        print("=" * 60)
        test_selector(problem)

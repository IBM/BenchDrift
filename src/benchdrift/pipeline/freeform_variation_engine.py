"""
BenchDrift — Free-form variation generation engine.

Generates problem variations using LLM with full creative freedom,
guided by domain-specific few-shot examples from an auto-expanding registry.

Flow:
1. Detect problem domain(s) via LLM
2. Match against synonym-aware registry
3. If unknown domain → generate + validate + store examples
4. Generate n variations using domain examples as few-shot
5. Validate variations using pipeline's existing validation prompts

The domain registry persists as JSON and grows over time.
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchdrift.pipeline.comprehensive_variation_engine_v2 import (
    clean_model_response,
    is_valid_question,
)
from benchdrift.pipeline.prompts import (
    FREEFORM_DOMAIN_DETECT_SYSTEM,
    FREEFORM_DOMAIN_DETECT_USER,
    FREEFORM_EXAMPLE_GEN_SYSTEM,
    FREEFORM_EXAMPLE_GEN_USER,
    FREEFORM_SINGLE_VARIATION_USER,
    FREEFORM_VARIATION_SYSTEM,
    FREEFORM_VARIATION_USER,
    VALIDATION_SYSTEM,
    VALIDATION_USER,
)

logger = logging.getLogger(__name__)

# Default registry path: alongside this module
_DEFAULT_REGISTRY_PATH = Path(__file__).parent / "freeform_domain_registry.json"

# Stop words to ignore during synonym matching
_STOP_WORDS = {"the", "a", "an", "of", "and", "or", "in", "for", "to", "with", "on", "is", "are"}


def _normalize(text: str) -> set:
    """Normalize a domain string to a set of significant words."""
    words = set(re.split(r'[\s_\-/]+', text.lower().strip()))
    return words - _STOP_WORDS


def load_registry(path: Optional[str] = None) -> dict:
    """Load domain registry from JSON file."""
    p = Path(path) if path else _DEFAULT_REGISTRY_PATH
    if p.exists():
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            return data.get("domains", {})
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load registry from {p}: {e}")
    return {}


def save_registry(domains: dict, path: Optional[str] = None):
    """Save domain registry to JSON file (atomic write)."""
    p = Path(path) if path else _DEFAULT_REGISTRY_PATH
    data = {
        "_meta": {"version": 1, "last_updated": "auto"},
        "domains": domains,
    }
    try:
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".json")
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(p))
    except Exception as e:
        logger.error(f"Failed to save registry: {e}")
        try:
            os.unlink(tmp)
        except Exception:
            pass


class FreeformVariationEngine:
    """Generates free-form problem variations with auto-expanding domain examples.

    Uses the pipeline's model client interface for all LLM calls.
    Compatible with both batch (get_model_response) and single (generate) clients.
    """

    def __init__(self, model_client=None, registry_path: Optional[str] = None):
        self.model_client = model_client
        self.registry_path = registry_path
        self.registry = load_registry(registry_path)

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  max_tokens: int = 1024, temperature: float = 0.5) -> str:
        """Single LLM call via the pipeline's model client."""
        if self.model_client is None:
            raise RuntimeError("No model client configured for FreeformVariationEngine")

        # Try batch interface first (most clients support this)
        if hasattr(self.model_client, 'get_model_response'):
            responses = self.model_client.get_model_response(
                [system_prompt], [user_prompt],
                max_new_tokens=max_tokens, temperature=temperature,
            )
            return responses[0] if responses else ""

        # Fallback to single-call interface
        if hasattr(self.model_client, 'generate'):
            return self.model_client.generate(user_prompt, system_prompt)

        # Direct call_native for OllamaClient
        if hasattr(self.model_client, 'call_native'):
            return self.model_client.call_native(
                system_prompt, user_prompt,
                max_new_tokens=max_tokens, temperature=temperature,
            )

        raise RuntimeError(f"Model client {type(self.model_client)} has no supported call method")

    # ------------------------------------------------------------------
    # Step 1: Domain detection
    # ------------------------------------------------------------------

    def detect_domains(self, problem: str) -> List[str]:
        """Detect problem domain(s) via LLM. Returns 1-3 domain strings."""
        user = FREEFORM_DOMAIN_DETECT_USER.format(problem=problem[:500])
        try:
            raw = self._call_llm(FREEFORM_DOMAIN_DETECT_SYSTEM, user,
                                 max_tokens=64, temperature=0.0)
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}")
            return ["general_knowledge"]

        # Parse: one domain per line, strip numbering/bullets
        domains = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip "1. ", "- ", "* " prefixes
            line = re.sub(r'^[\d\.\)\-\*\s]+', '', line).strip()
            if line and len(line) < 50:  # sanity: domain names are short
                domains.append(line.lower())
        return domains[:3] if domains else ["general_knowledge"]

    # ------------------------------------------------------------------
    # Step 2: Synonym-aware registry lookup
    # ------------------------------------------------------------------

    def match_domain(self, detected: str) -> Optional[str]:
        """Find a matching registry key for a detected domain string.

        Uses normalized keyword overlap against each domain's synonym list.
        Returns the registry key or None if no match.
        """
        detected_words = _normalize(detected)
        if not detected_words:
            return None

        best_key = None
        best_overlap = 0

        for key, entry in self.registry.items():
            synonyms = entry.get("synonyms", [key])
            for syn in synonyms:
                syn_words = _normalize(syn)
                overlap = len(detected_words & syn_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_key = key

        # Require at least 1 significant word overlap
        return best_key if best_overlap >= 1 else None

    # ------------------------------------------------------------------
    # Step 3: Example generation + validation for new domains
    # ------------------------------------------------------------------

    def generate_and_validate_examples(self, domain: str,
                                        n_generate: int = 5,
                                        min_valid: int = 2) -> List[dict]:
        """Generate example problem→variation pairs for a new domain, validate them.

        Returns list of {"original": str, "variation": str} dicts that passed validation.
        """
        user = FREEFORM_EXAMPLE_GEN_USER.format(domain=domain)
        try:
            raw = self._call_llm(FREEFORM_EXAMPLE_GEN_SYSTEM, user,
                                 max_tokens=2048, temperature=0.5)
        except Exception as e:
            logger.warning(f"Example generation failed for domain '{domain}': {e}")
            return []

        # Parse ORIGINAL: ... / VARIATION: ... pairs
        pairs = self._parse_example_pairs(raw)
        if not pairs:
            return []

        # Validate each pair using pipeline's validation prompt
        validated = []
        for pair in pairs:
            if self._validate_single_pair(pair["original"], pair["variation"]):
                validated.append(pair)

        # If too few passed, retry once
        if len(validated) < min_valid:
            logger.info(f"Only {len(validated)} examples validated for '{domain}', retrying...")
            try:
                raw2 = self._call_llm(FREEFORM_EXAMPLE_GEN_SYSTEM, user,
                                      max_tokens=2048, temperature=0.7)
                pairs2 = self._parse_example_pairs(raw2)
                for pair in pairs2:
                    if self._validate_single_pair(pair["original"], pair["variation"]):
                        validated.append(pair)
                        if len(validated) >= n_generate:
                            break
            except Exception:
                pass

        return validated[:n_generate]

    def _parse_example_pairs(self, raw: str) -> List[dict]:
        """Parse ORIGINAL:/VARIATION: formatted text into pairs."""
        pairs = []
        lines = raw.strip().split("\n")
        current_original = None
        current_variation = None

        for line in lines:
            line = line.strip()
            if not line:
                # Blank line might separate pairs — save current if complete
                if current_original and current_variation:
                    pairs.append({"original": current_original, "variation": current_variation})
                    current_original = None
                    current_variation = None
                continue

            m_orig = re.match(r'^ORIGINAL:\s*(.+)', line, re.IGNORECASE)
            m_var = re.match(r'^VARIATION:\s*(.+)', line, re.IGNORECASE)

            if m_orig:
                # Save previous pair if complete
                if current_original and current_variation:
                    pairs.append({"original": current_original, "variation": current_variation})
                current_original = m_orig.group(1).strip()
                current_variation = None
            elif m_var:
                current_variation = m_var.group(1).strip()

        # Don't forget the last pair
        if current_original and current_variation:
            pairs.append({"original": current_original, "variation": current_variation})

        return pairs

    def _validate_single_pair(self, original: str, variation: str) -> bool:
        """Validate one original→variation pair using pipeline's validation prompt."""
        user = VALIDATION_USER.format(original=original, variation=variation)
        try:
            raw = self._call_llm(VALIDATION_SYSTEM, user,
                                 max_tokens=10, temperature=0.0)
            resp = raw.strip().upper()
            return "VALID" in resp and "INVALID" not in resp
        except Exception:
            return False

    def register_domain(self, domain: str, examples: List[dict]):
        """Add a new domain to the registry and save to disk."""
        key = re.sub(r'[^a-z0-9_]', '_', domain.lower().strip())
        self.registry[key] = {
            "synonyms": [domain.lower().strip(), key],
            "examples": examples,
        }
        save_registry(self.registry, self.registry_path)
        logger.info(f"Registered new domain '{key}' with {len(examples)} examples")

    # ------------------------------------------------------------------
    # Step 4a: Streaming-friendly single-variation generation
    # ------------------------------------------------------------------

    def prepare_context(self, problem: str) -> dict:
        """Setup step: detect domains and collect few-shot examples.

        Call once before generating individual variations with generate_single_variation().
        Returns a context dict to pass into generate_single_variation().
        """
        domains = self.detect_domains(problem)
        logger.info(f"Detected domains: {domains}")

        all_examples = []
        resolved_domains = []
        for domain in domains:
            matched_key = self.match_domain(domain)
            if matched_key:
                all_examples.extend(self.registry[matched_key].get("examples", []))
                resolved_domains.append(matched_key)
            else:
                new_examples = self.generate_and_validate_examples(domain)
                if new_examples:
                    self.register_domain(domain, new_examples)
                    all_examples.extend(new_examples)
                    resolved_domains.append(domain)

        domain_label = "+".join(resolved_domains[:2]) if resolved_domains else "unknown"
        return {
            "domains": resolved_domains,
            "domain_label": domain_label,
            "domain_examples_text": self._format_examples(all_examples[:4]),
        }

    # Rotating diversity strategies — each variation uses a different technique
    DIVERSITY_STRATEGIES = [
        "Rewrite in a highly formal, academic tone — use technical vocabulary and complex sentence structure.",
        "Rewrite in casual, conversational language — as if explaining to a friend over coffee.",
        "Completely reorder the information: present the question first, then the setup (reverse the original flow).",
        "Rewrite from a different perspective — use third person narrative or frame it as a real-world scenario with named characters.",
        "Make it maximally concise — strip all unnecessary words, use the tersest phrasing possible.",
        "Make it elaborate — add contextual details and descriptive language (without changing the math/logic).",
        "Convert to passive voice throughout and restructure all clauses.",
        "Frame as an instruction/command rather than a question (e.g. 'Determine...' 'Calculate...' 'Find...').",
        "Rewrite using completely different vocabulary — replace every content word with a synonym where possible.",
        "Change the temporal framing — describe the scenario as past events being recalled, or as a hypothetical.",
        "Rewrite as a multi-sentence story or narrative before posing the question.",
        "Use an enumerated/structured format — break the information into numbered points or conditions.",
        "Rewrite from a pedagogical angle — as if a textbook is presenting this as a worked example prompt.",
        "Swap between interrogative and declarative forms — if original asks a question, state what needs to be found.",
        "Use domain-specific jargon and technical terminology appropriate to the subject area.",
        "Restructure using conditional phrasing — 'Given that...', 'Assuming...', 'Suppose...'.",
        "Rewrite with emphasis on different aspects — highlight what was background info in the original.",
        "Convert numerical representations — write out numbers as words, change units where valid.",
        "Frame as a verification task — 'Is it true that...' or 'Verify whether...'.",
        "Rewrite with inverted clause order — put subordinate clauses first, main clause last.",
    ]

    def generate_single_variation(self, problem: str, context: dict,
                                   previous_variations: List[str] = None,
                                   validate: bool = True,
                                   max_retries: int = 2) -> Optional[dict]:
        """Generate one variation, streaming-friendly.

        Args:
            problem: Original problem text.
            context: Dict from prepare_context().
            previous_variations: List of already-generated variation texts (for diversity).
            validate: Whether to validate the variation.
            max_retries: Number of generation retries on failure.

        Returns:
            Variation dict with "modified_problem", "transformation_type", etc., or None.
        """
        n_prev = len(previous_variations) if previous_variations else 0
        strategy = self.DIVERSITY_STRATEGIES[n_prev % len(self.DIVERSITY_STRATEGIES)]

        # Build diversity clause from previous variations
        if previous_variations:
            prev_text = "\n".join(
                f"  - Already generated: {v[:120]}" for v in previous_variations[-5:]
            )
            previous_block = f"Variations already generated (do NOT repeat or closely resemble these):\n{prev_text}"
            diversity_clause = " AND from all previously generated variations"
        else:
            previous_block = ""
            diversity_clause = ""

        user = FREEFORM_SINGLE_VARIATION_USER.format(
            problem=problem,
            domain_examples=context["domain_examples_text"],
            previous_variations=previous_block,
            diversity_clause=diversity_clause,
            strategy=strategy,
        )

        for attempt in range(max_retries + 1):
            try:
                raw = self._call_llm(FREEFORM_VARIATION_SYSTEM, user,
                                     max_tokens=1024, temperature=0.6)
            except Exception as e:
                logger.warning(f"Single variation generation failed (attempt {attempt+1}): {e}")
                if attempt < max_retries:
                    continue
                return None

            # Parse — extract first <question> tag
            variations = self._parse_variations(raw, problem, context.get("domains", []))
            if not variations:
                if attempt < max_retries:
                    continue
                return None

            variation = variations[0]

            # Check diversity against previous
            new_text = variation["modified_problem"].strip().lower()
            if previous_variations:
                is_dup = any(new_text == p.strip().lower() for p in previous_variations)
                if is_dup and attempt < max_retries:
                    continue

            # Validate
            if validate:
                is_valid = self._validate_single_pair(problem, variation["modified_problem"])
                if not is_valid:
                    if attempt < max_retries:
                        continue
                    return None

            return variation

        return None

    # ------------------------------------------------------------------
    # Step 4b: Batch variation generation (original method)
    # ------------------------------------------------------------------

    def generate_variations(self, problem: str, n: int = 8,
                            validate: bool = True) -> List[dict]:
        """Main entry point: detect domains, get examples, generate variations.

        Returns list of variation dicts in standard pipeline format:
        [{"original_problem": str, "modified_problem": str,
          "transformation_type": "freeform_<domain>", ...}, ...]
        """
        # Step 1: Detect domains
        domains = self.detect_domains(problem)
        logger.info(f"Detected domains: {domains}")

        # Step 2: Collect examples from registry (or generate new ones)
        all_examples = []
        resolved_domains = []
        for domain in domains:
            matched_key = self.match_domain(domain)
            if matched_key:
                all_examples.extend(self.registry[matched_key].get("examples", []))
                resolved_domains.append(matched_key)
            else:
                new_examples = self.generate_and_validate_examples(domain)
                if new_examples:
                    self.register_domain(domain, new_examples)
                    all_examples.extend(new_examples)
                    resolved_domains.append(domain)

        # Limit examples to 3-4 for few-shot (too many dilute the prompt)
        examples_for_prompt = all_examples[:4]

        # Step 3: Build prompt and generate
        domain_examples_text = self._format_examples(examples_for_prompt)
        user = FREEFORM_VARIATION_USER.format(
            problem=problem, n=n, domain_examples=domain_examples_text,
        )

        try:
            raw = self._call_llm(FREEFORM_VARIATION_SYSTEM, user,
                                 max_tokens=max(1024, n * 200),
                                 temperature=0.6)
        except Exception as e:
            logger.error(f"Variation generation failed: {e}")
            return []

        # Parse <question> tags
        variations = self._parse_variations(raw, problem, resolved_domains)

        # Step 4: Validate
        if validate and variations:
            variations = self._validate_variations(problem, variations)

        return variations

    def _format_examples(self, examples: List[dict]) -> str:
        """Format example pairs for inclusion in the variation prompt."""
        if not examples:
            return ""
        parts = ["Here are examples of good variations in this domain:"]
        for ex in examples:
            parts.append(f"  Original: {ex['original']}")
            parts.append(f"  Variation: {ex['variation']}")
            parts.append("")
        return "\n".join(parts)

    def _parse_variations(self, raw: str, problem: str,
                           domains: List[str]) -> List[dict]:
        """Parse LLM response for <question> tagged variations."""
        if not raw:
            return []

        # Clean thinking tags
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)

        variations = []
        # Find all <question>...</question> blocks
        matches = re.findall(r'<question>(.*?)</question>', raw, re.DOTALL | re.IGNORECASE)

        domain_label = "+".join(domains[:2]) if domains else "unknown"

        for i, match in enumerate(matches):
            text = clean_model_response(match.strip())
            if not text or not is_valid_question(text):
                continue
            # Skip if variation is too similar to original
            if text.strip().lower() == problem.strip().lower():
                continue

            variations.append({
                "original_problem": problem,
                "modified_problem": text,
                "transformation_type": f"freeform_{domain_label}",
                "generation_method": "freeform_llm",
                "confidence": "model_generated",
                "domains_involved": domains,
                "freeform_index": i + 1,
            })

        return variations

    # ------------------------------------------------------------------
    # Step 5: Validation of generated variations
    # ------------------------------------------------------------------

    def _validate_variations(self, problem: str,
                              variations: List[dict]) -> List[dict]:
        """Validate variations using pipeline's validation prompts.

        Drops invalid variations (no rectification in free-form mode since
        the LLM has full creative freedom — easier to just drop bad ones).
        """
        validated = []
        for v in variations:
            user = VALIDATION_USER.format(
                original=problem, variation=v["modified_problem"],
            )
            try:
                raw = self._call_llm(VALIDATION_SYSTEM, user,
                                     max_tokens=10, temperature=0.0)
                resp = raw.strip().upper()
                if "VALID" in resp and "INVALID" not in resp:
                    validated.append(v)
                else:
                    logger.debug(f"Dropped invalid freeform variation: {v['modified_problem'][:60]}...")
            except Exception:
                # On error, keep the variation (benefit of the doubt)
                validated.append(v)

        return validated

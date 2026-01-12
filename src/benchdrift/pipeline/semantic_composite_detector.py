#!/usr/bin/env python3
"""
Semantic Composite Detector - Embedding-based cluster discovery

Uses semantic clustering to discover semantically similar words:
1. Cluster candidates by embedding similarity (hierarchical clustering)
2. Use CAGrad at cluster level to find inter-cluster dependencies
3. Generate variations within and across dependent clusters

No spatial proximity - pure semantic similarity.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from benchdrift.models.model_client import RITSClient
import math
import logging


@dataclass
class Candidate:
    """Atomic candidate (word, number, phrase)."""
    text: str
    span: Tuple[int, int]  # (start, end) character positions
    type: str  # 'number', 'entity', 'verb', etc.


@dataclass
class SemanticCluster:
    """Semantic cluster - candidates grouped by embedding similarity."""
    cluster_id: int
    members: List[Candidate]  # All candidates in this cluster
    centroid: np.ndarray  # Average embedding of cluster members

    @property
    def texts(self) -> List[str]:
        """Get all texts in cluster."""
        return [m.text for m in self.members]

    @property
    def spans(self) -> List[Tuple[int, int]]:
        """Get all spans in cluster."""
        return [m.span for m in self.members]


@dataclass
class Segment:
    """
    Segment - represents a semantic cluster (legacy compatibility).

    In pure semantic clustering, this is equivalent to SemanticCluster.
    Kept for backward compatibility with existing pipeline code.
    """
    text: str
    span: Tuple[int, int]
    members: List[Candidate]
    segment_type: str  # 'single', 'cluster', etc.


@dataclass
class LinkedChunkGroup:
    """
    LinkedChunkGroup - DEPRECATED in pure semantic clustering.

    In the pure semantic approach, we only use clusters (no spatial linking).
    This class exists only for backward compatibility.
    """
    chunks: List[Candidate]
    linkage_type: str  # 'semantic', 'coreference', etc.
    group_id: int


class SemanticCompositeDetector:
    """
    Detects composites using embeddings + spatial proximity.

    Algorithm:
    1. Embed all candidates
    2. Build graph: edges = similar AND nearby
    3. Find connected components
    4. Classify as contiguous segments OR linked chunks
    """

    def __init__(
        self,
        # embedding_model: str = 'all-MiniLM-L6-v2',
        embedding_model: str = 'all-mpnet-base-v2',
        semantic_threshold: float = 0.35,  # Distance threshold for clustering
        verbose: bool = False,
    ):
        """
        Args:
            embedding_model: Sentence-transformers model name
            semantic_threshold: Distance threshold for hierarchical clustering
                              (lower = stricter, fewer merges; higher = more merging)
            verbose: If True, show detailed debug output
        """
        self.logger = logging.getLogger('BenchDrift.SemanticDetector')

        self.logger.info(f"🔧 Initializing Semantic Cluster Detector")
        self.logger.debug(f"   Embedding model: {embedding_model}")
        self.logger.debug(f"   Clustering threshold: {semantic_threshold}")

        self.embedder = SentenceTransformer(embedding_model)
        self.semantic_threshold = semantic_threshold

    def detect_clusters(
        self,
        problem: str,
        candidates: List[Candidate]
    ) -> List[SemanticCluster]:
        """
        Cluster candidates by semantic similarity using hierarchical clustering.

        Returns:
            List of SemanticCluster objects
        """
        if not candidates:
            return []

        # Special case: single candidate
        if len(candidates) == 1:
            emb = self.embedder.encode([candidates[0].text], show_progress_bar=False)[0]
            return [SemanticCluster(cluster_id=0, members=[candidates[0]], centroid=emb)]

        # 1. Get embeddings
        texts = [c.text for c in candidates]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)

        self.logger.debug(f"\n   🔍 DEBUG: Candidate Embeddings")
        self.logger.debug(f"   {'='*60}")
        for i, (cand, emb) in enumerate(zip(candidates, embeddings)):
            self.logger.debug(f"   [{i}] '{cand.text}' @ {cand.span}")
            self.logger.debug(f"       Embedding (first 5): {emb[:5]}")

        # 2. Hierarchical clustering with cosine distance
        distances = pdist(embeddings, metric='cosine')
        linkage_matrix = linkage(distances, method='average')

        # Get cluster labels using threshold
        cluster_labels = fcluster(linkage_matrix, t=self.semantic_threshold, criterion='distance')

        self.logger.debug(f"\n   🎯 DEBUG: Hierarchical Clustering")
        self.logger.debug(f"   {'='*60}")
        self.logger.debug(f"   Distance threshold: {self.semantic_threshold}")
        self.logger.debug(f"   Number of clusters found: {max(cluster_labels)}")

        # 3. Group candidates by cluster label
        clusters_dict = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(i)

        # 4. Create SemanticCluster objects
        clusters = []
        for cluster_id, member_indices in clusters_dict.items():
            member_cands = [candidates[i] for i in member_indices]
            member_embs = embeddings[member_indices]
            centroid = np.mean(member_embs, axis=0)

            # Clean cluster members: deduplicate, remove subsumed, merge adjacent
            cleaned_members = self._clean_cluster_members(member_cands, problem)

            cluster = SemanticCluster(
                cluster_id=cluster_id,
                members=cleaned_members,
                centroid=centroid
            )
            clusters.append(cluster)

        self.logger.debug(f"\n   📦 DEBUG: Semantic Clusters")
        self.logger.debug(f"   {'='*60}")
        for cluster in clusters:
            self.logger.debug(f"   Cluster {cluster.cluster_id}: {cluster.texts}")
            self.logger.debug(f"       Positions: {cluster.spans}")

        self.logger.info(f"   Detected {len(clusters)} semantic clusters")

        return clusters

    def _clean_cluster_members(self, members: List[Candidate], problem: str) -> List[Candidate]:
        """
        Clean cluster members by:
        1. Removing exact duplicates (same text + same span)
        2. Removing subsumed spans (span A within span B and text A in text B)
        3. Merging adjacent keywords (consecutive spans in the problem)
        """
        if not members:
            return members

        # Step 1: Remove exact duplicates
        seen = set()
        unique_members = []
        for m in members:
            key = (m.text, m.span)
            if key not in seen:
                seen.add(key)
                unique_members.append(m)

        # Step 2: Remove subsumed spans
        filtered_members = []
        for i, m1 in enumerate(unique_members):
            is_subsumed = False
            for j, m2 in enumerate(unique_members):
                if i == j:
                    continue
                # Check if m1 is fully contained within m2's span
                if m1.span[0] >= m2.span[0] and m1.span[1] <= m2.span[1]:
                    # Also verify m1's text is substring of m2's text
                    if m1.text in m2.text:
                        is_subsumed = True
                        break
            if not is_subsumed:
                filtered_members.append(m1)

        # Step 3: Merge adjacent keywords
        # Sort by span start position
        sorted_members = sorted(filtered_members, key=lambda x: x.span[0])

        merged_members = []
        i = 0
        while i < len(sorted_members):
            current = sorted_members[i]
            # Try to merge with next members if they're adjacent
            merged_span_start = current.span[0]
            merged_span_end = current.span[1]
            j = i + 1

            while j < len(sorted_members):
                next_member = sorted_members[j]
                # Check if adjacent (end of current == start of next, or end+1 for space)
                if merged_span_end == next_member.span[0] or merged_span_end + 1 == next_member.span[0]:
                    merged_span_end = next_member.span[1]
                    j += 1
                else:
                    break

            # If we merged multiple members, create new merged candidate
            if j > i + 1:
                merged_text = problem[merged_span_start:merged_span_end]
                merged_candidate = Candidate(
                    text=merged_text,
                    span=(merged_span_start, merged_span_end),
                    type=current.type
                )
                merged_members.append(merged_candidate)
            else:
                # No merge, keep original
                merged_members.append(current)

            i = j

        return merged_members

    def _build_similarity_graph(
        self,
        candidates: List[Candidate],
        embeddings: np.ndarray
    ) -> nx.Graph:
        """
        Build graph where edges = semantically similar AND sequentially adjacent.

        Uses SEQUENTIAL POSITION (not character distance):
        - Candidates are already sorted by span position
        - Only connect if literally next to each other (position i and i+1)
        - AND semantically similar
        """
        G = nx.Graph()
        n = len(candidates)

        for i in range(n):
            G.add_node(i)

        # Only check ADJACENT candidates in sequence
        for i in range(n - 1):
            j = i + 1  # Next candidate in sequence

            # Semantic similarity check
            cos_dist = cosine(embeddings[i], embeddings[j])
            if cos_dist > self.semantic_threshold:
                continue

            # Check if they're actually close in the text (not separated by huge gap)
            gap = candidates[j].span[0] - candidates[i].span[1]
            if gap > self.spatial_threshold:
                continue

            # Both checks passed → add edge
            G.add_edge(i, j, semantic_dist=cos_dist, gap=gap)

        return G

    def _classify_components(
        self,
        components: List[Set[int]],
        candidates: List[Candidate],
        problem: str
    ) -> Tuple[List[Segment], List[LinkedChunkGroup]]:
        """
        Classify connected components as:
        - Contiguous segments (if spatially sequential)
        - Linked chunks (if spatially separated)
        """
        self.logger.debug(f"\n   🎯 DEBUG: Classifying Clusters into Segments")
        self.logger.debug(f"   {'='*60}")

        segments = []
        linked_groups = []

        for comp_idx, component_ids in enumerate(components):
            comp_list = sorted(list(component_ids))
            comp_texts = [candidates[i].text for i in comp_list]

            if len(component_ids) == 1:
                # Single candidate - treat as segment
                idx = list(component_ids)[0]
                self.logger.debug(f"   Cluster {comp_idx}: Single candidate '{candidates[idx].text}' → segment")
                segments.append(Segment(
                    text=candidates[idx].text,
                    span=candidates[idx].span,
                    members=[candidates[idx]],
                    segment_type='single'
                ))
                continue

            # Multiple candidates - check if contiguous
            sorted_ids = sorted(component_ids, key=lambda x: candidates[x].span[0])

            # Check contiguity
            is_contig = self._is_contiguous(sorted_ids, candidates)

            # Calculate gaps for debug
            gaps = []
            for i in range(len(sorted_ids) - 1):
                curr_end = candidates[sorted_ids[i]].span[1]
                next_start = candidates[sorted_ids[i + 1]].span[0]
                gap = next_start - curr_end
                gaps.append(gap)

            self.logger.debug(f"   Cluster {comp_idx}: {comp_texts}")
            self.logger.debug(f"      Gaps between members: {gaps}")
            self.logger.debug(f"      Contiguous_gap threshold: {self.contiguous_gap}")
            self.logger.debug(f"      Is contiguous: {is_contig}")

            if is_contig:
                # Merge into single segment
                segment = self._merge_into_segment(sorted_ids, candidates, problem)
                self.logger.debug(f"      → Merged into segment: '{segment.text}'")
                segments.append(segment)
            else:
                # Keep as linked chunks
                linked_group = self._create_linked_group(sorted_ids, candidates)
                self.logger.debug(f"      → Kept as linked chunks (type: {linked_group.linkage_type})")
                linked_groups.append(linked_group)

        return segments, linked_groups

    def _is_contiguous(self, sorted_ids: List[int], candidates: List[Candidate]) -> bool:
        """
        Check if candidates are spatially contiguous (sequential in text).
        """
        for i in range(len(sorted_ids) - 1):
            curr_end = candidates[sorted_ids[i]].span[1]
            next_start = candidates[sorted_ids[i + 1]].span[0]
            gap = next_start - curr_end

            # If gap too large, not contiguous
            if gap > self.contiguous_gap:
                return False

        return True

    def _merge_into_segment(
        self,
        sorted_ids: List[int],
        candidates: List[Candidate],
        problem: str
    ) -> Segment:
        """
        Merge contiguous candidates into single segment.
        """
        members = [candidates[i] for i in sorted_ids]
        start = min(c.span[0] for c in members)
        end = max(c.span[1] for c in members)

        # Extract text from original problem (preserves spacing)
        text = problem[start:end]

        return Segment(
            text=text,
            span=(start, end),
            members=members,
            segment_type='contiguous'
        )

    def _create_linked_group(
        self,
        sorted_ids: List[int],
        candidates: List[Candidate]
    ) -> LinkedChunkGroup:
        """
        Create linked chunk group for spatially separated candidates.
        """
        chunks = [candidates[i] for i in sorted_ids]

        # Determine linkage type based on candidate types
        types = set(c.type for c in chunks)

        if 'entity' in types or 'pronoun' in types:
            linkage_type = 'coreference'
        elif len(types) == 1:
            linkage_type = 'parallel'  # Same type, different locations
        else:
            linkage_type = 'semantic'

        return LinkedChunkGroup(
            chunks=chunks,
            linkage_type=linkage_type,
            group_id=hash(tuple(c.text for c in chunks))
        )

    def format_for_variation_prompt(
        self,
        segments: List[Segment],
        linked_groups: List[LinkedChunkGroup]
    ) -> str:
        """
        Format segments and linked chunks for variation generation prompt.

        Returns instruction text for LLM.
        """
        instructions = []

        # Contiguous segments
        if segments:
            instructions.append("CONTIGUOUS SEGMENTS (keep unchanged):")
            for seg in segments:
                instructions.append(f"  - '{seg.text}' at position {seg.span}")

        # Linked chunks
        if linked_groups:
            instructions.append("\nLINKED CHUNKS (change together or not at all):")
            for group in linked_groups:
                chunk_strs = [f"'{c.text}' at {c.span}" for c in group.chunks]
                instructions.append(f"  - Group ({group.linkage_type}): {', '.join(chunk_strs)}")

        return "\n".join(instructions)


class CAGradDependencyTester:
    """
    Test segment dependencies using CAGrad.

    Tests if segments are truly dependent via:
    |g_combined - sum(g_individual)| > threshold
    """

    def __init__(
        self,
        model_client,
        embedder=None,
        dependency_threshold: float = 0.10,
        rits_batch_size: int = 1
    ):
        """
        Args:
            model_client: Model client with get_model_response_with_logprobs
            embedder: SentenceTransformer model for finding semantic opposites (optional)
            dependency_threshold: Threshold for non-additive dependency
            rits_batch_size: Batch size for RITS API calls (default: 10)
        """
        self.model_client = model_client
        self.embedder = embedder
        self.dependency_threshold = dependency_threshold
        self.rits_batch_size = rits_batch_size

        # Pre-compute vocabulary embeddings if embedder provided
        if self.embedder:
            self._init_vocabulary()

    def build_dependency_graph(
        self,
        problem: str,
        answer: str,
        clusters: List[SemanticCluster]
    ) -> nx.Graph:
        """
        Build cluster-level dependency graph using CAGrad with batched inference.

        For each cluster, masks ALL occurrences of cluster members with [MASK].
        Tests individual cluster gradients and pairwise cluster combinations.

        Returns graph where:
        - Nodes = clusters
        - Edges = dependent clusters (non-additive interaction)
        """
        self.logger.debug(f"\n🧪 Cluster-Level CAGrad Testing (Batched)")
        self.logger.debug(f"   Testing {len(clusters)} semantic clusters")
        self.logger.debug(f"   {'='*60}")

        # Step 1: Collect all prompts to test (baseline + all masked versions)
        prompts_to_test = []
        prompt_metadata = []

        # Baseline probability (original problem)
        prompts_to_test.append(problem)
        prompt_metadata.append({'type': 'baseline'})

        # Individual cluster masked prompts
        for i, cluster in enumerate(clusters):
            masked = self._mask_spans(problem, cluster.spans)
            prompts_to_test.append(masked)
            prompt_metadata.append({'type': 'individual', 'cluster_id': i})

        # Pairwise cluster masked prompts (for significant clusters only - we'll do a second pass)
        # For now, just test individual clusters in batch

        # Step 2: Batch inference for all prompts
        self.logger.debug(f"\n   Running batched inference for {len(prompts_to_test)} prompts...")
        probabilities = self._get_answer_probabilities_batch(prompts_to_test, answer)

        # Step 3: Extract baseline and individual gradients
        baseline_prob = probabilities[0]
        individual_gradients = {}

        for i, cluster in enumerate(clusters):
            masked_prob = probabilities[i + 1]  # +1 because baseline is at index 0
            gradient = abs(baseline_prob - masked_prob)
            individual_gradients[i] = gradient

            cluster_text = ", ".join(cluster.texts[:3])
            if len(cluster.texts) > 3:
                cluster_text += f" (+{len(cluster.texts)-3} more)"
            self.logger.debug(f"   Cluster {i}: [{cluster_text}]")
            self.logger.debug(f"      Members: {len(cluster.members)} | Gradient: {gradient:.3f}")

        # Filter clusters with significant gradients
        significant_clusters = [i for i, g in individual_gradients.items() if g > 0.01]
        self.logger.debug(f"\n   Clusters with gradient > 0.01: {len(significant_clusters)}/{len(clusters)}")

        # Step 4: Test pairwise dependencies (batched)
        self.logger.debug(f"\n🔗 Testing Pairwise Cluster Dependencies (Batched)")
        self.logger.debug(f"   {'='*60}")

        pairwise_prompts = []
        pairwise_metadata = []

        for i in significant_clusters:
            for j in significant_clusters:
                if i >= j:
                    continue
                # Mask both clusters
                combined_spans = clusters[i].spans + clusters[j].spans
                masked = self._mask_spans(problem, combined_spans)
                pairwise_prompts.append(masked)
                pairwise_metadata.append({'cluster_i': i, 'cluster_j': j})

        # Batch inference for pairwise tests
        if pairwise_prompts:
            self.logger.debug(f"   Running batched inference for {len(pairwise_prompts)} pairwise tests...")
            pairwise_probs = self._get_answer_probabilities_batch(pairwise_prompts, answer)
        else:
            pairwise_probs = []

        # Step 5: Build dependency graph
        G = nx.Graph()
        for i in range(len(clusters)):
            G.add_node(i, cluster=clusters[i], gradient=individual_gradients[i])

        dependent_pairs = 0
        for idx, metadata in enumerate(pairwise_metadata):
            i = metadata['cluster_i']
            j = metadata['cluster_j']
            g_combined = abs(baseline_prob - pairwise_probs[idx])
            g_expected = individual_gradients[i] + individual_gradients[j]
            dependency_score = abs(g_combined - g_expected)

            if dependency_score > self.dependency_threshold:
                G.add_edge(i, j, dependency_score=dependency_score)
                dependent_pairs += 1
                self.logger.debug(f"   ✓ Clusters {i} ↔ {j} are DEPENDENT")
                self.logger.debug(f"      Combined gradient: {g_combined:.3f}")
                self.logger.debug(f"      Expected (additive): {g_expected:.3f}")
                self.logger.debug(f"      Dependency score: {dependency_score:.3f}")

        self.logger.debug(f"\n   📊 Summary:")
        self.logger.debug(f"      Tested {len(pairwise_metadata)} pairs")
        self.logger.debug(f"      Found {dependent_pairs} dependent cluster pairs")

        return G

    def _init_vocabulary(self):
        """Initialize and pre-compute embeddings for replacement vocabularies."""
        # Type-specific vocabularies for counterfactual replacements
        self.vocabularies = {
            'number': ['1', '2', '3', '5', '7', '10', '15', '20', '25', '50', '100'],
            'entity': ['Alex', 'Jordan', 'Taylor', 'Sam', 'Morgan', 'Casey', 'Riley'],
            'object': ['apple', 'book', 'car', 'pen', 'table', 'chair', 'box'],
            'temporal': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'default': ['something', 'anything', 'nothing', 'everything', 'item', 'thing', 'object']
        }

        # Pre-compute embeddings for each vocabulary
        self.vocab_embeddings = {}
        for vocab_type, words in self.vocabularies.items():
            self.vocab_embeddings[vocab_type] = self.embedder.encode(words, show_progress_bar=False)

    def _infer_candidate_type(self, text: str) -> str:
        """Infer candidate type from text."""
        import re
        text = text.strip()

        if re.match(r'^-?\d+\.?\d*$', text):
            return 'number'
        if text[0].isupper() and len(text) > 1 and text.isalpha():
            return 'entity'
        if text.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            return 'temporal'
        return 'default'

    def _find_opposite_word(self, original_text: str) -> str:
        """
        Find semantically opposite word using embedding distance.

        Uses the same embedder as clustering for principled counterfactuals.
        Finds the word in type-specific vocabulary with maximum cosine distance.
        """
        if not self.embedder:
            return "[MASK]"  # Fallback if no embedder

        # Infer type and get appropriate vocabulary
        candidate_type = self._infer_candidate_type(original_text)
        vocab = self.vocabularies.get(candidate_type, self.vocabularies['default'])
        vocab_embs = self.vocab_embeddings.get(candidate_type, self.vocab_embeddings['default'])

        # Get embedding of original text
        original_emb = self.embedder.encode([original_text], show_progress_bar=False)[0]

        # Find word with maximum cosine distance (most semantically different)
        from scipy.spatial.distance import cosine
        distances = [cosine(original_emb, ve) for ve in vocab_embs]

        # Filter out the original word if it's in vocabulary
        valid_indices = [i for i, word in enumerate(vocab) if word.lower() != original_text.lower()]

        if valid_indices:
            max_idx = max(valid_indices, key=lambda i: distances[i])
            return vocab[max_idx]
        else:
            # Fallback: use word with max distance regardless
            return vocab[np.argmax(distances)]

    def _mask_spans(self, text: str, spans: List[Tuple[int, int]]) -> str:
        """
        Replace spans with semantically opposite words using embedding distance.

        For each span, finds the most semantically distant word from a type-specific
        vocabulary using the same embedding model as clustering.

        Args:
            text: Original text
            spans: List of (start, end) positions to replace

        Returns:
            Text with spans replaced by semantically opposite words
        """
        if not self.embedder:
            # Fallback to [MASK] if no embedder
            masked = text
            for span in sorted(spans, reverse=True):
                masked = masked[:span[0]] + "[MASK]" + masked[span[1]:]
            return masked

        # Replace each span with semantically opposite word
        masked = text
        for span in sorted(spans, reverse=True):
            original_text = text[span[0]:span[1]]
            replacement = self._find_opposite_word(original_text)
            masked = masked[:span[0]] + replacement + masked[span[1]:]

        return masked

    def _get_answer_probabilities_batch(
        self,
        prompts: List[str],
        expected_answer: str
    ) -> List[float]:
        """
        Get probabilities for multiple prompts using RITS batch chunking.

        Args:
            prompts: List of prompts to test
            expected_answer: Expected answer

        Returns:
            List of probabilities (one per prompt)
        """
        if not prompts:
            return []

        # Chunk prompts into batches of rits_batch_size
        system_prompt = "You are a helpful assistant. Provide concise, direct answers."
        all_probabilities = []

        for i in range(0, len(prompts), self.rits_batch_size):
            batch_prompts = prompts[i:i + self.rits_batch_size]
            batch_system_prompts = [system_prompt] * len(batch_prompts)

            self.logger.debug(f"      Batch {i//self.rits_batch_size + 1}: Processing {len(batch_prompts)} prompts...")
            self.logger.debug(f"      DEBUG: First prompt preview: {batch_prompts[0][:100]}...")
            self.logger.debug(f"      DEBUG: Calling get_model_response_with_logprobs...")

            # Call RITS API for this batch
            results = self.model_client.get_model_response_with_logprobs(
                batch_system_prompts,
                batch_prompts,
                max_new_tokens=50,
                temperature=0.1
            )

            self.logger.debug(f"      DEBUG: Got {len(results)} results back")

            # Extract probabilities from results
            for result in results:
                logprobs = result.get('logprobs', [])
                if logprobs:
                    avg_logprob = sum(logprobs) / len(logprobs)
                    prob = math.exp(avg_logprob)
                else:
                    prob = 0.5
                all_probabilities.append(min(prob, 1.0))

        return all_probabilities

    def _get_gradient(
        self,
        problem: str,
        answer: str,
        span: Tuple[int, int] or List[Tuple[int, int]]
    ) -> float:
        """
        Get CAGrad gradient by masking span(s) and measuring impact.
        """
        # Handle both single span and list of spans (for linked groups)
        if isinstance(span, list):
            # Mask all spans
            masked = problem
            for s in sorted(span, reverse=True):  # Reverse to maintain positions
                masked = masked[:s[0]] + "[MASK]" + masked[s[1]:]
        else:
            # Single span
            masked = problem[:span[0]] + "[MASK]" + problem[span[1]:]

        # Get probabilities
        baseline_prob = self._get_answer_probability(problem, answer)
        masked_prob = self._get_answer_probability(masked, answer)

        # Gradient = change in probability
        gradient = abs(baseline_prob - masked_prob)

        return gradient

    def _get_answer_probability(self, prompt: str, expected_answer: str) -> float:
        """
        Get probability of expected answer using logprobs.
        """
        import math

        results = self.model_client.get_model_response_with_logprobs(
            system_prompts=["You are a helpful assistant. Provide concise, direct answers."],
            user_prompts=[prompt],
            max_new_tokens=50,
            temperature=0.1
        )

        if not results:
            return 0.5

        result = results[0]
        logprobs = result.get('logprobs', [])

        if logprobs:
            avg_logprob = sum(logprobs) / len(logprobs)
            prob = math.exp(avg_logprob)
        else:
            prob = 0.5

        return min(prob, 1.0)

    def _combine_spans(
        self,
        span1: Tuple[int, int] or List[Tuple[int, int]],
        span2: Tuple[int, int] or List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Combine two spans (handles both single and multiple spans).
        """
        spans = []

        if isinstance(span1, tuple):
            spans.append(span1)
        else:
            spans.extend(span1)

        if isinstance(span2, tuple):
            spans.append(span2)
        else:
            spans.extend(span2)

        return spans


def example_usage():
    """Example demonstrating cluster-based detection with CAGrad."""

    # Example problem
    problem = "Sarah bought 3 apples for $2 each on Monday. She bought 5 oranges for $1.50 each on Tuesday. How much did Sarah spend in total?"
    answer = "$16.50"

    # Simulate detected candidates
    candidates = [
        Candidate("Sarah", (0, 5), "entity"),
        Candidate("bought", (6, 12), "verb"),
        Candidate("3", (13, 14), "number"),
        Candidate("apples", (15, 21), "object"),
        Candidate("$2", (26, 28), "money"),
        Candidate("each", (29, 33), "quantifier"),
        Candidate("Monday", (37, 43), "temporal"),
        Candidate("She", (45, 48), "pronoun"),
        Candidate("5", (56, 57), "number"),
        Candidate("oranges", (58, 65), "object"),
        Candidate("$1.50", (70, 75), "money"),
        Candidate("Tuesday", (84, 91), "temporal"),
    ]

    # 1. Detect semantic clusters
    detector = SemanticCompositeDetector()
    clusters = detector.detect_clusters(problem, candidates)

    self.logger.debug("\n" + "="*70)
    self.logger.debug("SEMANTIC CLUSTERS")
    self.logger.debug("="*70)

    for cluster in clusters:
        self.logger.debug(f"\nCluster {cluster.cluster_id}:")
        self.logger.debug(f"  Members: {cluster.texts}")
        self.logger.debug(f"  Positions: {cluster.spans}")

    # 2. Test cluster dependencies with CAGrad
    self.logger.debug("\n" + "="*70)
    self.logger.debug("CAGRAD DEPENDENCY TESTING")
    self.logger.debug("="*70)

    client = RITSClient("phi-4")
    dependency_tester = CAGradDependencyTester(
        client,
        embedder=detector.embedder  # Pass embedder for principled counterfactuals
    )

    dep_graph = dependency_tester.build_dependency_graph(
        problem, answer, clusters
    )

    self.logger.debug("\n" + "="*70)
    self.logger.debug("DEPENDENCY GRAPH SUMMARY")
    self.logger.debug("="*70)
    self.logger.debug(f"Total clusters: {dep_graph.number_of_nodes()}")
    self.logger.debug(f"Dependent cluster pairs: {dep_graph.number_of_edges()}")

    if dep_graph.number_of_edges() > 0:
        self.logger.debug("\nDependent Cluster Pairs:")
        for i, j, data in dep_graph.edges(data=True):
            cluster_i = clusters[i]
            cluster_j = clusters[j]
            self.logger.debug(f"  Cluster {i} {cluster_i.texts} ↔ Cluster {j} {cluster_j.texts}")
            self.logger.debug(f"    Dependency score: {data['dependency_score']:.3f}")


if __name__ == '__main__':
    example_usage()

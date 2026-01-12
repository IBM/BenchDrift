#!/usr/bin/env python3
"""
Variation Answer Generator

Generates model responses to problem variations with chain-of-thought reasoning.
Supports both streaming (as variations are generated) and offline (from JSONL) processing.
Uses RITS client with parallel processing or VLLM with batching.
"""

import json
from tqdm import tqdm
import logging
import time
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import sys
import os

# Add the project root to path for imports

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchdrift.pipeline.comprehensive_variation_engine_v2 import create_model_client_for_variations

logger = logging.getLogger('BenchDrift')

@dataclass
class VariationProblem:
    """Data class for a problem variation"""
    variation_id: str
    original_problem: str
    modified_problem: str
    transformation_type: str
    original_component: str
    new_component: str
    expected_answer: Optional[str] = None
    domain: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class GeneratedResponse:
    """Data class for generated model response"""
    variation_id: str
    problem: str
    model_response: str
    thinking: str
    final_answer: str
    generation_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class VariationAnswerGenerator:
    """
    Main class for generating answers to problem variations using evaluation models.
    Supports both RITS (parallel) and VLLM (batching) client modes.
    """
    
    def __init__(self,
                 client_type: str = 'rits',
                 model_name: str = 'microsoft/Phi-4-reasoning',
                 max_workers: int = 4,
                 batch_size: int = 8,
                 rits_batch_size: int = 5,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1,
                 disable_cot: bool = False):
        """
        Initialize the answer generator.

        Args:
            client_type: 'rits' or 'vllm'
            model_name: Model to use for generation
            max_workers: Only used for non-RITS clients (RITS handles parallelization internally)
            batch_size: Batch size for VLLM processing
            rits_batch_size: Batch size for RITS processing (default: 100)
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            disable_cot: Disable chain-of-thought reasoning for faster generation (default: False)
        """
        self.client_type = client_type
        self.model_name = model_name
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.rits_batch_size = rits_batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.disable_cot = disable_cot
        
        # Initialize model client
        self.model_client = self._initialize_model_client()
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_time': 0.0,
            'avg_time_per_problem': 0.0
        }
        
        logger.debug(f"✅ Initialized VariationAnswerGenerator")
        logger.debug(f"   Client: {client_type}")
        logger.debug(f"   Model: {model_name}")
        if client_type == 'rits':
            logger.debug(f"   RITS batch size: {rits_batch_size}")
        else:
            logger.debug(f"   VLLM batch size: {batch_size}")
        if disable_cot:
            logger.debug(f"   ⚡ CoT DISABLED: Using direct answer mode for faster generation")
        
    def _initialize_model_client(self):
        """Initialize the appropriate model client"""
        try:
            model_client = create_model_client_for_variations(
                client_type=self.client_type,
                model_name=self.model_name
            )
            return model_client
        except Exception as e:
            logger.debug(f"❌ Failed to initialize {self.client_type} client: {e}")
            raise
    
    def create_chain_of_thought_prompt(self, problem: str, domain: Optional[str] = None) -> tuple:
        """
        Create a chain-of-thought prompt for problem solving.
        Returns (system_prompt, user_prompt) tuple for proper separation.

        Args:
            problem: The problem to solve
            domain: Optional domain hint for specialized prompting

        Returns:
            Tuple of (system_prompt, user_prompt) strings
        """

        # If CoT is disabled, use simplified direct answer prompt
        if self.disable_cot:
            system_prompt = """You are a precise assistant. Provide ONLY the final answer with NO reasoning.

The task may be: answering a question, completing a sentence, selecting from multiple choices, or any other instruction format.

CRITICAL INSTRUCTIONS:
1. DO NOT include <think> tags
2. DO NOT show your working or reasoning
3. DO NOT write explanations
4. ONLY provide the final answer in <answer> tags

CORRECT FORMAT (do this):
<answer>
42
</answer>

WRONG FORMAT (DO NOT do this):
<think>
Step 1: First I need to...
Step 2: Then I calculate...
</think>
<answer>
42
</answer>

Your response must contain ONLY the <answer> section with the final result. Nothing else."""

            user_prompt = f"""TASK: {problem}

Now complete the task following the exact format specified."""

            return (system_prompt, user_prompt)

        # CoT prompt - separate system (general) and user (task-specific)
        system_prompt = """You are a precise assistant. You MUST follow the exact format specified below.

The task may be: answering a question, completing a sentence, selecting from multiple choices, understanding intent, or any other instruction format. Regardless of task type, use the SAME format.

ABSOLUTE FORMATTING RULES - VIOLATION WILL RESULT IN FAILURE:
1. Your response MUST contain EXACTLY these sections in this order:
   - <think> section with your reasoning, calculations, and working
   - </think> to close thinking section
   - <answer> section with ONLY the final result
   - </answer> to close answer section
   - NOTHING ELSE - YOUR RESPONSE ENDS HERE

2. THINK SECTION RULES:
   - Your reasoning MUST contain AT LEAST 2-3 steps - this is REQUIRED
   - Show ALL relevant calculations and logic step by step
   - Even for simple tasks, you MUST show your work
   - Empty <think> sections are FORBIDDEN and will result in failure
   - Be clear and focused, but MUST include actual reasoning content

3. ANSWER SECTION RULES:
   - The <answer> section MUST contain your final result - NEVER leave it empty
   - Contains ONLY the final result (number, time, date, letter, word, phrase)
   - NO explanations, NO reasoning, NO additional text
   - NO "The answer is", NO "Therefore", NO calculations
   - Just the bare result that completes the task

4. CRITICAL: STOP IMMEDIATELY AFTER </answer>
   - DO NOT write any additional text after </answer>
   - DO NOT provide further explanations outside the tags
   - DO NOT repeat your reasoning after the tags
   - YOUR RESPONSE MUST END with </answer>

REQUIRED FORMAT TEMPLATE:
<think>
[Your step-by-step reasoning - MUST have content]
[Your calculations - MUST show work]
[Your verification - MUST be present]
</think>

<answer>
[ONLY final result - MUST NOT be empty]
</answer>

FORMAT EXAMPLE (for demonstration purposes ONLY - do NOT use this data in your actual response):
<think>
Time per wall: 2h 30m 45s = 9045 seconds
Total for 4 walls: 9045 × 4 = 36180 seconds
Convert back: 36180s = 10h 3m 0s
</think>

<answer>
10:03:00
</answer>

CRITICAL WARNING: The example above is ONLY to show the format structure. When responding to the actual task below, you MUST:
- Analyze the ACTUAL task given to you
- Provide reasoning specific to THAT task
- Give the answer for THAT task
- NEVER reference or reuse content from the format example

ABSOLUTELY WRONG - EMPTY THINK IS FORBIDDEN:
<think>
</think>

<answer>
10:03:00
</answer>

ABSOLUTELY WRONG - EMPTY ANSWER IS FORBIDDEN:
<think>
Step 1: Analyzing the task...
Step 2: Considering options...
Step 3: Making decision...
</think>

<answer>
</answer>

WRONG - Text after answer tag:
<think>
Some calculation
</think>

<answer>
10:03:00
</answer>

The calculation shows that we need 10 hours...

REMEMBER: Your response MUST END immediately after </answer> with no additional text."""

        # User prompt - task-specific (task + domain guidance)
        domain_specific_guidance = ""
        if domain:
            if domain.lower() == 'math':
                domain_specific_guidance = """
For mathematical tasks:
- Show all calculations step by step
- Verify your arithmetic
- State any formulas or theorems you use
- Check if your answer makes sense
"""
            elif domain.lower() == 'temporal':
                domain_specific_guidance = """
For temporal/time-based tasks:
- Convert times/dates to a consistent format if needed
- Show your date/time calculations clearly
- Account for different time zones if relevant
- Double-check your temporal arithmetic
"""
            elif domain.lower() in ['nl', 'natural_language']:
                domain_specific_guidance = """
For reasoning/comprehension tasks:
- Break down the task into logical steps
- Consider all given information
- Make explicit connections between facts
- Verify your reasoning is sound
"""

        user_prompt = f"""TASK: {problem}

{domain_specific_guidance}

Now complete THIS SPECIFIC TASK following the EXACT format specified.
Remember: Both <think> and <answer> sections MUST contain actual content - empty sections are FORBIDDEN."""

        return (system_prompt, user_prompt)
    
    def test_prompt_and_parsing(self, test_problem: str = "What is 2+2?") -> None:
        """Test the prompt format and parsing with a simple problem."""
        logger.debug("🧪 Testing prompt format and parsing...")

        # Create test prompt
        system_prompt, user_prompt = self.create_chain_of_thought_prompt(test_problem)
        logger.debug(f"📝 Generated system prompt:\n{system_prompt[:200]}...\n")
        logger.debug(f"📝 Generated user prompt:\n{user_prompt[:200]}...\n")
        
        # Test with a correctly formatted response
        test_response = """<think>
Step 1: This is a simple addition problem
Step 2: Add 2 + 2 = 4
Step 3: Verify: 4 is correct
Verification: Yes, 2+2=4
</think>

<answer>
4
</answer>"""
        
        thinking, answer = self.parse_model_response(test_response)
        logger.debug(f"✅ Test parsing results:")
        logger.debug(f"   Thinking extracted: {bool(thinking)}")
        logger.debug(f"   Answer extracted: {bool(answer)}")
        logger.debug(f"   Thinking: {thinking[:50]}...")
        logger.debug(f"   Answer: {answer}")
        
        if thinking and answer and "Step 1" in thinking and answer == "4":
            logger.debug("🎉 Prompt and parsing test PASSED!")
        else:
            logger.debug("❌ Prompt and parsing test FAILED!")

        return (system_prompt, user_prompt)
    
    def parse_model_response(self, response: str) -> tuple[str, str]:
        """
        Parse model response to extract thinking and final answer with robust error handling.
        
        Args:
            response: Raw model response
            
        Returns:
            Tuple of (thinking, final_answer)
        """
        thinking = ""
        final_answer = ""
        
        if not response or not response.strip():
            return "Empty response received", "No answer"
        
        # Clean the response
        response = response.strip()
        
        try:
            # Extract thinking section with case-insensitive search
            think_patterns = [
                ("<think>", "</think>"),
                ("<THINK>", "</THINK>"),
                ("**Think:**", "**Answer:**"),
                ("Think:", "Answer:")
            ]
            
            for start_tag, end_tag in think_patterns:
                think_start = response.find(start_tag)
                think_end = response.find(end_tag)
                
                if think_start != -1 and think_end != -1:
                    thinking = response[think_start + len(start_tag):think_end].strip()
                    break
            
            # Extract answer section with multiple possible patterns
            answer_patterns = [
                ("<answer>", "</answer>"),
                ("<ANSWER>", "</ANSWER>"),
                ("**Answer:**", "\n\n"),
                ("Answer:", "\n\n"),
                ("Final Answer:", "\n\n")
            ]

            for start_tag, end_tag in answer_patterns:
                # For <answer> tags, find the LAST occurrence to avoid matching instructional text
                if start_tag in ["<answer>", "<ANSWER>"]:
                    # Find last occurrence of opening tag
                    answer_start = response.rfind(start_tag)
                else:
                    answer_start = response.find(start_tag)

                if answer_start != -1:
                    content_start = answer_start + len(start_tag)

                    if end_tag == "\n\n":
                        # Find end of line or end of string
                        answer_end = response.find("\n", content_start)
                        if answer_end == -1:
                            answer_end = len(response)
                    else:
                        answer_end = response.find(end_tag, content_start)
                        if answer_end == -1:
                            # Opening tag found but no closing tag - take everything after opening
                            answer_end = len(response)

                    raw_answer = response[content_start:answer_end].strip()

                    # If <answer> tag is present (with or without closing tag), use content as-is
                    # Only apply cleaning heuristics if NO answer tags at all
                    if start_tag in ["<answer>", "<ANSWER>"]:
                        # Answer tag present - keep everything within tags as-is
                        final_answer = raw_answer if raw_answer else "No answer found"
                    else:
                        # Non-tag patterns (like "Answer:") - apply cleaning heuristics
                        final_answer = self._extract_clean_final_answer(raw_answer)
                    break
            
            # Fallback strategies
            if not thinking and not final_answer:
                # Try to split on common patterns
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in ['think', 'step', 'reasoning']):
                        thinking = '\n'.join(lines[:i+3])  # Take a few lines
                        break
                
                # Look for final answer in last few lines
                for line in reversed(lines[-5:]):
                    if line.strip() and any(char.isdigit() or char.isalpha() for char in line):
                        final_answer = line.strip()
                        break
                
                # Ultimate fallback
                if not thinking:
                    thinking = response[:200] + "..." if len(response) > 200 else response
                if not final_answer:
                    final_answer = "Unable to parse answer"
            
            # Ensure we have both components
            if not thinking:
                thinking = "Reasoning not found in expected format"
            if not final_answer:
                final_answer = "Answer not found in expected format"
                
        except Exception as e:
            logger.debug(f"   ⚠️ Error parsing response: {e}")
            thinking = f"Parse error: {str(e)}"
            final_answer = "Parse failed"
        
        # Debug info for troubleshooting
        if not thinking or not final_answer or "not found" in thinking.lower() or "not found" in final_answer.lower():
            logger.debug(f"   🔍 Debug - FULL RAW RESPONSE:")
            logger.debug(f"{'='*80}")
            print(response)
            logger.debug(f"{'='*80}")
            logger.debug(f"   🔍 Debug - Extracted thinking: {thinking[:50]}...")
            logger.debug(f"   🔍 Debug - Extracted answer: {final_answer}")
            # Show where <answer> tag is if present
            answer_pos = response.find('<answer>')
            if answer_pos != -1:
                logger.debug(f"   🔍 Debug - <answer> tag found at position {answer_pos}")
                logger.debug(f"   🔍 Debug - Content around tag: ...{response[max(0,answer_pos-50):answer_pos+150]}...")
        
        return thinking, final_answer

    def _extract_clean_final_answer(self, raw_answer: str) -> str:
        """
        Extract the clean final answer from potentially messy answer content.

        Args:
            raw_answer: Raw content extracted from <answer> tags

        Returns:
            Clean final answer
        """
        if not raw_answer or not raw_answer.strip():
            return "No answer found"

        # Clean whitespace
        raw_answer = raw_answer.strip()

        # Strategy 1: Look for explicit "Final answer:", "Answer:", etc.
        final_answer_patterns = [
            r'Final answer:\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'Result:\s*(.+?)(?:\n|$)',
            r'Solution:\s*(.+?)(?:\n|$)',
            r'The answer is:\s*(.+?)(?:\n|$)',
            r'Therefore:\s*(.+?)(?:\n|$)'
        ]

        import re
        for pattern in final_answer_patterns:
            match = re.search(pattern, raw_answer, re.IGNORECASE | re.MULTILINE)
            if match:
                clean_answer = match.group(1).strip()
                # Remove any trailing punctuation or extra text
                clean_answer = re.sub(r'[.!?]+\s*.*$', '', clean_answer)
                if clean_answer:
                    return clean_answer

        # Strategy 2: If it's a short answer (likely just the answer), return as-is
        if len(raw_answer) <= 50 and '\n' not in raw_answer:
            # Remove common prefixes if present
            clean_answer = re.sub(r'^(The answer is|Answer:|Final answer:|Result:|Solution:)\s*', '', raw_answer, flags=re.IGNORECASE)
            return clean_answer.strip()

        # Strategy 3: Look for the last line that contains meaningful content
        lines = [line.strip() for line in raw_answer.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            # Check if it looks like a final answer (short, no "Step" or "Let me")
            if (len(last_line) <= 50 and
                not any(keyword in last_line.lower() for keyword in ['step', 'let me', 'now', 'first', 'next', 'then', 'calculate'])):
                return last_line

        # Strategy 4: Look for isolated numbers/times/dates in the last few lines
        for line in reversed(lines[-3:]):  # Check last 3 lines
            # Look for patterns like numbers, times, dates, etc.
            patterns = [
                r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$',  # Numbers
                r'^\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?$',      # Times
                r'^\d{1,2}/\d{1,2}/\d{2,4}$',                   # Dates
                r'^[A-Za-z]+(?:\s+\d+)?$'                       # Short words/phrases
            ]

            for pattern in patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    return line.strip()

        # Strategy 5: Fallback - take first 50 characters and try to clean
        fallback = raw_answer[:50]
        # Remove step indicators and common thinking phrases
        fallback = re.sub(r'^(?:Step \d+:|Let me|Now|First|Then|Next).*?[.:]?\s*', '', fallback, flags=re.IGNORECASE)

        # If we cleaned it down to something reasonable, use it
        if len(fallback.strip()) > 0:
            return fallback.strip()

        # Ultimate fallback
        return raw_answer[:30].strip() + "..." if len(raw_answer) > 30 else raw_answer.strip()

    def generate_single_response(self, variation: VariationProblem) -> GeneratedResponse:
        """
        Generate response for a single problem variation.
        
        Args:
            variation: VariationProblem to solve
            
        Returns:
            GeneratedResponse with model output
        """
        start_time = time.time()
        
        try:
            # Create prompt
            system_prompt, user_prompt = self.create_chain_of_thought_prompt(
                variation.modified_problem,
                variation.domain
            )

            # Generate response using model client
            if hasattr(self.model_client, 'get_model_response'):
                # RITS-style client
                logger.debug(f"   🎯 Generating response for: {variation.variation_id}")
                responses = self.model_client.get_model_response(
                    [system_prompt],
                    [user_prompt],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                response = responses[0] if responses and len(responses) > 0 else ""
                logger.debug(f"   📥 Raw response length: {len(response)} chars")
            else:
                # Other client types
                logger.debug(f"   🎯 Generating response for: {variation.variation_id}")
                # For non-RITS clients, combine prompts
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = str(self.model_client.generate(
                    combined_prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                ))
                logger.debug(f"   📥 Raw response length: {len(response)} chars")
            
            # Validate response is not empty
            if not response or len(response.strip()) < 10:
                raise ValueError(f"Empty or too short response: '{response}'")
            
            # Parse response
            thinking, final_answer = self.parse_model_response(response)
            
            # Validate parsing succeeded
            if not thinking or not final_answer:
                logger.debug(f"   ⚠️ Parsing incomplete - thinking: {bool(thinking)}, answer: {bool(final_answer)}")
            
            logger.debug(f"   ✅ Successfully parsed response for {variation.variation_id}")
            
            generation_time = time.time() - start_time
            
            return GeneratedResponse(
                variation_id=variation.variation_id,
                problem=variation.modified_problem,
                model_response=response,
                thinking=thinking,
                final_answer=final_answer,
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            return GeneratedResponse(
                variation_id=variation.variation_id,
                problem=variation.modified_problem,
                model_response="",
                thinking="",
                final_answer="",
                generation_time=generation_time,
                success=False,
                error_message=str(e)
            )
    
    def generate_responses_parallel(self, variations: List[VariationProblem]) -> List[GeneratedResponse]:
        """
        Generate responses using batch processing with RITS client parallelization.
        This matches the pattern from universal_quality_judge_optimized.py.
        
        Args:
            variations: List of VariationProblem to process
            
        Returns:
            List of GeneratedResponse
        """
        # Use client-specific batch size
        if self.client_type == 'vllm':
            response_batch_size = self.batch_size
            logger.debug(f"🔄 Generating responses for {len(variations)} variations...")
            logger.debug(f"   Settings: batch_size={response_batch_size}, max_workers={self.max_workers}")
        else:
            response_batch_size = self.rits_batch_size
            logger.debug(f"🔄 Generating responses for {len(variations)} variations...")
            logger.debug(f"   Settings: batch_size={response_batch_size}, max_workers={self.max_workers}")

        all_responses = []

        # Process in batches (like universal_quality_judge_optimized.py)
        for i in range(0, len(variations), response_batch_size):
            batch = variations[i:i + response_batch_size]
            batch_num = i // response_batch_size + 1
            total_batches = (len(variations) + response_batch_size - 1) // response_batch_size
            
            logger.debug(f"  📤 Processing batch {batch_num}/{total_batches} ({len(batch)} variations)...")

            # Prepare prompts for this batch
            system_prompts = []
            user_prompts = []
            for var in batch:
                system_prompt, user_prompt = self.create_chain_of_thought_prompt(var.modified_problem, var.domain)
                system_prompts.append(system_prompt)
                user_prompts.append(user_prompt)
            
            start_time = time.time()
            
            try:
                # Send batch to RITS client (it uses max_workers internally)
                if hasattr(self.model_client, 'get_model_response'):
                    responses = self.model_client.get_model_response(
                        system_prompts=system_prompts,
                        user_prompts=user_prompts,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature
                    )

                else:
                    # Fallback for other client types
                    responses = []
                    for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                        # For non-RITS clients, combine prompts
                        combined_prompt = f"{sys_prompt}\n\n{usr_prompt}"
                        response = str(self.model_client.generate(
                            combined_prompt,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature
                        ))
                        responses.append(response)
                
                batch_time = time.time() - start_time
                logger.debug(f"    📥 Batch {batch_num} completed in {batch_time:.2f}s")
                
                # Process responses for this batch
                batch_responses = []
                for variation, response in zip(batch, responses):
                    try:
                        thinking, final_answer = self.parse_model_response(response)
                        
                        result_response = GeneratedResponse(
                            variation_id=variation.variation_id,
                            problem=variation.modified_problem,
                            model_response=response,
                            thinking=thinking,
                            final_answer=final_answer,
                            generation_time=batch_time / len(batch),  # Approximate per-item time
                            success=True
                        )
                        
                    except Exception as e:
                        result_response = GeneratedResponse(
                            variation_id=variation.variation_id,
                            problem=variation.modified_problem,
                            model_response="",
                            thinking="",
                            final_answer="",
                            generation_time=0.0,
                            success=False,
                            error_message=f"Parse error: {e}"
                        )
                    
                    batch_responses.append(result_response)
                
                all_responses.extend(batch_responses)
                
                # Show progress
                successful = sum(1 for r in batch_responses if r.success)
                logger.debug(f"    ✅ {successful}/{len(batch)} successful in batch {batch_num}")
                
                # Progress update every 100 processed
                if len(all_responses) % 100 == 0:
                    logger.debug(f"  📊 Progress: {len(all_responses)}/{len(variations)} variations processed")
                
            except Exception as e:
                logger.debug(f"    ❌ Batch {batch_num} failed: {e}")
                # Add error responses for failed batch
                for variation in batch:
                    error_response = GeneratedResponse(
                        variation_id=variation.variation_id,
                        problem=variation.modified_problem,
                        model_response="",
                        thinking="",
                        final_answer="",
                        generation_time=0.0,
                        success=False,
                        error_message=f"Batch error: {e}"
                    )
                    all_responses.append(error_response)
        
        total_successful = sum(1 for r in all_responses if r.success)
        logger.debug(f"🎉 Total: {total_successful}/{len(all_responses)} successful responses")
        
        return all_responses
    
    def _generate_single_rits_response(self, variation: VariationProblem) -> GeneratedResponse:
        """Generate response for single variation using direct RITS call (matching working script pattern)"""
        start_time = time.time()

        try:
            # Create prompt
            system_prompt, user_prompt = self.create_chain_of_thought_prompt(variation.modified_problem, variation.domain)

            # Make individual RITS API call (like working script)
            if hasattr(self.model_client, 'call_rits_llm'):
                # Use direct RITS call method if available
                response = self.model_client.call_rits_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
            elif hasattr(self.model_client, 'get_single_response'):
                # Use single response method
                response = self.model_client.get_single_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
            else:
                # Fallback: use batch method with single item
                responses = self.model_client.get_model_response(
                    [system_prompt],
                    [user_prompt],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                response = responses[0] if responses else ""
            
            # Parse response
            thinking, final_answer = self.parse_model_response(response)
            
            generation_time = time.time() - start_time
            
            return GeneratedResponse(
                variation_id=variation.variation_id,
                problem=variation.modified_problem,
                model_response=response,
                thinking=thinking,
                final_answer=final_answer,
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            return GeneratedResponse(
                variation_id=variation.variation_id,
                problem=variation.modified_problem,
                model_response="",
                thinking="",
                final_answer="",
                generation_time=generation_time,
                success=False,
                error_message=str(e)
            )
    
    def generate_responses_batch(self, variations: List[VariationProblem]) -> List[GeneratedResponse]:
        """
        Generate responses in batches (for VLLM client).
        
        Args:
            variations: List of VariationProblem to process
            
        Returns:
            List of GeneratedResponse
        """
        logger.debug(f"🔄 Generating responses for {len(variations)} variations in batches of {self.batch_size}...")
        
        all_responses = []
        
        # Process in batches
        batch_range = range(0, len(variations), self.batch_size)
        if not getattr(self, 'verbose', False):
            batch_range = tqdm(batch_range, desc='Processing variations', unit='batch', total=len(list(range(0, len(variations), self.batch_size))))
        for i in batch_range:
            batch = variations[i:i + self.batch_size]
            batch_start = time.time()
            
            logger.debug(f"  Processing batch {i//self.batch_size + 1}/{(len(variations)-1)//self.batch_size + 1} ({len(batch)} problems)")

            # Create prompts for batch
            prompt_pairs = [
                self.create_chain_of_thought_prompt(var.modified_problem, var.domain)
                for var in batch
            ]
            system_prompts = [sys_prompt for sys_prompt, _ in prompt_pairs]
            user_prompts = [usr_prompt for _, usr_prompt in prompt_pairs]
            
            try:
                # Generate batch responses
                if hasattr(self.model_client, 'get_model_response'):
                    # RITS-style batch processing
                    responses = self.model_client.get_model_response(
                        system_prompts,
                        user_prompts,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature
                    )
                else:
                    # Other batch processing
                    responses = []
                    for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                        # For non-RITS clients, combine prompts
                        combined_prompt = f"{sys_prompt}\n\n{usr_prompt}"
                        response = str(self.model_client.generate(
                            combined_prompt,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature
                        ))
                        responses.append(response)
                
                # Process batch results
                batch_responses = []
                for j, (variation, response) in enumerate(zip(batch, responses)):
                    try:
                        thinking, final_answer = self.parse_model_response(response)
                        
                        batch_responses.append(GeneratedResponse(
                            variation_id=variation.variation_id,
                            problem=variation.modified_problem,
                            model_response=response,
                            thinking=thinking,
                            final_answer=final_answer,
                            generation_time=(time.time() - batch_start) / len(batch),  # Approximate per-item time
                            success=True
                        ))
                        
                    except Exception as e:
                        batch_responses.append(GeneratedResponse(
                            variation_id=variation.variation_id,
                            problem=variation.modified_problem,
                            model_response="",
                            thinking="",
                            final_answer="",
                            generation_time=0.0,
                            success=False,
                            error_message=str(e)
                        ))
                
                all_responses.extend(batch_responses)
                
                batch_time = time.time() - batch_start
                successful = sum(1 for r in batch_responses if r.success)
                logger.debug(f"    ✅ {successful}/{len(batch)} successful ({batch_time:.2f}s total)")
                
            except Exception as e:
                logger.debug(f"    ❌ Batch failed: {e}")
                # Add error responses for entire batch
                for variation in batch:
                    all_responses.append(GeneratedResponse(
                        variation_id=variation.variation_id,
                        problem=variation.modified_problem,
                        model_response="",
                        thinking="",
                        final_answer="",
                        generation_time=0.0,
                        success=False,
                        error_message=f"Batch error: {e}"
                    ))
        
        return all_responses
    
    def process_variations(self, variations: List[VariationProblem]) -> List[GeneratedResponse]:
        """
        Process variations using the appropriate method based on client type.
        
        Args:
            variations: List of VariationProblem to process
            
        Returns:
            List of GeneratedResponse
        """
        start_time = time.time()
        
        if self.client_type == 'rits':
            responses = self.generate_responses_parallel(variations)
        else:  # vllm or other
            responses = self.generate_responses_batch(variations)
        
        # Update statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in responses if r.success)
        
        self.stats['total_processed'] += len(variations)
        self.stats['successful_generations'] += successful
        self.stats['failed_generations'] += (len(variations) - successful)
        self.stats['total_time'] += total_time
        self.stats['avg_time_per_problem'] = self.stats['total_time'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0.0
        
        return responses
    
    def process_variations_streaming(self, variation_generator_engine, problem_text: str, output_filepath: Optional[str] = None) -> Iterator[GeneratedResponse]:
        """
        Process variations by streaming directly from the variation engine as they're generated.
        
        Args:
            variation_generator_engine: ComprehensiveVariationEngine instance
            problem_text: Original problem text to generate variations for
            output_filepath: Optional file to save responses as they're generated
            
        Yields:
            GeneratedResponse objects as they're generated
        """
        logger.debug(f"🔄 Streaming variations and generating responses for problem...")
        print(f"   Problem: {problem_text[:100]}..." if len(problem_text) > 100 else f"   Problem: {problem_text}")
        
        output_file = open(output_filepath, 'w') if output_filepath else None
        
        try:
            # Generate variations from the engine
            variations = variation_generator_engine.generate_comprehensive_variations(problem_text)
            logger.debug(f"📊 Generated {len(variations)} variations, processing responses...")
            
            variation_objects = []
            for i, variation_dict in enumerate(variations):
                # Convert variation dict to VariationProblem object
                variation = VariationProblem(
                    variation_id=f"var_{i+1}",
                    original_problem=problem_text,
                    modified_problem=variation_dict.get('modified_problem', ''),
                    transformation_type=variation_dict.get('transformation_type', ''),
                    original_component=variation_dict.get('original_component', ''),
                    new_component=variation_dict.get('new_component', ''),
                    domain=variation_dict.get('domain'),
                    metadata=variation_dict
                )
                variation_objects.append(variation)
            
            # Process in streaming fashion
            if self.client_type == 'rits':
                # For RITS, process all in parallel but yield results as they complete
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_variation = {
                        executor.submit(self.generate_single_response, var): var 
                        for var in variation_objects
                    }
                    
                    for i, future in enumerate(as_completed(future_to_variation), 1):
                        try:
                            response = future.result()
                            
                            # Update stats
                            self.stats['total_processed'] += 1
                            if response.success:
                                self.stats['successful_generations'] += 1
                            else:
                                self.stats['failed_generations'] += 1
                            self.stats['total_time'] += response.generation_time
                            self.stats['avg_time_per_problem'] = self.stats['total_time'] / self.stats['total_processed']
                            
                            # Save to file if specified
                            if output_file:
                                data = {
                                    'variation_id': response.variation_id,
                                    'problem': response.problem,
                                    'model_response': response.model_response,
                                    'thinking': response.thinking,
                                    'final_answer': response.final_answer,
                                    'generation_time': response.generation_time,
                                    'success': response.success,
                                    'error_message': response.error_message,
                                    'timestamp': time.time()
                                }
                                output_file.write(json.dumps(data) + '\n')
                                output_file.flush()
                            
                            status = "✅" if response.success else "❌"
                            logger.debug(f"  {status} {i}/{len(variation_objects)} - {response.variation_id} ({response.generation_time:.2f}s)")
                            
                            yield response
                            
                        except Exception as e:
                            variation = future_to_variation[future]
                            logger.debug(f"  ❌ {i}/{len(variation_objects)} - {variation.variation_id} failed: {e}")
                            
                            error_response = GeneratedResponse(
                                variation_id=variation.variation_id,
                                problem=variation.modified_problem,
                                model_response="",
                                thinking="",
                                final_answer="",
                                generation_time=0.0,
                                success=False,
                                error_message=str(e)
                            )
                            
                            self.stats['total_processed'] += 1
                            self.stats['failed_generations'] += 1
                            
                            if output_file:
                                data = {
                                    'variation_id': error_response.variation_id,
                                    'problem': error_response.problem,
                                    'model_response': error_response.model_response,
                                    'thinking': error_response.thinking,
                                    'final_answer': error_response.final_answer,
                                    'generation_time': error_response.generation_time,
                                    'success': error_response.success,
                                    'error_message': error_response.error_message,
                                    'timestamp': time.time()
                                }
                                output_file.write(json.dumps(data) + '\n')
                                output_file.flush()
                            
                            yield error_response
            else:
                # For VLLM, process in batches but still yield individual responses
                for i in range(0, len(variation_objects), self.batch_size):
                    batch = variation_objects[i:i + self.batch_size]
                    batch_responses = self.generate_responses_batch(batch)
                    
                    for response in batch_responses:
                        # Save to file if specified
                        if output_file:
                            data = {
                                'variation_id': response.variation_id,
                                'problem': response.problem,
                                'model_response': response.model_response,
                                'thinking': response.thinking,
                                'final_answer': response.final_answer,
                                'generation_time': response.generation_time,
                                'success': response.success,
                                'error_message': response.error_message,
                                'timestamp': time.time()
                            }
                            output_file.write(json.dumps(data) + '\n')
                            output_file.flush()
                        
                        yield response
                        
        finally:
            if output_file:
                output_file.close()
                if output_filepath:
                    logger.debug(f"💾 Saved streaming responses to: {output_filepath}")
        
        logger.debug(f"\n📊 Batch Summary:")
        logger.debug(f"   Total: {len(variations)} variations")
        logger.debug(f"   Successful: {successful} ({successful/len(variations)*100:.1f}%)")
        logger.debug(f"   Failed: {len(variations) - successful}")
        logger.debug(f"   Time: {total_time:.2f}s ({total_time/len(variations):.2f}s per variation)")
        
        return responses
    
    def load_variations_from_json(self, filepath: str) -> Iterator[VariationProblem]:
        """
        Load variations from JSON file (unified_variation_engine format).
        
        Args:
            filepath: Path to JSON file
            
        Yields:
            VariationProblem objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            # Handle list of variations (unified engine format)
            if isinstance(data, list):
                for i, variation in enumerate(data):
                    yield VariationProblem(
                        variation_id=variation.get('variation_id', f'var_{i}'),
                        original_problem=variation.get('original_problem', ''),
                        modified_problem=variation.get('modified_problem', ''),
                        transformation_type=variation.get('transformation_type', ''),
                        original_component=variation.get('original_component', ''),
                        new_component=variation.get('new_component', ''),
                        expected_answer=None,  # We don't have expected answers yet
                        domain=variation.get('debugging_capability', ''),
                        metadata={
                            'problem_id': variation.get('problem_id', ''),
                            'debugging_capability': variation.get('debugging_capability', ''),
                            'generation_method': variation.get('generation_method', ''),
                            'detection_method': variation.get('detection_method', ''),
                            'domains_involved': variation.get('domains_involved', []),
                            'combination_size': variation.get('combination_size', 1),
                            'cross_domain': variation.get('cross_domain', False)
                        }
                    )
            else:
                # Handle single variation object
                yield VariationProblem(
                    variation_id=data.get('variation_id', 'single_var'),
                    original_problem=data.get('original_problem', ''),
                    modified_problem=data.get('modified_problem', ''),
                    transformation_type=data.get('transformation_type', ''),
                    original_component=data.get('original_component', ''),
                    new_component=data.get('new_component', ''),
                    expected_answer=None,
                    domain=data.get('debugging_capability', ''),
                    metadata=data
                )
    
    def load_variations_from_jsonl(self, filepath: str) -> Iterator[VariationProblem]:
        """
        Load variations from JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Yields:
            VariationProblem objects
        """
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Extract variation data (handle different formats)
                    if 'all_variations' in data:
                        # Format from comprehensive variation engine
                        for i, var in enumerate(data['all_variations']):
                            yield VariationProblem(
                                variation_id=f"{data.get('id', line_num)}_{i}",
                                original_problem=data.get('problem', ''),
                                modified_problem=var.get('modified_problem', ''),
                                transformation_type=var.get('transformation_type', ''),
                                original_component=var.get('original_component', ''),
                                new_component=var.get('new_component', ''),
                                expected_answer=data.get('answer', ''),
                                domain=data.get('domain', ''),
                                metadata=var
                            )
                    else:
                        # Simple format
                        yield VariationProblem(
                            variation_id=f"{data.get('id', line_num)}",
                            original_problem=data.get('original_problem', ''),
                            modified_problem=data.get('modified_problem', data.get('problem', '')),
                            transformation_type=data.get('transformation_type', ''),
                            original_component=data.get('original_component', ''),
                            new_component=data.get('new_component', ''),
                            expected_answer=data.get('answer', ''),
                            domain=data.get('domain', ''),
                            metadata=data
                        )
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"❌ Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"❌ Error processing line {line_num}: {e}")
                    continue
    
    def save_responses_to_jsonl(self, responses: List[GeneratedResponse], output_filepath: str):
        """
        Save generated responses to JSONL file.
        
        Args:
            responses: List of GeneratedResponse to save
            output_filepath: Output file path
        """
        with open(output_filepath, 'w') as f:
            for response in responses:
                data = {
                    'variation_id': response.variation_id,
                    'problem': response.problem,
                    'model_response': response.model_response,
                    'thinking': response.thinking,
                    'final_answer': response.final_answer,
                    'generation_time': response.generation_time,
                    'success': response.success,
                    'error_message': response.error_message,
                    'timestamp': time.time()
                }
                f.write(json.dumps(data) + '\n')
        
        logger.debug(f"💾 Saved {len(responses)} responses to: {output_filepath}")
    
    def print_statistics(self):
        """Print generation statistics"""
        stats = self.stats
        logger.debug(f"\n📈 Generation Statistics:")
        logger.debug(f"   Total processed: {stats['total_processed']}")
        logger.debug(f"   Successful: {stats['successful_generations']} ({stats['successful_generations']/stats['total_processed']*100:.1f}%)")
        logger.debug(f"   Failed: {stats['failed_generations']} ({stats['failed_generations']/stats['total_processed']*100:.1f}%)")
        logger.debug(f"   Total time: {stats['total_time']:.2f}s")
        logger.debug(f"   Avg time per problem: {stats['avg_time_per_problem']:.2f}s")

def stream_variations_and_generate_answers(problem_text: str, 
                                         output_filepath: str,
                                         client_type: str = 'rits',
                                         model_name: str = 'microsoft/Phi-4-reasoning',
                                         max_workers: int = 4,
                                         batch_size: int = 8,
                                         with_model: bool = True) -> List[GeneratedResponse]:
    """
    Convenient function to stream variations directly from the variation engine 
    and generate answers in one call.
    
    Args:
        problem_text: Original problem text
        output_filepath: File to save responses 
        client_type: 'rits' or 'vllm'
        model_name: Model for generation
        max_workers: Parallel workers for RITS
        batch_size: Batch size for VLLM  
        with_model: Whether to use model client for variations
        
    Returns:
        List of GeneratedResponse
    """
    from benchdrift.pipeline.comprehensive_variation_engine import ComprehensiveVariationEngine
    
    # Initialize variation engine
    model_client_for_variations = None
    if with_model:
        try:
            model_client_for_variations = create_model_client_for_variations()
        except Exception as e:
            logger.debug(f"⚠️  Warning: Could not initialize model client for variations: {e}")
            logger.debug("   Proceeding with deterministic variations only")
    
    variation_engine = ComprehensiveVariationEngine(model_client=model_client_for_variations)
    
    # Initialize answer generator
    answer_generator = VariationAnswerGenerator(
        client_type=client_type,
        model_name=model_name,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Stream and process
    responses = []
    for response in answer_generator.process_variations_streaming(
        variation_engine, 
        problem_text, 
        output_filepath
    ):
        responses.append(response)
    
    # Print final statistics
    answer_generator.print_statistics()
    
    return responses

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Generate answers for problem variations")
    
    # Mode selection
    parser.add_argument('--mode', choices=['stream', 'offline'], default='offline',
                       help='Processing mode: stream from variation engine or offline from JSON/JSONL (default: offline)')
    
    # Input/Output for offline mode
    parser.add_argument('--input', help='Input JSON/JSONL file with variations (for offline mode)')
    parser.add_argument('--output', required=True, help='Output JSONL file for responses')
    
    # Input for streaming mode
    parser.add_argument('--problem', help='Problem text to generate variations for (for stream mode)')
    
    # Model configuration
    parser.add_argument('--client-type', choices=['rits', 'vllm'], default='rits',
                       help='Model client type (default: rits)')
    parser.add_argument('--model', default='microsoft/Phi-4-reasoning',
                       help='Model name for generation')
    
    # Processing configuration
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers for non-RITS clients (default: 4)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for VLLM processing (default: 8)')
    parser.add_argument('--rits-batch-size', type=int, default=100,
                       help='Batch size for RITS processing (default: 100)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of variations to process')
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=1024,
                       help='Maximum tokens to generate (default: 1024)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for generation (default: 0.1)')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'stream':
        if not args.problem:
            logger.debug("❌ --problem is required for stream mode")
            return 1
    else:  # offline mode
        if not args.input:
            logger.debug("❌ --input is required for offline mode")
            return 1
    
    if args.mode == 'stream':
        # Streaming mode: generate variations and answers directly
        logger.debug(f"🚀 Streaming mode: generating variations and answers...")
        
        try:
            responses = stream_variations_and_generate_answers(
                problem_text=args.problem,
                output_filepath=args.output,
                client_type=args.client_type,
                model_name=args.model,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                with_model=True  # Always use model client for variations in streaming mode
            )
            
            logger.debug(f"\n🎉 Streaming processing complete! Generated {len(responses)} responses")
            
        except Exception as e:
            logger.debug(f"❌ Streaming failed: {e}")
            return 1
            
    else:
        # Offline mode: process from JSONL file
        logger.debug(f"📂 Offline mode: processing from JSONL file...")
        
        # Initialize generator
        try:
            generator = VariationAnswerGenerator(
                client_type=args.client_type,
                model_name=args.model,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                rits_batch_size=args.rits_batch_size,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
        except Exception as e:
            logger.debug(f"❌ Failed to initialize generator: {e}")
            return 1
        
        # Load and process variations
        logger.debug(f"📂 Loading variations from: {args.input}")
        
        # Detect file format and use appropriate loader
        input_path = Path(args.input)
        if input_path.suffix.lower() == '.json':
            logger.debug(f"   📄 Detected JSON format (unified_variation_engine)")
            variation_loader = generator.load_variations_from_json(args.input)
        elif input_path.suffix.lower() == '.jsonl':
            logger.debug(f"   📄 Detected JSONL format (legacy)")
            variation_loader = generator.load_variations_from_jsonl(args.input)
        else:
            logger.debug(f"❌ Unsupported file format: {input_path.suffix}")
            logger.debug(f"   Supported formats: .json (unified engine), .jsonl (legacy)")
            return 1
        
        variations = []
        for i, variation in enumerate(variation_loader):
            variations.append(variation)
            if args.max_samples and i + 1 >= args.max_samples:
                break
        
        if not variations:
            logger.debug("❌ No variations found in input file")
            return 1
        
        logger.debug(f"📊 Loaded {len(variations)} variations")
        
        # Generate responses
        responses = generator.process_variations(variations)
        
        # Save results
        generator.save_responses_to_jsonl(responses, args.output)
        
        # Print final statistics
        generator.print_statistics()
        
        logger.debug(f"\n🎉 Offline processing complete!")
    
    return 0

if __name__ == "__main__":
    exit(main())
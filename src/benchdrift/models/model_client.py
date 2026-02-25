"""
Unified Model Client Interface
Supports both VLLM and Gemini models for temporal reasoning synthesis
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import time
import logging
import requests
import concurrent.futures
from threading import Lock

# Logging setup

logger = logging.getLogger('BenchDrift.ModelClient')

# VLLM imports - completely lazy loaded to avoid CUDA initialization
VLLM_AVAILABLE = None  # Will be checked when actually needed

def _check_vllm_availability():
    """Check if VLLM is available - only called when actually needed."""
    global VLLM_AVAILABLE
    if VLLM_AVAILABLE is None:
        try:
            import vllm
            import transformers
            VLLM_AVAILABLE = True
        except ImportError:
            logger.warning("VLLM not available. Only RITS and Gemini clients will work.")
            VLLM_AVAILABLE = False
    return VLLM_AVAILABLE

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    logger.info("Google GenerativeAI not available. Only VLLM client will work.")
    GEMINI_AVAILABLE = False

# RITS configuration - always available (uses requests)
RITS_AVAILABLE = True
RITS_API_KEY = os.getenv("RITS_API_KEY", "")  # Set RITS_API_KEY env var to use RITS backend
RITS_MODELS = {
    "llama-3-1-8b": {"endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-8b-instruct", "name": "meta-llama/Llama-3.1-8B-Instruct"},
    "granite-3-1-8b": {"endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct", "name": "ibm-granite/granite-3.1-8b-instruct"},
    "granite-3-0-8b": {"endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct", "name": "ibm-granite/granite-3.0-8b-instruct"},
    "granite-3-3-8b": {"endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-instruct", "name": "ibm-granite/granite-3.3-8b-instruct"},
    "phi-4": {"endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4", "name": "microsoft/phi-4"},
    "phi-4-reasoning": {"endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/phi-4-reasoning", "name": "microsoft/Phi-4-reasoning"},
    "franconia": {"endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-4-0-small-franconia", "name": "ibm-granite/granite-4.0-small-prerelease-franconia.r250523a"},
    "openoss": {"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b","name":"openai/gpt-oss-120b"},
    "llama_3_3_70b": {"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct" , "name":"meta-llama/llama-3-3-70b-instruct"},
    "gpt_oss_20b": {"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-20b", "name":"openai/gpt-oss-20b"},
    "mistral_small_3_2_instruct": {"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mistral-small-3-2-24b-2506", "name":"mistralai/Mistral-Small-3.2-24B-Instruct-2506"},
    "qwen_3_8b": {"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen3-8b","name":"Qwen/Qwen3-8B"},
    "granite-4-small":{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-4-h-small","name":"ibm-granite/granite-4.0-h-small"},
    "qwen-3-8b":{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen3-8b","name":"Qwen/Qwen3-8B"},
    "granite-4-micro":{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-4-micro","name":"ibm-granite/granite-4.0-micro"},
    "granite-4-8b":{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-4-8b","name":"ibm-granite/granite-4.0-8b"}


}

RITS_CHAT_ENDPOINT = "/v1/chat/completions"

# Ollama configuration - always available (uses requests)
OLLAMA_AVAILABLE = True
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_ENDPOINT = "/v1/chat/completions"
RITS_HEADERS = {
    "accept": "application/json",
    "RITS_API_KEY": RITS_API_KEY,
    "Content-Type": "application/json"
}

# Groq configuration - cloud API (uses requests, needs GROQ_API_KEY)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "qwen-qwq-32b",
]

file_lock = Lock()

class BaseModelClient(ABC):
    """Abstract base class for model clients"""
    
    @abstractmethod
    def get_model_response(self, system_prompts: List[str], user_prompts: List[str], 
                          max_new_tokens: int = 1000, temperature: float = 0.1, 
                          **kwargs) -> List[str]:
        """Generate responses for given prompts"""
        pass
    
    @abstractmethod
    def get_single_response(self, system_prompt: str, user_prompt: str, 
                           max_new_tokens: int = 1000, temperature: float = 0.1,
                           **kwargs) -> str:
        """Generate single response"""
        pass

class VLLMClient(BaseModelClient):
    """Client for VLLM model inference"""

    def __init__(self, model_name: str = "microsoft/phi-4", max_model_len: int = 8192):
        if not _check_vllm_availability():
            raise ImportError("VLLM is not available. Please install vllm and transformers.")

        # Lazy import VLLM components to avoid CUDA initialization until needed
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        # Store for later use
        self.SamplingParams = SamplingParams

        # Fix VLLM multiprocessing issue
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        logger.debug(f"Initializing VLLMClient with model: {model_name}, max_model_len: {max_model_len}")
        try:
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.llm = LLM(
                model=model_name,
                disable_log_stats=True,
                tensor_parallel_size=1,
                max_model_len=max_model_len,
                gpu_memory_utilization=0.90,
                trust_remote_code=True
            )
            logger.debug("VLLMClient initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize VLLMClient: {str(e)}")
            raise

    def get_model_response(self, system_prompts: List[str], user_prompts: List[str], 
                          max_new_tokens: int = 1000, temperature: float = 0.1, 
                          top_k: int = 40, top_p: float = 0.9, **kwargs) -> List[str]:
        """Generate responses for batch of prompts"""
        logger.debug(f"Generating responses for {len(user_prompts)} prompts...")
        
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens
        )
        
        # Prepare messages
        messages_list = [
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
            for sys, usr in zip(system_prompts, user_prompts)
        ]
        
        # Apply chat template
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
            for messages in messages_list
        ]
        
        # Generate responses
        outputs = self.llm.generate(texts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def get_single_response(self, system_prompt: str, user_prompt: str,
                           max_new_tokens: int = 1000, temperature: float = 0.1,
                           **kwargs) -> str:
        """Generate single response"""
        responses = self.get_model_response(
            [system_prompt], [user_prompt], max_new_tokens, temperature, **kwargs
        )
        return responses[0]

    def get_model_response_with_logprobs(self, system_prompts: List[str], user_prompts: List[str],
                                        max_new_tokens: int = 50, temperature: float = 0.1,
                                        **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses with logprobs for batch of prompts.

        Returns:
            List of dicts with 'text' and 'logprobs' keys
        """
        logger.debug(f"Generating responses with logprobs for {len(user_prompts)} prompts...")

        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=kwargs.get('top_p', 0.9),
            top_k=kwargs.get('top_k', 40),
            max_tokens=max_new_tokens,
            logprobs=1,  # Return logprobs for top-1 token
            prompt_logprobs=0  # Don't need prompt logprobs
        )

        # Prepare messages
        messages_list = [
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
            for sys, usr in zip(system_prompts, user_prompts)
        ]

        # Apply chat template
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]

        # Generate with logprobs
        outputs = self.llm.generate(texts, sampling_params)

        # Extract results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            logprobs_data = output.outputs[0].logprobs if output.outputs[0].logprobs else []

            # Extract logprob values
            logprob_values = []
            for token_logprobs in logprobs_data:
                if token_logprobs:
                    for token_id, logprob_obj in token_logprobs.items():
                        logprob_values.append(logprob_obj.logprob)
                        break  # Only need first (selected token)

            results.append({
                'text': generated_text,
                'logprobs': logprob_values
            })

        return results






# from google import genai
# from google.genai.errors import APIError 

# # --- Configuration (Simulated for Structure) ---
# GEMINI_MODELS = {
#     "flash": {"name": "gemini-2.5-flash"},
#     "pro": {"name": "gemini-2.5-pro"},
# }
# # -----------------------------------------------

# # Placeholder for the base client/model structure
# class BaseModelClient:
#     pass

# class GeminiClient(BaseModelClient):
#     """
#     Client for Gemini API model inference using the official google-genai SDK,
#     supporting concurrent generation via ThreadPoolExecutor and structural 
#     compliance with the logprobs output format.
#     """
    
#     def __init__(self, model_name: str = "flash", max_workers: int = 50, **kwargs):
#         if model_name not in GEMINI_MODELS:
#             raise ValueError(
#                 f"Unknown Gemini model alias: {model_name}. "
#                 f"Available: {list(GEMINI_MODELS.keys())}"
#             )
            
#         self.model_config = GEMINI_MODELS[model_name]
#         self.model_name = self.model_config['name']
#         self.max_workers = max_workers

#         try:
#             self.client = genai.Client()
#         except Exception as e:
#             logging.error(f"Failed to initialize Gemini Client: {e}")
#             raise RuntimeError("Gemini API key (GEMINI_API_KEY) not found or invalid.")
        
#         print(f"GeminiClient initialized with model: {self.model_name}")
#         print(f"Concurrent worker limit set to: {self.max_workers}")

#     # --- Core Single API Call ---
#     def call_gemini_llm(self, system_prompt: str, user_prompt: str, 
#                         max_output_tokens: int = 1000, temperature: float = 0.5, 
#                         max_retries: int = 3, **kwargs) -> str:
#         """Call Gemini LLM API with retries using the SDK"""
        
#         contents = [
#             genai.types.Content(role="user", parts=[genai.types.Part.from_text(user_prompt)])
#         ]

#         config = genai.types.GenerateContentConfig(
#             system_instruction=system_prompt, 
#             temperature=temperature,
#             max_output_tokens=max_output_tokens,
#             top_p=kwargs.get('top_p', 0.85), 
#             top_k=kwargs.get('top_k', 30),
#             stop_sequences=kwargs.get('stop_sequences', None)
#         )
        
#         for attempt in range(1, max_retries + 1):
#             try:
#                 response = self.client.models.generate_content(
#                     model=self.model_name,
#                     contents=contents,
#                     config=config
#                 )
                
#                 if not response.text:
#                     # Treat content blocking/empty response as an error after retries
#                     logging.warning(f"Response blocked (attempt {attempt}/{max_retries})")
#                     if attempt < max_retries:
#                         time.sleep(2 ** attempt)
#                         continue
#                     return "[ERROR: Response blocked or empty text]"

#                 return response.text
            
#             except APIError as e:
#                 logging.error(f"Gemini API call failed (attempt {attempt}/{max_retries}): {e}")
#                 if attempt < max_retries:
#                     time.sleep(2 ** attempt)
#                 continue
#             except Exception as e:
#                 logging.error(f"An unexpected error occurred (attempt {attempt}/{max_retries}): {e}")
#                 if attempt < max_retries:
#                     time.sleep(2 ** attempt)
#                 continue
        
#         logging.error(f"Gemini API call failed after {max_retries} attempts")
#         return "[ERROR: API call failed after multiple attempts]"

#     # --- Simple Response Methods ---
#     def get_single_response(self, system_prompt: str, user_prompt: str,
#                            max_output_tokens: int = 1000, temperature: float = 0.1,
#                            **kwargs) -> str:
#         """Generate single response by calling the core LLM method."""
#         return self.call_gemini_llm(
#             system_prompt, user_prompt, max_output_tokens, temperature, **kwargs
#         )

#     def test_api_connection(self, timeout: int = 10) -> bool:
#         """Test Gemini API connection with a simple request."""
#         # Logic remains the same, using call_gemini_llm
#         # ... (omitted for brevity)
#         return True # Assuming the logic runs successfully

#     # --- Concurrent Execution (Batching) ---
#     def get_model_response(self, system_prompts: List[str], user_prompts: List[str], 
#                           max_output_tokens: int = 1000, temperature: float = 0.1, 
#                           **kwargs) -> List[str]:
#         """
#         Generate responses using concurrent execution.
#         """
#         if len(system_prompts) != len(user_prompts):
#             raise ValueError("Number of system and user prompts must match")
        
#         print(f"Generating responses for {len(user_prompts)} prompts using Gemini concurrently...")
#         responses = []
        
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
            
#             for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
#                 future = executor.submit(
#                     self.call_gemini_llm, 
#                     sys_prompt, 
#                     user_prompt, 
#                     max_output_tokens, 
#                     temperature, 
#                     **kwargs
#                 )
#                 futures.append(future)
            
#             for future in futures:
#                 try:
#                     response = future.result()
#                     responses.append(response if response is not None else "")
#                 except Exception as e:
#                     logging.error(f"Gemini concurrent call failed: {e}")
#                     responses.append("[ERROR: Concurrent call failed]")
        
#         return responses

#     # --- Concurrent Execution with Logprobs Placeholder (Restored) ---
#     def get_model_response_with_logprobs(self, system_prompts: List[str], user_prompts: List[str],
#                                         max_output_tokens: int = 50, temperature: float = 0.1,
#                                         **kwargs) -> List[Dict[str, Any]]:
#         """
#         Generate responses for a batch of prompts, setting logprobs to an empty list
#         or placeholder since the native Gemini API does not return them.
#         """
#         if len(system_prompts) != len(user_prompts):
#             raise ValueError("Number of system and user prompts must match")
            
#         print(f"Generating responses with logprobs placeholder for {len(user_prompts)} prompts using Gemini...")

#         def get_single_with_logprobs_placeholder(sys_prompt, usr_prompt):
#             """Get response using call_gemini_llm and wrap it in the required dict format."""
#             # Use the core LLM call to get the text response
#             text = self.call_gemini_llm(
#                 sys_prompt, usr_prompt, max_output_tokens, temperature, **kwargs
#             )
            
#             # Since logprobs are not available, return the required structure with an empty list
#             return {'text': text, 'logprobs': []} 

#         results = []
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
            
#             for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
#                 future = executor.submit(
#                     get_single_with_logprobs_placeholder, sys_prompt, usr_prompt
#                 )
#                 futures.append(future)

#             # Maintain original order
#             for future in futures:
#                 try:
#                     result = future.result()
#                     results.append(result)
#                 except Exception





class RITSClient(BaseModelClient):
    """Client for RITS API model inference"""

    def __init__(self, model_name: str = "franconia", max_workers: int = 50, max_new_tokens: int = 1000):
        self.model_name = model_name
        self.max_workers = max_workers
        self.default_max_new_tokens = max_new_tokens  # Store default max_new_tokens

        if model_name not in RITS_MODELS:
            raise ValueError(f"Unknown RITS model: {model_name}. Available: {list(RITS_MODELS.keys())}")

        self.model_config = RITS_MODELS[model_name]
        self.headers = RITS_HEADERS

        logger.debug(f"RITSClient initialized with model: {self.model_config['name']}, max_new_tokens: {max_new_tokens}")

    def test_api_connection(self, timeout: int = 10) -> bool:
        """
        Test RITS API connection with a simple request.
        Returns True if successful, False otherwise.
        """
        logger.debug(f"\n🔌 Testing RITS API connection for {self.model_name}...")

        test_system = "You are a helpful assistant."
        test_user = "Say 'OK' if you can hear me."

        try:
            start_time = time.time()
            response = self.call_rits_llm(
                test_system,
                test_user,
                max_new_tokens=10,
                temperature=0.0,
                max_retries=2
            )
            elapsed = time.time() - start_time

            if response and not response.startswith("[ERROR"):
                logger.debug(f"✅ RITS API test successful!")
                logger.debug(f"   Model: {self.model_config['name']}")
                logger.debug(f"   Response time: {elapsed:.2f}s")
                logger.debug(f"   Test response: {response[:100]}")
                return True
            else:
                logger.debug(f"❌ RITS API test failed!")
                logger.debug(f"   Error response: {response}")
                return False

        except Exception as e:
            logger.debug(f"❌ RITS API test failed with exception!")
            logger.debug(f"   Error: {e}")
            return False

    def call_rits_llm(self, system_prompt: str, user_prompt: str, max_new_tokens: int = None,
                      temperature: float = 0.5, max_retries: int = 3, **kwargs) -> str:
        """Call RITS LLM API with retries"""
        # Use instance default if not provided
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        body = {
            "messages": messages,
            "model": self.model_config['name'],
            "temperature": temperature,
            "stop_sequences": [""],
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1,
            "top_p": kwargs.get('top_p', 0.85),
            "top_k": kwargs.get('top_k', 30),
            "do_sample": True
        }
        
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    url=f"{self.model_config['endpoint']}{RITS_CHAT_ENDPOINT}",
                    json=body,
                    headers=self.headers,
                    timeout=60
                )
                
                if response.status_code != 200:
                    logging.error(f"RITS API call failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return f"[ERROR: API call failed with status {response.status_code}]"
                
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' not in content_type:
                    logging.error(f"Non-JSON response received: Content-Type={content_type}")
                    return "[ERROR: Non-JSON response received]"
                
                chat_response = response.json()
                if 'choices' not in chat_response or not chat_response['choices']:
                    logging.error(f"Invalid response format: No 'choices' field. Response={chat_response}")
                    return "[ERROR: Invalid response format]"
                
                model_output = chat_response["choices"][0]["message"]["content"]
                return model_output
            
            except requests.exceptions.RequestException as e:
                logging.error(f"RITS API call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue
        
        logging.error(f"RITS API call failed after {max_retries} attempts")
        return "[ERROR: API call failed after multiple attempts]"
    
    def get_model_response(self, system_prompts: List[str], user_prompts: List[str],
                          max_new_tokens: int = None, temperature: float = 0.1,
                          **kwargs) -> List[str]:
        """Generate responses using concurrent execution for RITS"""
        # Use instance default if not provided
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        if len(system_prompts) != len(user_prompts):
            raise ValueError("Number of system and user prompts must match")

        logger.debug(f"Generating responses for {len(user_prompts)} prompts using RITS...")
        responses = []

        # Use concurrent execution for RITS
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
                future = executor.submit(
                    self.call_rits_llm, sys_prompt, user_prompt, max_new_tokens, temperature, **kwargs
                )
                futures.append(future)
            
            # FIXED: Maintain original order by iterating through futures list directly
            for future in futures:
                try:
                    response = future.result()
                    responses.append(response if response is not None else "")
                except Exception as e:
                    logging.error(f"RITS concurrent call failed: {e}")
                    responses.append("[ERROR: Concurrent call failed]")
        
        return responses
    
    def get_single_response(self, system_prompt: str, user_prompt: str,
                           max_new_tokens: int = None, temperature: float = 0.1,
                           **kwargs) -> str:
        """Generate single response"""
        # Use instance default if not provided
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        return self.call_rits_llm(system_prompt, user_prompt, max_new_tokens, temperature, **kwargs)

    def get_model_response_with_logprobs(self, system_prompts: List[str], user_prompts: List[str],
                                        max_new_tokens: int = 50, temperature: float = 0.1,
                                        **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses with logprobs for batch of prompts (RITS API).

        Returns:
            List of dicts with 'text' and 'logprobs' keys
        """
        logger.debug(f"Generating responses with logprobs for {len(user_prompts)} prompts using RITS...")

        def get_single_with_logprobs(sys_prompt, usr_prompt):
            """Get response with logprobs for single prompt."""
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt}
            ]

            body = {
                "messages": messages,
                "model": self.model_config['name'],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "logprobs": True,  # Enable logprobs
                "top_logprobs": 1,  # Return top-1 logprobs
                "top_p": kwargs.get('top_p', 0.85),
                "top_k": kwargs.get('top_k', 30)
            }

            for attempt in range(3):
                try:
                    response = requests.post(
                        url=f"{self.model_config['endpoint']}{RITS_CHAT_ENDPOINT}",
                        json=body,
                        headers=self.headers,
                        timeout=60
                    )
                    response.raise_for_status()

                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        text = choice['message']['content']

                        # Extract logprobs if available
                        logprobs = []
                        if 'logprobs' in choice and choice['logprobs']:
                            content_logprobs = choice['logprobs'].get('content', [])
                            logprobs = [token['logprob'] for token in content_logprobs if 'logprob' in token]

                        return {'text': text, 'logprobs': logprobs}
                    else:
                        return {'text': '', 'logprobs': []}

                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        logging.error(f"Failed to get logprobs: {e}")
                        return {'text': '', 'logprobs': []}

            return {'text': '', 'logprobs': []}

        # Use concurrent execution
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                future = executor.submit(get_single_with_logprobs, sys_prompt, usr_prompt)
                futures.append(future)

            # Maintain original order
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Failed to get logprobs: {e}")
                    results.append({'text': '', 'logprobs': []})

        return results

def _empty_logit_result():
    """Return an empty logit result dict for error fallbacks."""
    return {
        'text': '',
        'logit_stats': {
            'mean_logprob': 0.0, 'min_logprob': 0.0, 'max_logprob': 0.0,
            'num_tokens': 0, 'mean_top1_prob': 0.0,
            'mean_entropy': 0.0, 'max_entropy': 0.0,
            'first_token_logprob': 0.0, 'first_token_prob': 0.0,
        },
        'first_token_top_k': [],
        'selected_logprobs': [],
        'token_logprobs': [],
    }


class OllamaClient(BaseModelClient):
    """Client for Ollama local model inference (OpenAI-compatible API)"""

    def __init__(self, model_name: str = "qwen2.5:7b", max_workers: int = 4, max_new_tokens: int = 1000):
        self.model_name = model_name
        self.max_workers = max_workers
        self.default_max_new_tokens = max_new_tokens
        self.base_url = OLLAMA_BASE_URL

        logger.debug(f"OllamaClient initialized with model: {self.model_name}, max_new_tokens: {max_new_tokens}")

    def test_api_connection(self, timeout: int = 10) -> bool:
        """Test Ollama API connection — checks server is running and model is available."""
        logger.debug(f"\n🔌 Testing Ollama API connection for {self.model_name}...")

        # Step 1: Check if Ollama server is running
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Cannot connect to Ollama at {self.base_url}. Is `ollama serve` running?")
            return False
        except Exception as e:
            logger.error(f"❌ Ollama server check failed: {e}")
            return False

        # Step 2: Check if the requested model is pulled
        try:
            data = resp.json()
            available_models = [m.get('name', '') for m in data.get('models', [])]
            # Ollama model names may or may not have a tag; match flexibly
            model_found = any(
                self.model_name == m or self.model_name == m.split(':')[0] or m.startswith(self.model_name)
                for m in available_models
            )
            if not model_found:
                logger.error(
                    f"❌ Model '{self.model_name}' not found in Ollama. "
                    f"Available: {available_models}. Run `ollama pull {self.model_name}` first."
                )
                return False
        except Exception as e:
            logger.warning(f"⚠️ Could not verify model availability: {e}")

        # Step 3: Quick inference test
        try:
            start_time = time.time()
            response = self.call_ollama_llm(
                "You are a helpful assistant.",
                "Say 'OK' if you can hear me.",
                max_new_tokens=50,
                temperature=0.0,
                max_retries=2
            )
            elapsed = time.time() - start_time

            if response and not response.startswith("[ERROR"):
                logger.debug(f"✅ Ollama API test successful!")
                logger.debug(f"   Model: {self.model_name}")
                logger.debug(f"   Response time: {elapsed:.2f}s")
                logger.debug(f"   Test response: {response[:100]}")
                return True
            else:
                logger.error(f"❌ Ollama API test failed! Response: {response}")
                return False
        except Exception as e:
            logger.error(f"❌ Ollama API test failed with exception: {e}")
            return False

    def call_native(self, system_prompt: str, user_prompt: str, max_new_tokens: int = None,
                     temperature: float = 0.5, think: bool = False, timeout: int = 0,
                     max_retries: int = 3, return_reasoning: bool = False, **kwargs):
        """Call Ollama via native /api/chat endpoint. Supports think parameter for thinking models.

        Args:
            return_reasoning: If True and think=True, returns dict {"content": str, "thinking": str}
                              instead of just the content string. This lets callers show CoT.
        Returns:
            str if return_reasoning=False, dict if return_reasoning=True
        """
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        tout = timeout or 120
        model_lower = self.model_name.lower()

        # Build messages — granite3.3 uses control role for thinking
        messages = []
        is_granite = "granite" in model_lower
        if think and is_granite:
            messages.append({"role": "control", "content": "thinking"})
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
                "top_p": kwargs.get("top_p", 0.85),
            },
            "stream": False,
        }
        # Explicitly control thinking — qwen3 enables thinking by default when field is absent
        if not is_granite:
            body["think"] = think
        empty = {"content": "", "thinking": ""} if return_reasoning else ""
        think_stripped = False
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat", json=body, timeout=tout,
                )
                if response.status_code == 400 and think and not think_stripped:
                    # Model doesn't support thinking — strip think params and retry
                    logging.warning(f"Model {self.model_name} returned 400 with think enabled, retrying without think")
                    body.pop("think", None)
                    # Remove control role message for granite fallback
                    body["messages"] = [m for m in body["messages"] if m.get("role") != "control"]
                    think_stripped = True
                    continue
                if response.status_code != 200:
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    err = f"[ERROR: API call failed with status {response.status_code}]"
                    return {"content": err, "thinking": ""} if return_reasoning else err
                data = response.json()
                msg = data.get("message", {})
                content = msg.get("content", "").strip()
                if return_reasoning:
                    thinking = msg.get("thinking", "").strip()
                    return {"content": content, "thinking": thinking}
                return content
            except requests.exceptions.RequestException as e:
                logging.error(f"Ollama native call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue
        err = "[ERROR: Ollama native call failed after multiple attempts]"
        return {"content": err, "thinking": ""} if return_reasoning else err

    def call_ollama_llm(self, system_prompt: str, user_prompt: str, max_new_tokens: int = None,
                        temperature: float = 0.5, max_retries: int = 3, **kwargs) -> str:
        """Call Ollama via OpenAI-compatible chat completions endpoint with retries."""
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": kwargs.get('top_p', 0.85),
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    url=f"{self.base_url}{OLLAMA_CHAT_ENDPOINT}",
                    json=body,
                    timeout=120  # Longer timeout for local inference
                )

                if response.status_code != 200:
                    logging.error(f"Ollama API call failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return f"[ERROR: API call failed with status {response.status_code}]"

                content_type = response.headers.get('Content-Type', '')
                if 'application/json' not in content_type:
                    logging.error(f"Non-JSON response received: Content-Type={content_type}")
                    return "[ERROR: Non-JSON response received]"

                chat_response = response.json()
                if 'choices' not in chat_response or not chat_response['choices']:
                    logging.error(f"Invalid response format: No 'choices' field. Response={chat_response}")
                    return "[ERROR: Invalid response format]"

                message = chat_response["choices"][0]["message"]
                model_output = message.get("content", "")
                # Handle thinking models (e.g., qwen3) that put output in 'reasoning' when content is empty
                if not model_output and message.get("reasoning"):
                    model_output = message["reasoning"]
                return model_output

            except requests.exceptions.RequestException as e:
                logging.error(f"Ollama API call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

        logging.error(f"Ollama API call failed after {max_retries} attempts")
        return "[ERROR: API call failed after multiple attempts]"

    def get_model_response(self, system_prompts: List[str], user_prompts: List[str],
                           max_new_tokens: int = None, temperature: float = 0.1,
                           **kwargs) -> List[str]:
        """Generate responses using concurrent execution via native /api/chat endpoint."""
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        if len(system_prompts) != len(user_prompts):
            raise ValueError("Number of system and user prompts must match")

        logger.debug(f"Generating responses for {len(user_prompts)} prompts using Ollama native API...")
        responses = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
                future = executor.submit(
                    self.call_native, sys_prompt, user_prompt,
                    max_new_tokens=max_new_tokens, temperature=temperature, **kwargs
                )
                futures.append(future)

            for future in futures:
                try:
                    response = future.result()
                    responses.append(response if response is not None else "")
                except Exception as e:
                    logging.error(f"Ollama concurrent call failed: {e}")
                    responses.append("[ERROR: Concurrent call failed]")

        return responses

    def get_single_response(self, system_prompt: str, user_prompt: str,
                            max_new_tokens: int = None, temperature: float = 0.1,
                            **kwargs) -> str:
        """Generate single response via native /api/chat endpoint."""
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        return self.call_native(system_prompt, user_prompt,
                                max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)

    def get_model_response_with_logprobs(self, system_prompts: List[str], user_prompts: List[str],
                                         max_new_tokens: int = 50, temperature: float = 0.1,
                                         **kwargs) -> List[Dict[str, Any]]:
        """Generate responses with logprobs for batch of prompts (Ollama OpenAI-compat API)."""
        logger.debug(f"Generating responses with logprobs for {len(user_prompts)} prompts using Ollama...")

        def get_single_with_logprobs(sys_prompt, usr_prompt):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt}
            ]

            body = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "logprobs": True,
                "top_logprobs": 1,
                "top_p": kwargs.get('top_p', 0.85),
            }

            for attempt in range(3):
                try:
                    response = requests.post(
                        url=f"{self.base_url}{OLLAMA_CHAT_ENDPOINT}",
                        json=body,
                        timeout=120
                    )
                    response.raise_for_status()

                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]
                        text = choice['message']['content']

                        logprobs = []
                        if 'logprobs' in choice and choice['logprobs']:
                            content_logprobs = choice['logprobs'].get('content', [])
                            logprobs = [token['logprob'] for token in content_logprobs if 'logprob' in token]

                        return {'text': text, 'logprobs': logprobs}
                    else:
                        return {'text': '', 'logprobs': []}

                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        logging.error(f"Failed to get logprobs from Ollama: {e}")
                        return {'text': '', 'logprobs': []}

            return {'text': '', 'logprobs': []}

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                future = executor.submit(get_single_with_logprobs, sys_prompt, usr_prompt)
                futures.append(future)

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Failed to get logprobs from Ollama: {e}")
                    results.append({'text': '', 'logprobs': []})

        return results


class OllamaClientWithLogits(OllamaClient):
    """
    Ollama Client with enhanced logit collection for drift analysis.
    Mirrors VLLMClientWithLogits but uses Ollama's OpenAI-compatible API.

    Collects:
    - Top-k logprobs at each position (default k=5)
    - Summary statistics: mean, min, entropy, top1_prob
    - First token details
    """

    def __init__(self, model_name: str = "qwen2.5:7b", max_workers: int = 4,
                 max_new_tokens: int = 1000, logprobs_k: int = 5):
        super().__init__(model_name, max_workers, max_new_tokens)
        self.logprobs_k = logprobs_k
        logger.debug(f"OllamaClientWithLogits initialized with logprobs_k={logprobs_k}")

    def get_model_response_with_logits(self, system_prompts: List[str], user_prompts: List[str],
                                        max_new_tokens: int = 1000, temperature: float = 0.1,
                                        **kwargs) -> List[Dict[str, Any]]:
        """Generate responses with comprehensive logit statistics via Ollama."""
        import math

        logger.debug(f"Generating responses with logits for {len(user_prompts)} prompts using Ollama...")

        def get_single_with_logits(sys_prompt, usr_prompt):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt}
            ]

            body = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "logprobs": True,
                "top_logprobs": self.logprobs_k,
                "top_p": kwargs.get('top_p', 0.85),
            }

            for attempt in range(3):
                try:
                    response = requests.post(
                        url=f"{self.base_url}{OLLAMA_CHAT_ENDPOINT}",
                        json=body,
                        timeout=120
                    )
                    response.raise_for_status()

                    data = response.json()
                    if 'choices' not in data or not data['choices']:
                        return _empty_logit_result()

                    choice = data['choices'][0]
                    text = choice['message']['content']

                    # Parse OpenAI-format logprobs
                    selected_logprobs = []
                    all_entropies = []
                    all_top1_probs = []
                    first_token_top_k = []
                    token_logprobs_full = []

                    content_logprobs = []
                    if 'logprobs' in choice and choice['logprobs']:
                        content_logprobs = choice['logprobs'].get('content', [])

                    for pos_idx, token_data in enumerate(content_logprobs):
                        token_text = token_data.get('token', '')
                        token_logprob = token_data.get('logprob', 0.0)
                        top_logprobs_list = token_data.get('top_logprobs', [])

                        selected_logprobs.append(token_logprob)
                        top1_prob = math.exp(token_logprob)
                        all_top1_probs.append(top1_prob)

                        # Build position tokens from top_logprobs
                        position_tokens = []
                        for tlp in top_logprobs_list:
                            lp = tlp.get('logprob', 0.0)
                            position_tokens.append({
                                'token': tlp.get('token', ''),
                                'logprob': lp,
                                'prob': math.exp(lp)
                            })

                        # Compute entropy
                        if position_tokens:
                            probs = [pt['prob'] for pt in position_tokens]
                            prob_sum = sum(probs)
                            if prob_sum > 0:
                                normalized = [p / prob_sum for p in probs]
                                entropy = -sum(p * math.log(p + 1e-10) for p in normalized if p > 0)
                                all_entropies.append(entropy)

                        token_logprobs_full.append({
                            'position': pos_idx,
                            'selected_token': token_text,
                            'selected_logprob': token_logprob,
                            'top_k': position_tokens[:self.logprobs_k]
                        })

                        if pos_idx == 0:
                            first_token_top_k = position_tokens[:self.logprobs_k]

                    # Compute summary statistics
                    logit_stats = {}
                    if selected_logprobs:
                        logit_stats['mean_logprob'] = sum(selected_logprobs) / len(selected_logprobs)
                        logit_stats['min_logprob'] = min(selected_logprobs)
                        logit_stats['max_logprob'] = max(selected_logprobs)
                        logit_stats['num_tokens'] = len(selected_logprobs)
                    else:
                        logit_stats['mean_logprob'] = 0.0
                        logit_stats['min_logprob'] = 0.0
                        logit_stats['max_logprob'] = 0.0
                        logit_stats['num_tokens'] = 0

                    logit_stats['mean_top1_prob'] = (
                        sum(all_top1_probs) / len(all_top1_probs) if all_top1_probs else 0.0
                    )
                    logit_stats['mean_entropy'] = (
                        sum(all_entropies) / len(all_entropies) if all_entropies else 0.0
                    )
                    logit_stats['max_entropy'] = max(all_entropies) if all_entropies else 0.0

                    if first_token_top_k:
                        logit_stats['first_token_logprob'] = first_token_top_k[0]['logprob']
                        logit_stats['first_token_prob'] = first_token_top_k[0]['prob']
                    else:
                        logit_stats['first_token_logprob'] = 0.0
                        logit_stats['first_token_prob'] = 0.0

                    return {
                        'text': text,
                        'logit_stats': logit_stats,
                        'first_token_top_k': first_token_top_k,
                        'selected_logprobs': selected_logprobs,
                        'token_logprobs': token_logprobs_full,
                    }

                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        logging.error(f"Failed to get logits from Ollama: {e}")
                        return _empty_logit_result()

            return _empty_logit_result()

        # Concurrent execution
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
                future = executor.submit(get_single_with_logits, sys_prompt, usr_prompt)
                futures.append(future)

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Failed to get logits from Ollama: {e}")
                    results.append(_empty_logit_result())

        return results

    def get_model_response(self, system_prompts: List[str], user_prompts: List[str],
                           max_new_tokens: int = None, temperature: float = 0.1,
                           **kwargs) -> List[str]:
        """Override to also collect logits (stores in instance variable for retrieval)."""
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        try:
            results = self.get_model_response_with_logits(
                system_prompts, user_prompts, max_new_tokens, temperature, **kwargs
            )
            self._last_logit_results = results
            return [r['text'] for r in results]
        except Exception as e:
            logger.warning(f"Logit collection failed, falling back to standard response: {e}")
            self._last_logit_results = None
            return super().get_model_response(system_prompts, user_prompts, max_new_tokens, temperature, **kwargs)

    def get_last_logit_results(self) -> Optional[List[Dict[str, Any]]]:
        """Retrieve the full logit results from the last get_model_response call."""
        return getattr(self, '_last_logit_results', None)


class GroqClient(BaseModelClient):
    """Client for Groq Cloud API inference (OpenAI-compatible)."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile",
                 max_workers: int = 10, max_new_tokens: int = 1000):
        self.model_name = model_name
        self.max_workers = max_workers
        self.default_max_new_tokens = max_new_tokens
        self.api_key = GROQ_API_KEY
        self.api_url = GROQ_API_URL

        if not self.api_key:
            logger.warning("GROQ_API_KEY not set. Export it: export GROQ_API_KEY=gsk_...")
        logger.debug(f"GroqClient initialized with model: {self.model_name}")

    def test_api_connection(self, timeout: int = 10) -> bool:
        """Test Groq API connection."""
        if not self.api_key:
            logger.error("❌ GROQ_API_KEY not set.")
            return False
        try:
            start_time = time.time()
            response = self.call_groq_llm(
                "You are a helpful assistant.",
                "Say 'OK' if you can hear me.",
                max_new_tokens=10, temperature=0.0, max_retries=2,
            )
            elapsed = time.time() - start_time
            if response and not response.startswith("[ERROR"):
                logger.debug(f"✅ Groq API test successful! ({elapsed:.2f}s)")
                return True
            logger.error(f"❌ Groq API test failed: {response}")
            return False
        except Exception as e:
            logger.error(f"❌ Groq API test failed: {e}")
            return False

    @staticmethod
    def get_available_models() -> List[str]:
        """Fetch available models from Groq API, fallback to hardcoded list."""
        if not GROQ_API_KEY:
            return GROQ_DEFAULT_MODELS
        try:
            resp = requests.get(
                f"{GROQ_API_URL}/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=5,
            )
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])
                      if m.get("active", True)]
            return sorted(models) if models else GROQ_DEFAULT_MODELS
        except Exception:
            return GROQ_DEFAULT_MODELS

    def call_groq_llm(self, system_prompt: str, user_prompt: str,
                      max_new_tokens: int = None, temperature: float = 0.5,
                      max_retries: int = 3, **kwargs) -> str:
        """Call Groq API with retries."""
        if not self.api_key:
            return "[ERROR: GROQ_API_KEY not set]"
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.85),
            "stream": False,
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=kwargs.get("timeout", 60),
                )
                if response.status_code != 200:
                    logging.error(f"Groq API call failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                        continue
                    return f"[ERROR: API call failed with status {response.status_code}]"

                chat_response = response.json()
                if "choices" not in chat_response or not chat_response["choices"]:
                    return "[ERROR: Invalid response format]"
                return chat_response["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                logging.error(f"Groq API call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

        return "[ERROR: Groq API call failed after multiple attempts]"

    def get_model_response(self, system_prompts: List[str], user_prompts: List[str],
                           max_new_tokens: int = None, temperature: float = 0.1,
                           **kwargs) -> List[str]:
        """Generate responses for batch of prompts using concurrent execution."""
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        if len(system_prompts) != len(user_prompts):
            raise ValueError("Number of system and user prompts must match")

        logger.debug(f"Generating responses for {len(user_prompts)} prompts using Groq...")
        responses = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
                future = executor.submit(
                    self.call_groq_llm, sys_prompt, user_prompt,
                    max_new_tokens, temperature, **kwargs,
                )
                futures.append(future)

            for future in futures:
                try:
                    response = future.result()
                    responses.append(response if response is not None else "")
                except Exception as e:
                    logging.error(f"Groq concurrent call failed: {e}")
                    responses.append("[ERROR: Concurrent call failed]")

        return responses

    def get_single_response(self, system_prompt: str, user_prompt: str,
                            max_new_tokens: int = None, temperature: float = 0.1,
                            **kwargs) -> str:
        """Generate single response."""
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        return self.call_groq_llm(system_prompt, user_prompt, max_new_tokens, temperature, **kwargs)


class ModelClientFactory:
    """Factory for creating model clients"""
    
    @staticmethod
    def create_client(client_type: str, model_name: Optional[str] = None, **kwargs) -> BaseModelClient:
        """
        Create a model client of specified type
        
        Args:
            client_type: 'vllm', 'gemini', or 'rits'
            model_name: Model name (optional, uses defaults)
            **kwargs: Additional arguments for client initialization
            
        Returns:
            BaseModelClient instance
        """
        client_type = client_type.lower()
        
        if client_type == 'vllm':
            if not _check_vllm_availability():
                raise ImportError("VLLM is not available")
            default_model = "microsoft/phi-4"
            return VLLMClient(model_name or default_model, **kwargs)

        elif client_type == 'vllm_logits':
            if not _check_vllm_availability():
                raise ImportError("VLLM is not available")
            default_model = "microsoft/phi-4"
            logprobs_k = kwargs.pop('logprobs_k', 5)
            return VLLMClientWithLogits(model_name or default_model, logprobs_k=logprobs_k, **kwargs)
            
        elif client_type == 'gemini':
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "Gemini client is not available. The GeminiClient class is currently "
                    "disabled. To use Gemini, uncomment and fix the GeminiClient class in "
                    "model_client.py, or use --client-type vllm or --client-type rits instead."
                )
            default_model = "gemini-1.5-flash"
            return GeminiClient(model_name or default_model, **kwargs)
            
        elif client_type == 'rits':
            if not RITS_AVAILABLE:
                raise ImportError("RITS is not available")
            default_model = "franconia"
            max_workers = kwargs.get('max_workers', 50)
            max_new_tokens = kwargs.get('max_new_tokens', 1000)
            return RITSClient(model_name or default_model, max_workers=max_workers, max_new_tokens=max_new_tokens)

        elif client_type == 'ollama':
            default_model = "qwen2.5:7b"
            max_workers = kwargs.get('max_workers', 4)
            max_new_tokens = kwargs.get('max_new_tokens', 1000)
            return OllamaClient(model_name or default_model, max_workers=max_workers, max_new_tokens=max_new_tokens)

        elif client_type == 'ollama_logits':
            default_model = "qwen2.5:7b"
            max_workers = kwargs.get('max_workers', 4)
            max_new_tokens = kwargs.get('max_new_tokens', 1000)
            logprobs_k = kwargs.pop('logprobs_k', 5)
            return OllamaClientWithLogits(model_name or default_model, max_workers=max_workers,
                                          max_new_tokens=max_new_tokens, logprobs_k=logprobs_k)

        elif client_type == 'groq':
            default_model = "llama-3.3-70b-versatile"
            max_workers = kwargs.get('max_workers', 10)
            max_new_tokens = kwargs.get('max_new_tokens', 1000)
            return GroqClient(model_name or default_model, max_workers=max_workers, max_new_tokens=max_new_tokens)

        else:
            raise ValueError(f"Unknown client type: {client_type}. Use 'vllm', 'gemini', 'rits', 'ollama', or 'groq'")
    
    @staticmethod
    def get_available_clients() -> List[str]:
        """Get list of available client types"""
        available = []
        if _check_vllm_availability():
            available.append('vllm')
        if GEMINI_AVAILABLE:
            available.append('gemini')
        if RITS_AVAILABLE:
            available.append('rits')
        if OLLAMA_AVAILABLE:
            available.append('ollama')
        if GROQ_API_KEY:
            available.append('groq')
        return available
    
    @staticmethod
    def get_rits_models() -> List[str]:
        """Get list of available RITS models"""
        return list(RITS_MODELS.keys())

# Convenience class for backward compatibility
class VLLMClient_samelength(VLLMClient):
    """Backward compatibility wrapper for existing code"""
    pass


class VLLMClientWithLogits(VLLMClient):
    """
    VLLM Client with enhanced logit collection for drift analysis.

    Use this client for: variation generation, validation, response generation.
    Do NOT use for: evaluation/judge (use regular VLLMClient instead).

    Collects:
    - Top-k logprobs at each position (default k=5)
    - Summary statistics: mean, min, entropy, top1_prob
    - First token details (often indicative of overall confidence)
    """

    def __init__(self, model_name: str = "microsoft/phi-4", max_model_len: int = 8192,
                 logprobs_k: int = 5):
        """
        Initialize with logprob collection settings.

        Args:
            model_name: HuggingFace model name
            max_model_len: Maximum context length
            logprobs_k: Number of top logprobs to collect per token (default 5)
        """
        super().__init__(model_name, max_model_len)
        self.logprobs_k = logprobs_k
        logger.debug(f"VLLMClientWithLogits initialized with logprobs_k={logprobs_k}")

    def get_model_response_with_logits(self, system_prompts: List[str], user_prompts: List[str],
                                       max_new_tokens: int = 1000, temperature: float = 0.1,
                                       **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses with comprehensive logit statistics.

        Returns:
            List of dicts with:
            - 'text': Generated text
            - 'logit_stats': Dict with summary statistics
            - 'first_token_top_k': Top-k alternatives at first position
            - 'selected_logprobs': Logprobs of selected tokens (full sequence)
            - 'token_logprobs': Full per-token top-k logprobs for entire sequence
        """
        import math

        logger.debug(f"Generating responses with logits for {len(user_prompts)} prompts...")

        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=kwargs.get('top_p', 0.9),
            top_k=kwargs.get('top_k', 40),
            max_tokens=max_new_tokens,
            logprobs=self.logprobs_k,  # Top-k logprobs per token
            prompt_logprobs=0  # Don't need prompt logprobs
        )

        # Prepare messages
        messages_list = [
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
            for sys, usr in zip(system_prompts, user_prompts)
        ]

        # Apply chat template
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]

        # Generate with logprobs
        outputs = self.llm.generate(texts, sampling_params)

        # Extract results with comprehensive statistics
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            logprobs_data = output.outputs[0].logprobs if output.outputs[0].logprobs else []

            # Extract selected token logprobs and compute statistics
            selected_logprobs = []
            all_entropies = []
            all_top1_probs = []
            first_token_top_k = []
            token_logprobs_full = []  # Full per-token top-k for entire sequence

            for pos_idx, token_logprobs in enumerate(logprobs_data):
                if not token_logprobs:
                    continue

                # Get all logprobs at this position (dict: token_id -> LogprobInfo)
                position_logprobs = []
                position_tokens = []

                for token_id, logprob_obj in token_logprobs.items():
                    position_logprobs.append(logprob_obj.logprob)
                    # Get token text if available
                    token_text = getattr(logprob_obj, 'decoded_token',
                                        self.tokenizer.decode([token_id]) if hasattr(self, 'tokenizer') else str(token_id))
                    position_tokens.append({
                        'token': token_text,
                        'logprob': logprob_obj.logprob,
                        'prob': math.exp(logprob_obj.logprob)
                    })

                # Sort by logprob (highest first) - these are the top-k alternatives
                position_tokens.sort(key=lambda x: x['logprob'], reverse=True)

                # Store full top-k for this position
                token_logprobs_full.append({
                    'position': pos_idx,
                    'selected_token': position_tokens[0]['token'] if position_tokens else '',
                    'selected_logprob': position_tokens[0]['logprob'] if position_tokens else 0.0,
                    'top_k': position_tokens[:self.logprobs_k]
                })

                # First token gets special treatment
                if pos_idx == 0:
                    first_token_top_k = position_tokens[:self.logprobs_k]

                # Selected token is the one with highest logprob (first in sorted list)
                if position_tokens:
                    selected_logprobs.append(position_tokens[0]['logprob'])
                    all_top1_probs.append(position_tokens[0]['prob'])

                    # Compute entropy at this position: -sum(p * log(p))
                    probs = [math.exp(lp) for lp in position_logprobs]
                    prob_sum = sum(probs)
                    if prob_sum > 0:
                        normalized_probs = [p / prob_sum for p in probs]
                        entropy = -sum(p * math.log(p + 1e-10) for p in normalized_probs if p > 0)
                        all_entropies.append(entropy)

            # Compute summary statistics
            logit_stats = {}
            if selected_logprobs:
                logit_stats['mean_logprob'] = sum(selected_logprobs) / len(selected_logprobs)
                logit_stats['min_logprob'] = min(selected_logprobs)
                logit_stats['max_logprob'] = max(selected_logprobs)
                logit_stats['num_tokens'] = len(selected_logprobs)
            else:
                logit_stats['mean_logprob'] = 0.0
                logit_stats['min_logprob'] = 0.0
                logit_stats['max_logprob'] = 0.0
                logit_stats['num_tokens'] = 0

            if all_top1_probs:
                logit_stats['mean_top1_prob'] = sum(all_top1_probs) / len(all_top1_probs)
            else:
                logit_stats['mean_top1_prob'] = 0.0

            if all_entropies:
                logit_stats['mean_entropy'] = sum(all_entropies) / len(all_entropies)
                logit_stats['max_entropy'] = max(all_entropies)
            else:
                logit_stats['mean_entropy'] = 0.0
                logit_stats['max_entropy'] = 0.0

            # First token stats (often predictive of overall confidence)
            if first_token_top_k:
                logit_stats['first_token_logprob'] = first_token_top_k[0]['logprob']
                logit_stats['first_token_prob'] = first_token_top_k[0]['prob']
            else:
                logit_stats['first_token_logprob'] = 0.0
                logit_stats['first_token_prob'] = 0.0

            results.append({
                'text': generated_text,
                'logit_stats': logit_stats,
                'first_token_top_k': first_token_top_k,
                'selected_logprobs': selected_logprobs,  # Full sequence for detailed analysis
                'token_logprobs': token_logprobs_full  # Full per-token top-k for entire sequence
            })

        return results

    def get_model_response(self, system_prompts: List[str], user_prompts: List[str],
                          max_new_tokens: int = 1000, temperature: float = 0.1,
                          **kwargs) -> List[str]:
        """
        Override to also collect logits (stores in instance variable for retrieval).
        Falls back to parent if logit collection fails.
        """
        try:
            # Get responses with logits
            results = self.get_model_response_with_logits(
                system_prompts, user_prompts, max_new_tokens, temperature, **kwargs
            )
            # Store for later retrieval
            self._last_logit_results = results
            # Return just the text for compatibility
            return [r['text'] for r in results]
        except Exception as e:
            logger.warning(f"Logit collection failed, falling back to standard response: {e}")
            self._last_logit_results = None
            return super().get_model_response(system_prompts, user_prompts, max_new_tokens, temperature, **kwargs)

    def get_last_logit_results(self) -> Optional[List[Dict[str, Any]]]:
        """Retrieve the full logit results from the last get_model_response call."""
        return getattr(self, '_last_logit_results', None)

def test_clients():
    """Test function to verify clients work"""
    available_clients = ModelClientFactory.get_available_clients()
    logger.info(f"Available clients: {available_clients}")
    
    test_system = "You are a helpful assistant."
    test_user = "What is 2+2?"
    
    for client_type in available_clients:
        try:
            logger.debug(f"\nTesting {client_type} client...")
            
            if client_type == 'vllm':
                client = ModelClientFactory.create_client('vllm', 'microsoft/phi-4')
            elif client_type == 'gemini':
                # Skip Gemini test if no API key available
                if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                    logger.info(f"Skipping Gemini test - no API key available")
                    continue
                client = ModelClientFactory.create_client('gemini')
            else:  # rits
                client = ModelClientFactory.create_client('rits', 'franconia')
            
            response = client.get_single_response(test_system, test_user, max_new_tokens=50)
            logger.debug(f"Response: {response[:100]}...")
            logger.debug(f"{client_type} client working correctly!")
            
        except Exception as e:
            logger.debug(f"Error testing {client_type} client: {e}")

if __name__ == "__main__":
    # Test the clients
    test_clients()
    
    # Example usage
    print("\n" + "="*50)
    logger.debug("EXAMPLE USAGE:")
    print("="*50)
    
    example_code = '''
# Using VLLM client
vllm_client = ModelClientFactory.create_client('vllm', 'microsoft/phi-4')
response = vllm_client.get_single_response(
    "You are a temporal reasoning expert.", 
    "What is 323 BC + 938 years?",
    temperature=0.1
)

# Using Gemini client (requires API key)
gemini_client = ModelClientFactory.create_client('gemini', api_key="your-api-key")
response = gemini_client.get_single_response(
    "You are a temporal reasoning expert.", 
    "What is 323 BC + 938 years?",
    temperature=0.1
)

# Using RITS client
rits_client = ModelClientFactory.create_client('rits', 'franconia')
response = rits_client.get_single_response(
    "You are a temporal reasoning expert.", 
    "What is 323 BC + 938 years?",
    temperature=0.1
)

# Batch processing
responses = client.get_model_response(
    ["System prompt 1", "System prompt 2"],
    ["User prompt 1", "User prompt 2"]
)
    '''
    
    print(example_code)
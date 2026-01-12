"""
Model Configuration Manager
Handles model-client mapping and configuration loading
"""

import yaml
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ModelConfigManager:
    """Manages model configuration and client selection"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load model configuration from YAML file"""
        if not self.config_path.exists():
            # Create default config if not exists
            default_config = self._get_default_config()
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            return default_config
            
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                return config
        except Exception as e:
            logger.debug(f"Warning: Error loading model config {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'default_client_preference': ['rits', 'vllm', 'gemini'],
            'model_client_mapping': {
                'granite-3-1-8b': 'rits',
                'granite-3-0-8b': 'rits', 
                'granite-3-3-8b': 'rits',
                'phi-4': 'rits',
                'phi-4-reasoning': 'rits',
                'microsoft/Phi-4-reasoning': 'rits',
                'franconia': 'rits',
                'llama_3_3_70b': 'rits',
                'microsoft/phi-4': 'vllm',
                'gemini-1.5-flash': 'gemini',
                'mistral_small_3_2_instruct': 'rits',
                'openoss': 'rits',
                'gpt_oss_20b':'rits',
                'qwen-3-8b':'rits',
                'granite-4-small':'rits'

                
            },
            'variation_generation': {
                'default_model': 'granite-3-1-8b',
                'default_client': 'rits',
                'fallback_models': ['phi-4-reasoning', 'franconia']
            },
            'evaluation': {
                'default_model': 'phi-4-reasoning',
                'default_client': 'rits', 
                'fallback_models': ['granite-3-1-8b', 'franconia']
            },
            'judge_model': {
                'default_model': 'phi-4-reasoning',
                'default_client': 'rits',
                'fallback_models': ['granite-3-1-8b']
            },
            'client_settings': {
                'rits': {
                    'max_workers': 50,
                    'max_retries': 3,
                    'timeout': 30,
                    'default_temperature': 0.1,
                    'default_max_tokens': 1000
                },
                'vllm': {
                    'tensor_parallel_size': 1,
                    'max_model_len': 8192,
                    'gpu_memory_utilization': 0.90,
                    'trust_remote_code': True,
                    'default_temperature': 0.1,
                    'default_max_tokens': 1000
                }
            }
        }
    
    def get_client_for_model(self, model_name: str) -> str:
        """Get the appropriate client type for a given model"""
        # Check explicit mapping first
        mapping = self.config.get('model_client_mapping', {})
        if model_name in mapping:
            return mapping[model_name]
        
        # Check if model name contains hints
        model_lower = model_name.lower()
        if any(hint in model_lower for hint in ['granite', 'phi', 'llama', 'franconia']):
            return 'rits'
        elif 'microsoft/' in model_name or 'meta-llama/' in model_name or '/' in model_name:
            return 'vllm'  
        elif 'gemini' in model_lower:
            return 'gemini'
            
        # Default to first preference
        preferences = self.config.get('default_client_preference', ['rits'])
        return preferences[0]
    
    def get_model_for_task(self, task: str) -> Tuple[str, str]:
        """
        Get model name and client for a specific task
        
        Args:
            task: 'variation_generation', 'evaluation', or 'judge_model'
            
        Returns:
            Tuple of (model_name, client_type)
        """
        task_config = self.config.get(task, {})
        model_name = task_config.get('default_model', 'granite-3-1-8b')
        client_type = task_config.get('default_client')
        
        # If no explicit client, determine from model
        if not client_type:
            client_type = self.get_client_for_model(model_name)
            
        return model_name, client_type
    
    def get_fallback_models(self, task: str) -> List[str]:
        """Get fallback models for a task"""
        task_config = self.config.get(task, {})
        return task_config.get('fallback_models', [])
    
    def get_client_settings(self, client_type: str) -> Dict:
        """Get configuration settings for a client type"""
        client_settings = self.config.get('client_settings', {})
        return client_settings.get(client_type, {})
    
    def validate_model_client_combination(self, model_name: str, client_type: str) -> bool:
        """Validate if a model-client combination is supported"""
        expected_client = self.get_client_for_model(model_name)
        return expected_client == client_type
    
    def get_recommended_model_client(self, 
                                   preferred_model: Optional[str] = None, 
                                   preferred_client: Optional[str] = None,
                                   task: str = 'variation_generation') -> Tuple[str, str]:
        """
        Get recommended model-client combination based on preferences
        
        Args:
            preferred_model: User's preferred model (optional)
            preferred_client: User's preferred client (optional)  
            task: Task type for default selection
            
        Returns:
            Tuple of (model_name, client_type)
        """
        # If both specified, use them as-is (respect user's explicit choice)
        if preferred_model and preferred_client:
            return preferred_model, preferred_client
        
        # If only model specified
        if preferred_model:
            client_type = self.get_client_for_model(preferred_model)
            return preferred_model, client_type
            
        # If only client specified, find compatible model
        if preferred_client:
            task_model, task_client = self.get_model_for_task(task)
            if task_client == preferred_client:
                return task_model, preferred_client
            else:
                # Find first model that uses this client
                mapping = self.config.get('model_client_mapping', {})
                for model, client in mapping.items():
                    if client == preferred_client:
                        return model, preferred_client
                # If no match, use task default but warn
                logger.debug(f"Warning: No models configured for {preferred_client}, using task default")
                return self.get_model_for_task(task)
        
        # Neither specified, use task defaults
        return self.get_model_for_task(task)
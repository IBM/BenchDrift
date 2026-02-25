#!/usr/bin/env python3
"""
Quick VLLM Client Test - Run this FIRST on remote server before full experiments
Tests that VLLMClient works correctly with minimal overhead (takes ~2-5 minutes)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("="*70)
print("VLLM CLIENT QUICK TEST")
print("="*70)

# Test 1: Check if VLLM is available
print("\n✓ Test 1: Checking VLLM availability...")
try:
    import vllm
    import transformers
    print("  ✅ VLLM and transformers installed")
except ImportError as e:
    print(f"  ❌ VLLM not available: {e}")
    print("  Install with: pip install vllm transformers")
    sys.exit(1)

# Test 2: Import VLLMClient
print("\n✓ Test 2: Importing VLLMClient...")
try:
    from benchdrift.models.model_client import VLLMClient, ModelClientFactory
    print("  ✅ VLLMClient imported successfully")
except Exception as e:
    print(f"  ❌ Failed to import: {e}")
    sys.exit(1)

# Test 3: Create client (loads model into GPU)
print("\n✓ Test 3: Creating VLLMClient with phi-4...")
print("  (This downloads model if not cached, may take a few minutes)")
try:
    client = VLLMClient(model_name="microsoft/phi-4", max_model_len=2048)
    print("  ✅ VLLMClient created, model loaded into GPU")
except Exception as e:
    print(f"  ❌ Failed to create client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Single response
print("\n✓ Test 4: Generating single response...")
try:
    response = client.get_single_response(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2 + 2?",
        max_new_tokens=50,
        temperature=0.1
    )
    print(f"  ✅ Response: {response[:100]}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 5: Batch responses (key for performance)
print("\n✓ Test 5: Testing batch generation (VLLM's strength)...")
try:
    responses = client.get_model_response(
        system_prompts=["You are helpful.", "You are helpful.", "You are helpful."],
        user_prompts=["What is 3 + 5?", "What is 10 - 4?", "What is 7 * 3?"],
        max_new_tokens=50,
        temperature=0.1
    )
    print(f"  ✅ Generated {len(responses)} responses in one batch")
    for i, resp in enumerate(responses, 1):
        print(f"     {i}. {resp[:80]}...")
except Exception as e:
    print(f"  ❌ Batch generation failed: {e}")
    sys.exit(1)

# Test 6: Factory pattern
print("\n✓ Test 6: Testing ModelClientFactory.create_client...")
try:
    factory_client = ModelClientFactory.create_client(
        client_type='vllm',
        model_name='microsoft/phi-4',
        max_model_len=2048
    )
    test_resp = factory_client.get_single_response(
        "You are helpful.",
        "Say hello",
        max_new_tokens=20
    )
    print(f"  ✅ Factory works: {test_resp[:80]}")
except Exception as e:
    print(f"  ❌ Factory failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nVLLM client is working correctly.")
print("You can now run experiments with:")
print("  python scripts/run_paper_experiments.py --models phi-4 --benchmarks gsm8k")

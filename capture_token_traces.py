#!/usr/bin/env python3
import subprocess
import os
import time

def create_test_script(token_id):
    """Create a test script for a specific token"""
    script_content = f'''from llama_cpp import Llama
import sys

llm = Llama(model_path="/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf",
            vocab_only=True, verbose=False)

result = llm.detokenize([{token_id}])
'''
    
    with open(f'test_token_{token_id}.py', 'w') as f:
        f.write(script_content)

def run_pin_trace(token_id):
    """Run PIN trace for a specific token"""
    pin_cmd = [
        '/home/user/pin-3.31-intel/pin',
        '-t', '/home/user/pin-3.31-intel/source/tools/detokenize_tracer/obj-intel64/detokenize_tracer.so',
        '--', 'python3', f'test_token_{token_id}.py'
    ]
    
    print(f"Tracing token {token_id}...")
    try:
        result = subprocess.run(pin_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            print(f"Error tracing token {token_id}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout tracing token {token_id}")
        return False

def main():
    # Create output directory
    os.makedirs('token_traces', exist_ok=True)
    
    # Activate virtual environment
    os.environ['PATH'] = '/home/user/Projects/llamacpp_detection/.venv/bin:' + os.environ['PATH']
    
    successful_traces = 0
    failed_traces = []
    
    for token_id in range(128):
        print(f"\n=== Processing Token {token_id+100} ===")
        
        # Create test script
        create_test_script(token_id+100)
        
        # Run PIN trace
        if run_pin_trace(token_id+100):
            # Move the trace file
            if os.path.exists('detokenize_trace.out'):
                os.rename('detokenize_trace.out', f'token_traces/token_{token_id+100:03d}_addresses.txt')
                successful_traces += 1
                print(f"✓ Token {token_id+100} traced successfully")
            else:
                print(f"✗ Token {token_id+100}: trace file not created")
                failed_traces.append(token_id+100)
        else:
            failed_traces.append(token_id+100)
        
        # Clean up test script
        if os.path.exists(f'test_token_{token_id+100}.py'):
            os.remove(f'test_token_{token_id+100}.py')
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.5)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully traced: {successful_traces}/128 tokens")
    print(f"Failed traces: {len(failed_traces)}")
    if failed_traces:
        print(f"Failed tokens: {failed_traces}")
    
    print(f"\nTrace files saved in token_traces/ directory")

if __name__ == "__main__":
    main()

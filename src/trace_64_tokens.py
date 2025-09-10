# trace_64_tokens.py
import subprocess
import os
import json
from pathlib import Path
import sys

def trace_token(token_id):
    """Trace memory accesses for a single token"""
    
    # Create minimal script for this token
    script_content = f'''
from llama_cpp import Llama
result = Llama(
    model_path="/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf", 
    vocab_only=True, 
    verbose=False
).detokenize([{token_id}])
print(f"Token {token_id}: {{repr(result)}}")
'''
    
    # Write temporary script
    temp_script = f'temp_token_{token_id}.py'
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    try:
        # Run perf trace
        perf_data_file = f'token_{token_id}.data'
        cmd = [
            'perf', 'record', '-e', 'cpu/mem-loads/P', 
            '-o', perf_data_file,
            'python3', temp_script
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: perf record failed for token {token_id}: {result.stderr}")
            return []
        
        # Convert to text
        script_cmd = ['perf', 'script', '-i', perf_data_file]
        script_result = subprocess.run(script_cmd, capture_output=True, text=True)
        
        if script_result.returncode != 0:
            print(f"Warning: perf script failed for token {token_id}: {script_result.stderr}")
            return []
        
        # Extract libllama addresses with token_to_piece
        addresses = []
        for line in script_result.stdout.split('\n'):
            if 'libllama.so' in line and 'token_to_piece' in line:
                parts = line.split()
                if len(parts) > 5:
                    # The address should be in field 5 (0-indexed)
                    addr = parts[5]
                    if addr and len(addr) > 8:  # Valid hex address
                        addresses.append(addr)
        
        return addresses
        
    except Exception as e:
        print(f"Error tracing token {token_id}: {e}")
        return []
        
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_script):
                os.remove(temp_script)
            if os.path.exists(perf_data_file):
                os.remove(perf_data_file)
        except OSError:
            pass

def extract_cache_sets(addresses):
    """Convert addresses to cache sets"""
    cache_sets = []
    for addr in addresses:
        try:
            # Remove any leading/trailing whitespace
            addr = addr.strip()
            full_addr = int(addr, 16)
            cache_set = (full_addr >> 6) & 0x3F  # Extract bits 6-11 for 64 cache sets
            cache_sets.append(cache_set)
        except (ValueError, TypeError):
            continue
    return sorted(set(cache_sets))  # Remove duplicates and sort

def main():
    print("Starting automated tracing of 64 tokens...")
    print("This may take several minutes...")
    
    results = {}
    successful_traces = 0
    
    for token_id in range(64):
        print(f"Processing token {token_id:2d}/63...", end=' ', flush=True)
        
        addresses = trace_token(token_id)
        cache_sets = extract_cache_sets(addresses)
        
        if addresses:
            successful_traces += 1
            print(f"✓ Found {len(addresses)} addresses -> {len(cache_sets)} cache sets")
        else:
            print("✗ No addresses found")
        
        results[token_id] = {
            'addresses': addresses,
            'cache_sets': cache_sets,
            'address_count': len(addresses),
            'cache_set_count': len(cache_sets)
        }
    
    # Save results to JSON
    output_file = 'token_cache_fingerprints_2.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TRACING COMPLETE!")
    print(f"{'='*60}")
    print(f"Successfully traced: {successful_traces}/64 tokens")
    print(f"Results saved to: {output_file}")
    
    # Print summary of first 10 tokens
    print(f"\nSample results (first 10 tokens):")
    print(f"{'Token':<6} {'Cache Sets':<30} {'Addresses'}")
    print(f"{'-'*50}")
    
    for token_id in range(min(10, len(results))):
        sets = results[token_id]['cache_sets']
        addr_count = results[token_id]['address_count']
        sets_str = str(sets) if len(str(sets)) < 28 else str(sets)[:25] + "..."
        print(f"{token_id:<6} {sets_str:<30} {addr_count}")
    
    # Analysis summary
    all_cache_sets = set()
    tokens_with_data = 0
    
    for token_id, data in results.items():
        if data['cache_sets']:
            tokens_with_data += 1
            all_cache_sets.update(data['cache_sets'])
    
    print(f"\nAnalysis Summary:")
    print(f"- Tokens with cache data: {tokens_with_data}/64")
    print(f"- Unique cache sets used: {len(all_cache_sets)}")
    print(f"- Cache sets: {sorted(all_cache_sets)}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        # Clean up any remaining temp files
        for i in range(64):
            for f in [f'temp_token_{i}.py', f'token_{i}.data']:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except OSError:
                    pass
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
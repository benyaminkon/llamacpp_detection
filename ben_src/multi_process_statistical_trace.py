# multi_process_statistical_trace.py
import subprocess
import os
import json
import time
from pathlib import Path
from collections import Counter
import tempfile
import sys

def create_single_token_script(token_id, script_path, include_detokenize=True):
    """Create a script that optionally includes detokenization for differential analysis"""
    
    # Model loading stays the same
    # Detokenization line changes based on parameter
    if include_detokenize:
        detokenize_line = f"    result = llm.detokenize([{token_id}])"
        debug_line = f'    print(f"Token {token_id}: {{repr(result)}}", file=sys.stderr)'
    else:
        detokenize_line = f"    # result = llm.detokenize([{token_id}])  # DISABLED FOR CONTROL"
        debug_line = f'    print(f"Token {token_id}: CONTROL RUN (no detokenize)", file=sys.stderr)'
    
    script_content = f'''#!/usr/bin/env python3
import sys
import os
import time
import subprocess

from llama_cpp import Llama

try:
    # Load model first
    llm = Llama(
        model_path="/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf",
        vocab_only=True,
        verbose=False
    )
    
    time.sleep(0.2)  # Let model settle
    
    # Start perf
    perf_process = subprocess.Popen([
        'perf', 'record', '-e', 'cpu/mem-loads/P',
        '-p', str(os.getpid()), '-o', 'temp.data'
    ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    
    time.sleep(0.05)
    
{detokenize_line}
    
    time.sleep(0.05)
    
    # Stop perf
    perf_process.terminate()
    perf_process.wait()
    
    # Extract addresses
    result = subprocess.run(['perf', 'script', '-i', 'temp.data'], 
                           capture_output=True, text=True)
    
    for line in result.stdout.split('\\n'):
        if 'libllama.so' in line:
            parts = line.split()
            if len(parts) > 5 and len(parts[5]) >= 8:
                print(parts[5])
    
{debug_line}
    
    if os.path.exists('temp.data'):
        os.remove('temp.data')
        
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

def run_single_trace(token_id, run_number, temp_dir):
    """Run differential trace - with and without detokenization"""
    script_with_path = temp_dir / f"trace_token_{token_id}_run_{run_number}_with.py"
    script_without_path = temp_dir / f"trace_token_{token_id}_run_{run_number}_without.py"
    
    try:
        # Create both scripts
        create_single_token_script(token_id, script_with_path, include_detokenize=True)
        create_single_token_script(token_id, script_without_path, include_detokenize=False)
        
        # Run script WITH detokenization
        result_with = subprocess.run([
            'python3', str(script_with_path)
        ], capture_output=True, text=True, timeout=45, cwd=temp_dir)
        
        if result_with.returncode != 0:
            print(f"Warning: WITH script failed for token {token_id} run {run_number}: {result_with.stderr}")
            return []
        
        # Parse addresses from WITH run
        addresses_with = set()
        for line in result_with.stdout.strip().split('\n'):
            line = line.strip()
            if line and len(line) >= 8:
                addresses_with.add(line)
        
        # Small delay between runs
        time.sleep(0.1)
        
        # Run script WITHOUT detokenization  
        result_without = subprocess.run([
            'python3', str(script_without_path)
        ], capture_output=True, text=True, timeout=45, cwd=temp_dir)
        
        if result_without.returncode != 0:
            print(f"Warning: WITHOUT script failed for token {token_id} run {run_number}: {result_without.stderr}")
            return []
        
        # Parse addresses from WITHOUT run
        addresses_without = set()
        for line in result_without.stdout.strip().split('\n'):
            line = line.strip()
            if line and len(line) >= 8:
                addresses_without.add(line)
        
        # Find addresses that are ONLY in the WITH run (detokenization-specific)
        detokenize_specific_addresses = addresses_with - addresses_without
        
        return list(detokenize_specific_addresses)
        
    except subprocess.TimeoutExpired:
        print(f"Timeout: token {token_id} run {run_number}")
        return []
    except Exception as e:
        print(f"Error in token {token_id} run {run_number}: {e}")
        return []
    finally:
        # Cleanup
        try:
            if script_with_path.exists():
                script_with_path.unlink()
            if script_without_path.exists():
                script_without_path.unlink()
        except:
            pass

def analyze_address_consistency(addresses_across_runs, threshold_percentage=75):
    """Analyze which addresses appear consistently across runs"""
    if not addresses_across_runs:
        return {
            'consistent_addresses': [],
            'address_frequencies': {},
            'total_runs': 0,
            'threshold_used': threshold_percentage
        }
    
    total_runs = len(addresses_across_runs)
    threshold_count = int(total_runs * threshold_percentage / 100)
    
    # Count frequency of each address
    address_counter = Counter()
    for run_addresses in addresses_across_runs:
        for addr in run_addresses:
            address_counter[addr] += 1
    
    # Find addresses that meet threshold
    consistent_addresses = [
        addr for addr, count in address_counter.items() 
        if count >= threshold_count
    ]
    
    return {
        'consistent_addresses': sorted(consistent_addresses),
        'address_frequencies': dict(address_counter),
        'total_runs': total_runs,
        'threshold_used': threshold_percentage,
        'threshold_count': threshold_count
    }

def extract_cache_sets(addresses):
    """Convert addresses to cache sets"""
    cache_sets = []
    for addr in addresses:
        try:
            full_addr = int(addr, 16)
            cache_set = (full_addr >> 6) & 0x3F  # Extract bits 6-11
            cache_sets.append(cache_set)
        except (ValueError, TypeError):
            continue
    return sorted(set(cache_sets))

def trace_token_multiple_runs(token_id, num_runs=20, threshold=75):
    """Trace a single token across multiple process runs"""
    print(f"Tracing token {token_id} across {num_runs} runs...", flush=True)
    
    temp_dir = Path(tempfile.mkdtemp(prefix=f"token_{token_id}_"))
    addresses_across_runs = []
    successful_runs = 0
    
    try:
        for run_num in range(num_runs):
            print(f"  Run {run_num + 1:2d}/{num_runs}...", end=' ', flush=True)
            
            addresses = run_single_trace(token_id, run_num, temp_dir)
            
            if addresses:
                addresses_across_runs.append(addresses)
                successful_runs += 1
                print(f"✓ ({len(addresses)} addresses)")
            else:
                print("✗ (no addresses)")
            
            # Small delay between runs to reduce system interference
            time.sleep(0.1)
        
        # Analyze consistency
        analysis = analyze_address_consistency(addresses_across_runs, threshold)
        consistent_cache_sets = extract_cache_sets(analysis['consistent_addresses'])
        
        print(f"  Result: {len(analysis['consistent_addresses'])} consistent addresses "
              f"-> {len(consistent_cache_sets)} cache sets "
              f"(from {successful_runs}/{num_runs} successful runs)")
        
        return {
            'token_id': token_id,
            'successful_runs': successful_runs,
            'total_runs': num_runs,
            'all_run_addresses': addresses_across_runs,
            'consistency_analysis': analysis,
            'consistent_cache_sets': consistent_cache_sets,
            'success_rate': successful_runs / num_runs if num_runs > 0 else 0
        }
        
    finally:
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

def save_detailed_results(results, filename):
    """Save results with detailed analysis"""
    # Create summary for easier reading
    summary = {
        'overview': {
            'total_tokens': len(results),
            'successful_tokens': sum(1 for r in results.values() if r['consistent_cache_sets']),
            'average_success_rate': sum(r['success_rate'] for r in results.values()) / len(results)
        },
        'token_summaries': {},
        'detailed_results': results
    }
    
    for token_id, data in results.items():
        summary['token_summaries'][token_id] = {
            'consistent_addresses': len(data['consistency_analysis']['consistent_addresses']),
            'consistent_cache_sets': data['consistent_cache_sets'],
            'success_rate': data['success_rate'],
            'runs_successful': f"{data['successful_runs']}/{data['total_runs']}"
        }
    
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    print("MULTI-PROCESS STATISTICAL TOKEN TRACING")
    print("=" * 60)
    
    # Configuration
    NUM_RUNS = 20
    THRESHOLD = 75  # Percentage threshold for consistency
    TOKENS_TO_TEST = list(range(64))  # All 64 tokens
    
    # Option to test just a few tokens first
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        TOKENS_TO_TEST = [0, 1, 2, 18]  # Test subset
        print("TEST MODE: Only processing tokens 0, 1, 2, 18")
    
    print(f"Configuration:")
    print(f"  Tokens to process: {len(TOKENS_TO_TEST)}")
    print(f"  Runs per token: {NUM_RUNS}")
    print(f"  Consistency threshold: {THRESHOLD}%")
    print(f"  Estimated time: ~{len(TOKENS_TO_TEST) * NUM_RUNS * 0.5:.1f} minutes")
    print()
    
    # Confirm with user
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    results = {}
    start_time = time.time()
    
    try:
        for i, token_id in enumerate(TOKENS_TO_TEST):
            print(f"\n[{i+1}/{len(TOKENS_TO_TEST)}] Processing Token {token_id}")
            print("-" * 40)
            
            result = trace_token_multiple_runs(token_id, NUM_RUNS, THRESHOLD)
            results[token_id] = result
            
            # Save intermediate results (in case of interruption)
            if (i + 1) % 5 == 0:
                interim_file = f"interim_results_{i+1}.json"
                save_detailed_results(results, interim_file)
                print(f"  Saved interim results to {interim_file}")
    
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user after processing {len(results)} tokens.")
    
    # Save final results
    output_file = "statistical_token_fingerprints.json"
    save_detailed_results(results, output_file)
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"STATISTICAL TRACING COMPLETE")
    print(f"=" * 60)
    print(f"Processed: {len(results)} tokens")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    
    # Quick summary
    successful_tokens = sum(1 for r in results.values() if r['consistent_cache_sets'])
    avg_success_rate = sum(r['success_rate'] for r in results.values()) / len(results) if results else 0
    
    print(f"\nQuick Summary:")
    print(f"  Tokens with consistent patterns: {successful_tokens}/{len(results)}")
    print(f"  Average run success rate: {avg_success_rate:.1%}")
    
    # Show sample results
    print(f"\nSample Results (first 5 tokens):")
    for token_id in sorted(results.keys())[:5]:
        data = results[token_id]
        cache_sets = data['consistent_cache_sets']
        success = data['success_rate']
        print(f"  Token {token_id:2d}: {len(cache_sets)} cache sets, {success:.1%} success -> {cache_sets}")

if __name__ == "__main__":
    main()
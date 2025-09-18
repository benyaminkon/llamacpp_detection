#!/usr/bin/env python3
import os
import glob
import re

def analyze_token_trace(filename):
    """Analyze a single token trace file"""
    # Extract token ID from filename like 'token_traces/token_001_addresses.txt'
    match = re.search(r'token_(\d+)_addresses\.txt', filename)
    if not match:
        return None, None, 0
    
    token_id = int(match.group(1))
    
    if not os.path.exists(filename):
        return token_id, None, 0
    
    addresses = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('0x'):
                try:
                    addr = int(line, 16)
                    addresses.append(addr)
                except ValueError:
                    continue
    
    return token_id, addresses, len(addresses)

def main():
    print("=== Analyzing All Token Traces ===")
    
    # Find all trace files
    trace_files = glob.glob('token_traces/token_*_addresses.txt')
    trace_files.sort()
    
    if not trace_files:
        print("No trace files found in token_traces/ directory")
        return
    
    results = []
    cache_line_size = 64
    
    for filename in trace_files:
        token_id, addresses, count = analyze_token_trace(filename)
        
        if token_id is None:
            continue
            
        if addresses:
            # Calculate cache sets (assuming 64-set L1 cache)
            cache_sets = []
            for addr in addresses:
                cache_line = addr // cache_line_size
                cache_set = cache_line % 64
                cache_sets.append(cache_set)
            
            unique_sets = list(cache_sets)
            results.append({
                'token': token_id,
                'addresses': len(addresses),
                'cache_sets': unique_sets,
                'unique_sets': len(set(unique_sets))
            })
            
            print(f"Token {token_id:3d}: {len(addresses):2d} addresses, {len(set(unique_sets)):2d} cache sets: {unique_sets}")
        else:
            print(f"Token {token_id:3d}: No addresses captured")
    
    # Summary statistics
    if results:
        addr_counts = [r['addresses'] for r in results]
        set_counts = [r['unique_sets'] for r in results]
        
        print(f"\n=== SUMMARY ===")
        print(f"Tokens successfully traced: {len(results)}")
        print(f"Average addresses per token: {sum(addr_counts)/len(addr_counts):.1f}")
        print(f"Average cache sets per token: {sum(set_counts)/len(set_counts):.1f}")
        print(f"Address count range: {min(addr_counts)}-{max(addr_counts)}")
        print(f"Cache set count range: {min(set_counts)}-{max(set_counts)}")
        
        # Save summary
        with open('token_analysis_summary.txt', 'w') as f:
            f.write("Token\tAddresses\tCacheSets\tUniqueSets\n")
            for r in results:
                cache_sets_str = ','.join(map(str, r['cache_sets']))
                f.write(f"{r['token']}\t{r['addresses']}\t{cache_sets_str}\t{r['unique_sets']}\n")
        
        print(f"\nDetailed results saved to token_analysis_summary.txt")
    else:
        print("No successful traces found")

if __name__ == "__main__":
    main()
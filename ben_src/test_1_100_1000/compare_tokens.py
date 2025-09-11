#!/usr/bin/env python3

import re
import sys
from collections import Counter, defaultdict

def extract_offsets(filename):
    """Extract offsets from libllama.so addresses in perf script output"""
    
    print(f"Processing {filename}...")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find libllama.so lines
    libllama_lines = [line for line in lines if 'libllama.so' in line]
    
    if not libllama_lines:
        print(f"No libllama.so addresses found in {filename}")
        return {}, None
    
    # Extract the first address to use as base
    first_line = libllama_lines[0]
    base_addr_match = re.search(r'([0-9a-f]{12,16})', first_line)
    
    if not base_addr_match:
        print(f"Could not find base address in {filename}")
        return {}, None
    
    base_addr = int(base_addr_match.group(1), 16)
    print(f"Base address for {filename}: 0x{base_addr:x}")
    
    offsets = []
    
    for line in libllama_lines:
        # Extract address
        addr_match = re.search(r'([0-9a-f]{12,16})', line)
        if addr_match:
            addr = int(addr_match.group(1), 16)
            offset = addr - base_addr
            
            # Extract function name if available
            func_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\+0x[0-9a-f]+)?)\s*\(', line)
            func_name = func_match.group(1) if func_match else '[unknown]'
            
            offsets.append((offset, func_name))
    
    return offsets, base_addr

def analyze_token_patterns():
    """Compare memory access patterns across different tokens"""
    
    token_files = {
        1: 'memory_token1.txt',
        100: 'memory_token100.txt', 
        1000: 'memory_token1000.txt'
    }
    
    token_data = {}
    
    # Extract data for each token
    for token_id, filename in token_files.items():
        try:
            offsets, base_addr = extract_offsets(filename)
            if offsets:
                token_data[token_id] = {
                    'offsets': offsets,
                    'base_addr': base_addr,
                    'offset_counts': Counter([offset for offset, func in offsets])
                }
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping token {token_id}")
    
    if len(token_data) < 2:
        print("Need at least 2 token files to compare")
        return
    
    print(f"\n=== TOKEN COMPARISON ANALYSIS ===")
    print(f"Analyzing {len(token_data)} tokens: {list(token_data.keys())}")
    
    # Find common offsets (appear in all tokens)
    all_offsets = [set(data['offset_counts'].keys()) for data in token_data.values()]
    common_offsets = set.intersection(*all_offsets)
    
    # Find token-specific offsets
    token_specific = {}
    for token_id, data in token_data.items():
        other_tokens = [set(other_data['offset_counts'].keys()) for other_id, other_data in token_data.items() if other_id != token_id]
        if other_tokens:
            other_offsets = set.union(*other_tokens)
            specific = set(data['offset_counts'].keys()) - other_offsets
            token_specific[token_id] = specific
    
    print(f"\n=== COMMON OFFSETS (appear in ALL tokens) ===")
    print(f"Found {len(common_offsets)} common offsets")
    if common_offsets:
        # Get function names from first token
        first_token_data = list(token_data.values())[0]
        offset_to_func = {offset: func for offset, func in first_token_data['offsets']}
        
        for offset in sorted(common_offsets)[:10]:  # Show first 10
            func_name = offset_to_func.get(offset, '[unknown]')
            print(f"  0x{offset:x}: {func_name}")
        if len(common_offsets) > 10:
            print(f"  ... and {len(common_offsets) - 10} more")
    
    print(f"\n=== TOKEN-SPECIFIC OFFSETS ===")
    for token_id, specific_offsets in token_specific.items():
        print(f"\nToken {token_id} has {len(specific_offsets)} unique offsets:")
        if specific_offsets:
            data = token_data[token_id]
            offset_to_func = {offset: func for offset, func in data['offsets']}
            
            for offset in sorted(specific_offsets)[:5]:  # Show first 5
                func_name = offset_to_func.get(offset, '[unknown]')
                print(f"  0x{offset:x}: {func_name}")
            if len(specific_offsets) > 5:
                print(f"  ... and {len(specific_offsets) - 5} more")
    
    # Look for patterns in token_to_piece function specifically
    print(f"\n=== TOKEN_TO_PIECE FUNCTION ANALYSIS ===")
    token_to_piece_offsets = defaultdict(list)
    
    for token_id, data in token_data.items():
        for offset, func_name in data['offsets']:
            if 'token_to_piece' in func_name:
                token_to_piece_offsets[token_id].append(offset)
    
    print("token_to_piece offsets by token:")
    for token_id in sorted(token_to_piece_offsets.keys()):
        offsets = sorted(set(token_to_piece_offsets[token_id]))
        print(f"  Token {token_id}: {[hex(off) for off in offsets]}")
    
    # Calculate cache set mapping (assuming 64-byte cache lines, 64 sets)
    print(f"\n=== CACHE SET MAPPING ===")
    print("Cache sets accessed by each token (bits 6-11 of offset):")
    
    for token_id, data in token_data.items():
        cache_sets = set()
        for offset, func_name in data['offsets']:
            if 'token_to_piece' in func_name or 'llama_vocab' in func_name:
                # Calculate cache set: (offset >> 6) & 0x3f for 64 sets
                cache_set = (offset >> 6) & 0x3f
                cache_sets.add(cache_set)
        
        print(f"  Token {token_id}: cache sets {sorted(cache_sets)}")
    
    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    for token_id, data in token_data.items():
        total_accesses = sum(data['offset_counts'].values())
        unique_offsets = len(data['offset_counts'])
        print(f"Token {token_id}:")
        print(f"  Total libllama.so accesses: {total_accesses}")
        print(f"  Unique offsets: {unique_offsets}")
        
        # Count by function type
        func_counts = defaultdict(int)
        for offset, func_name in data['offsets']:
            if 'token_to_piece' in func_name:
                func_counts['token_to_piece'] += 1
            elif 'llama_vocab' in func_name:
                func_counts['llama_vocab'] += 1
            elif 'replace_all' in func_name:
                func_counts['replace_all'] += 1
            else:
                func_counts['other'] += 1
        
        print(f"  Function breakdown: {dict(func_counts)}")
    
    # Save detailed comparison
    with open('token_comparison_results.txt', 'w') as f:
        f.write("# Token Memory Access Pattern Comparison\n\n")
        
        f.write("## Cache Set Mapping by Token\n")
        for token_id, data in token_data.items():
            f.write(f"Token {token_id}:\n")
            cache_sets = []
            for offset, func_name in data['offsets']:
                if 'token_to_piece' in func_name or 'llama_vocab' in func_name:
                    cache_set = (offset >> 6) & 0x3f
                    cache_sets.append((offset, cache_set, func_name))
            
            for offset, cache_set, func_name in sorted(cache_sets):
                f.write(f"  0x{offset:x} -> cache_set_{cache_set} ({func_name})\n")
            f.write("\n")
    
    print("\nDetailed results saved to 'token_comparison_results.txt'")

if __name__ == '__main__':
    analyze_token_patterns()
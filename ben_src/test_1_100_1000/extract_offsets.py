#!/usr/bin/env python3

import re
import sys
from collections import Counter

def extract_offsets(filename):
    """Extract offsets from libllama.so addresses in perf script output"""
    
    print(f"Processing {filename}...")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find libllama.so lines
    libllama_lines = [line for line in lines if 'libllama.so' in line]
    
    if not libllama_lines:
        print(f"No libllama.so addresses found in {filename}")
        return {}
    
    # Extract the first address to use as base
    first_line = libllama_lines[0]
    base_addr_match = re.search(r'([0-9a-f]{12,16})', first_line)
    
    if not base_addr_match:
        print(f"Could not find base address in {filename}")
        return {}
    
    base_addr = int(base_addr_match.group(1), 16)
    print(f"Base address for {filename}: 0x{base_addr:x}")
    
    offsets = {}
    
    for line in libllama_lines:
        # Extract address
        addr_match = re.search(r'([0-9a-f]{12,16})', line)
        if addr_match:
            addr = int(addr_match.group(1), 16)
            offset = addr - base_addr
            
            # Extract function name if available
            func_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\+0x[0-9a-f]+)?)\s*\(', line)
            func_name = func_match.group(1) if func_match else '[unknown]'
            
            offsets[offset] = func_name
    
    return offsets

def main():
    # Extract offsets from both files
    offsets_with = extract_offsets('memory_with_decode.txt')
    offsets_without = extract_offsets('memory_without_decode.txt')
    
    if not offsets_with or not offsets_without:
        print("Failed to extract offsets from one or both files")
        return
    
    # Count frequencies
    with_counter = Counter(offsets_with.keys())
    without_counter = Counter(offsets_without.keys())
    
    print("\n=== COMPARISON RESULTS ===")
    
    # Find offsets only in "with decode"
    only_with_decode = set(offsets_with.keys()) - set(offsets_without.keys())
    
    print("Offsets that only appear WITH detokenize:")
    if only_with_decode:
        for offset in sorted(only_with_decode):
            print(f"  0x{offset:x}: {offsets_with[offset]}")
    else:
        print("  (None found - all offsets appear in both traces)")
    
    # Find offsets with higher frequency in "with decode"
    print("\nOffsets with HIGHER frequency in detokenize run:")
    higher_freq = []
    for offset in set(offsets_with.keys()) & set(offsets_without.keys()):
        count_with = with_counter[offset]
        count_without = without_counter[offset]
        if count_with > count_without:
            higher_freq.append((offset, count_with, count_without, offsets_with[offset]))
    
    if higher_freq:
        for offset, count_with, count_without, func_name in sorted(higher_freq):
            print(f"  0x{offset:x}: {count_with} vs {count_without} times - {func_name}")
    else:
        print("  (No offsets with higher frequency)")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total libllama.so accesses WITH detokenize: {sum(with_counter.values())}")
    print(f"Total libllama.so accesses WITHOUT detokenize: {sum(without_counter.values())}")
    print(f"Unique offsets WITH detokenize: {len(offsets_with)}")
    print(f"Unique offsets WITHOUT detokenize: {len(offsets_without)}")
    
    # Save detailed results
    with open('decode_specific_offsets.txt', 'w') as f:
        f.write("# Offsets only appearing during detokenize:\n")
        for offset in sorted(only_with_decode):
            f.write(f"0x{offset:x} {offsets_with[offset]}\n")
        
        f.write("\n# Offsets with higher frequency during detokenize:\n")
        for offset, count_with, count_without, func_name in sorted(higher_freq):
            f.write(f"0x{offset:x} {count_with}vs{count_without} {func_name}\n")
    
    print("\nDetailed results saved to 'decode_specific_offsets.txt'")

if __name__ == '__main__':
    main()
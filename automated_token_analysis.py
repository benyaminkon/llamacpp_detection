#!/usr/bin/env python3

import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

class TokenAnalyzer:
    def __init__(self, model_path, start_token=10000, end_token=10128):
        self.model_path = model_path
        self.start_token = start_token
        self.end_token = end_token
        self.results_dir = Path("token_analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Summary data
        self.token_cache_sets = {}
        self.token_functions = {}
        self.failed_tokens = []
        
    def create_test_scripts(self):
        """Create the with/without decode test scripts"""
        
        # Script WITH detokenize
        with_script = '''from llama_cpp import Llama
import sys

if len(sys.argv) != 2:
    print("Usage: python3 script.py <token_id>")
    sys.exit(1)

token_id = int(sys.argv[1])
model = Llama(model_path="{model_path}", vocab_only=True, verbose=False)
result = model.detokenize([token_id])
'''.format(model_path=self.model_path)
        
        # Script WITHOUT detokenize
        without_script = '''from llama_cpp import Llama
import sys

if len(sys.argv) != 2:
    print("Usage: python3 script.py <token_id>")
    sys.exit(1)

token_id = int(sys.argv[1])
model = Llama(model_path="{model_path}", vocab_only=True, verbose=False)
'''.format(model_path=self.model_path)
        
        with open('test_token_with.py', 'w') as f:
            f.write(with_script)
        with open('test_token_without.py', 'w') as f:
            f.write(without_script)
            
        print("✓ Created test scripts")
    
    def run_perf_trace(self, script_name, token_id, output_suffix):
        """Run perf trace for a specific token"""
        
        perf_data_file = f"perf_token{token_id}_{output_suffix}.data"
        memory_file = f"memory_token{token_id}_{output_suffix}.txt"
        
        try:
            # Run perf record
            cmd_record = [
                'perf', 'record', 
                '-e', 'cpu/mem-loads/P',
                '-o', perf_data_file,
                'python3', script_name, str(token_id)
            ]
            
            result = subprocess.run(cmd_record, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"    ❌ perf record failed: {result.stderr}")
                return None
                
            # Convert to text
            cmd_script = ['perf', 'script', '-i', perf_data_file]
            with open(memory_file, 'w') as f:
                result = subprocess.run(cmd_script, stdout=f, stderr=subprocess.PIPE, text=True, timeout=30)
                
            if result.returncode != 0:
                print(f"    ❌ perf script failed: {result.stderr}")
                return None
                
            # Clean up perf data file
            os.remove(perf_data_file)
            
            return memory_file
            
        except subprocess.TimeoutExpired:
            print(f"    ❌ Timeout during perf trace")
            return None
        except Exception as e:
            print(f"    ❌ Error during perf trace: {e}")
            return None
    
    def extract_offsets(self, filename):
        """Extract offsets from libllama.so addresses in perf script output"""
        
        if not os.path.exists(filename):
            return [], None
            
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Find libllama.so lines
        libllama_lines = [line for line in lines if 'libllama.so' in line]
        
        if not libllama_lines:
            return [], None
        
        # Extract the first address to use as base
        first_line = libllama_lines[0]
        base_addr_match = re.search(r'([0-9a-f]{12,16})', first_line)
        
        if not base_addr_match:
            return [], None
        
        base_addr = int(base_addr_match.group(1), 16)
        
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
    
    def analyze_token(self, token_id):
        """Analyze a single token's memory access patterns"""
        
        print(f"  📊 Analyzing token {token_id}...")
        
        # Extract offsets from both traces
        with_file = f"memory_token{token_id}_with.txt"
        without_file = f"memory_token{token_id}_without.txt"
        
        offsets_with, base_with = self.extract_offsets(with_file)
        offsets_without, base_without = self.extract_offsets(without_file)
        
        if not offsets_with:
            print(f"    ❌ No offsets found in {with_file}")
            return False
            
        if not offsets_without:
            print(f"    ❌ No offsets found in {without_file}")
            return False
        
        # Convert to sets for comparison
        with_offsets = set(offset for offset, func in offsets_with)
        without_offsets = set(offset for offset, func in offsets_without)
        
        # Find decode-specific offsets
        decode_only_offsets = with_offsets - without_offsets
        
        # Get function mapping
        offset_to_func = {offset: func for offset, func in offsets_with}
        
        # Find cache sets for decode-related functions
        decode_cache_sets = set()
        decode_functions = []
        
        for offset in decode_only_offsets:
            func_name = offset_to_func.get(offset, '[unknown]')
            
            if any(keyword in func_name for keyword in ['token_to_piece', 'llama_vocab']):
                cache_set = (offset >> 6) & 0x3f  # bits 6-11 for 64 cache sets
                decode_cache_sets.add(cache_set)
                decode_functions.append((offset, cache_set, func_name))
        
        # Store results
        self.token_cache_sets[token_id] = sorted(decode_cache_sets)
        self.token_functions[token_id] = decode_functions
        
        # Save individual analysis
        individual_file = self.results_dir / f"token_{token_id}_analysis.txt"
        with open(individual_file, 'w') as f:
            f.write(f"Token {token_id} Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Base address (with): 0x{base_with:x}\n")
            f.write(f"Base address (without): 0x{base_without:x}\n")
            f.write(f"Total offsets (with): {len(with_offsets)}\n")
            f.write(f"Total offsets (without): {len(without_offsets)}\n")
            f.write(f"Decode-only offsets: {len(decode_only_offsets)}\n\n")
            
            f.write("Cache Sets Accessed:\n")
            f.write(f"  {sorted(decode_cache_sets)}\n\n")
            
            f.write("Decode-Related Functions:\n")
            for offset, cache_set, func_name in decode_functions:
                f.write(f"  0x{offset:x} -> cache_set_{cache_set} ({func_name})\n")
        
        # Clean up memory trace files
        try:
            os.remove(with_file)
            os.remove(without_file)
        except:
            pass
            
        print(f"    ✓ Cache sets: {sorted(decode_cache_sets)}")
        return True
    
    def process_token(self, token_id, current, total):
        """Process a single token completely"""
        
        print(f"\n[{current:3d}/{total:3d}] Processing token {token_id}")
        
        # Run with decode
        print("  🔍 Tracing WITH detokenize...")
        with_file = self.run_perf_trace('test_token_with.py', token_id, 'with')
        if not with_file:
            self.failed_tokens.append(token_id)
            return False
            
        # Run without decode  
        print("  🔍 Tracing WITHOUT detokenize...")
        without_file = self.run_perf_trace('test_token_without.py', token_id, 'without')
        if not without_file:
            self.failed_tokens.append(token_id)
            return False
        
        # Analyze
        success = self.analyze_token(token_id)
        if not success:
            self.failed_tokens.append(token_id)
            return False
            
        return True
    
    def generate_summary(self):
        """Generate comprehensive summary of all tokens"""
        
        print(f"\n📋 Generating summary report...")
        
        summary_file = self.results_dir / "summary_report.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Token Cache Set Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analyzed tokens: {self.start_token}-{self.end_token}\n")
            f.write(f"Successful: {len(self.token_cache_sets)}\n")
            f.write(f"Failed: {len(self.failed_tokens)}\n\n")
            
            if self.failed_tokens:
                f.write(f"Failed tokens: {self.failed_tokens}\n\n")
            
            f.write("Cache Set Mapping by Token:\n")
            f.write("-" * 30 + "\n")
            
            for token_id in sorted(self.token_cache_sets.keys()):
                cache_sets = self.token_cache_sets[token_id]
                f.write(f"Token {token_id:5d}: {cache_sets}\n")
            
            f.write("\nDetailed Function Mappings:\n")
            f.write("-" * 30 + "\n")
            
            for token_id in sorted(self.token_functions.keys()):
                f.write(f"\nToken {token_id}:\n")
                functions = self.token_functions[token_id]
                for offset, cache_set, func_name in functions:
                    f.write(f"  0x{offset:x} -> cache_set_{cache_set} ({func_name})\n")
        
        # Generate CSV for easy processing
        csv_file = self.results_dir / "token_cache_mapping.csv"
        with open(csv_file, 'w') as f:
            f.write("token_id,cache_sets,num_cache_sets\n")
            for token_id in sorted(self.token_cache_sets.keys()):
                cache_sets = self.token_cache_sets[token_id]
                cache_sets_str = '|'.join(map(str, cache_sets))
                f.write(f"{token_id},{cache_sets_str},{len(cache_sets)}\n")
        
        print(f"✓ Summary saved to {summary_file}")
        print(f"✓ CSV data saved to {csv_file}")
    
    def run_analysis(self):
        """Run the complete analysis"""
        
        print("🚀 Starting automated token analysis")
        print(f"📊 Analyzing tokens {self.start_token}-{self.end_token}")
        print(f"📁 Results will be saved to {self.results_dir}")
        
        # Create test scripts
        self.create_test_scripts()
        
        # Process each token
        total_tokens = self.end_token - self.start_token + 1
        
        start_time = time.time()
        
        for i, token_id in enumerate(range(self.start_token, self.end_token + 1)):
            current = i + 1
            
            # Show progress
            elapsed = time.time() - start_time
            if current > 1:
                avg_time = elapsed / (current - 1)
                remaining = avg_time * (total_tokens - current + 1)
                print(f"⏱️  Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s")
            
            self.process_token(token_id, current, total_tokens)
        
        # Generate summary
        self.generate_summary()
        
        # Cleanup
        try:
            os.remove('test_token_with.py')
            os.remove('test_token_without.py')
        except:
            pass
        
        total_time = time.time() - start_time
        print(f"\n🎉 Analysis complete!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"✅ Successful: {len(self.token_cache_sets)}/{total_tokens}")
        print(f"❌ Failed: {len(self.failed_tokens)}/{total_tokens}")
        print(f"📁 Results in: {self.results_dir}")

def main():
    model_path = "/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf"
    
    analyzer = TokenAnalyzer(model_path, start_token=10000, end_token=10128)
    analyzer.run_analysis()

if __name__ == '__main__':
    main()
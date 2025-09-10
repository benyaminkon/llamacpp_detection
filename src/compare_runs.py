# compare_runs.py - Compare consistency between two experimental runs
import json
import sys
from pathlib import Path

def load_run_data(filename):
    """Load cache fingerprint data from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{filename}'!")
        return None

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def analyze_token_consistency(run1, run2):
    """Analyze consistency between runs for each token"""
    results = {
        'identical': [],
        'high_similarity': [],
        'medium_similarity': [],
        'low_similarity': [],
        'no_similarity': []
    }
    
    detailed_results = {}
    
    for token_id in range(64):
        token_str = str(token_id)
        
        # Get cache sets for both runs
        sets1 = set(run1[token_str]['cache_sets']) if token_str in run1 else set()
        sets2 = set(run2[token_str]['cache_sets']) if token_str in run2 else set()
        
        # Calculate similarity metrics
        jaccard = calculate_jaccard_similarity(sets1, sets2)
        intersection = sets1 & sets2
        union = sets1 | sets2
        only_in_1 = sets1 - sets2
        only_in_2 = sets2 - sets1
        
        # Categorize similarity
        if sets1 == sets2:
            category = 'identical'
        elif jaccard >= 0.8:
            category = 'high_similarity'
        elif jaccard >= 0.5:
            category = 'medium_similarity'
        elif jaccard >= 0.2:
            category = 'low_similarity'
        else:
            category = 'no_similarity'
        
        results[category].append(token_id)
        
        detailed_results[token_id] = {
            'run1_sets': sorted(sets1),
            'run2_sets': sorted(sets2),
            'jaccard_similarity': jaccard,
            'intersection': sorted(intersection),
            'union': sorted(union),
            'only_in_run1': sorted(only_in_1),
            'only_in_run2': sorted(only_in_2),
            'category': category
        }
    
    return results, detailed_results

def print_summary_statistics(results, run1, run2):
    """Print overall summary statistics"""
    total_tokens = 64
    
    print("CONSISTENCY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total tokens analyzed: {total_tokens}")
    print()
    
    print("Consistency Categories:")
    print(f"  ✓ Identical:        {len(results['identical']):2d}/64 ({len(results['identical'])/64*100:.1f}%)")
    print(f"  ★ High similarity:  {len(results['high_similarity']):2d}/64 ({len(results['high_similarity'])/64*100:.1f}%) [Jaccard ≥ 0.8]")
    print(f"  ◐ Medium similarity: {len(results['medium_similarity']):2d}/64 ({len(results['medium_similarity'])/64*100:.1f}%) [Jaccard ≥ 0.5]")
    print(f"  ◔ Low similarity:   {len(results['low_similarity']):2d}/64 ({len(results['low_similarity'])/64*100:.1f}%) [Jaccard ≥ 0.2]")
    print(f"  ✗ No similarity:    {len(results['no_similarity']):2d}/64 ({len(results['no_similarity'])/64*100:.1f}%) [Jaccard < 0.2]")
    print()
    
    # Overall reliability score
    reliable_tokens = len(results['identical']) + len(results['high_similarity'])
    reliability_score = reliable_tokens / total_tokens * 100
    
    print(f"Overall Reliability Score: {reliability_score:.1f}%")
    print(f"(Tokens with identical or high similarity patterns)")
    print()

def print_detailed_results(detailed_results, show_first_n=15):
    """Print detailed comparison for first N tokens"""
    print(f"DETAILED COMPARISON (First {show_first_n} tokens)")
    print("=" * 80)
    
    for token_id in range(min(show_first_n, 64)):
        details = detailed_results[token_id]
        
        # Status symbol
        status_symbols = {
            'identical': '✓',
            'high_similarity': '★',
            'medium_similarity': '◐',
            'low_similarity': '◔',
            'no_similarity': '✗'
        }
        
        symbol = status_symbols.get(details['category'], '?')
        
        print(f"{symbol} Token {token_id:2d} [{details['category'].replace('_', ' ').title()}] "
              f"(Jaccard: {details['jaccard_similarity']:.3f})")
        
        print(f"   Run 1: {details['run1_sets']}")
        print(f"   Run 2: {details['run2_sets']}")
        
        if details['intersection']:
            print(f"   Common: {details['intersection']}")
        if details['only_in_run1']:
            print(f"   Only in Run 1: {details['only_in_run1']}")
        if details['only_in_run2']:
            print(f"   Only in Run 2: {details['only_in_run2']}")
        print()

def analyze_overall_cache_usage(run1, run2):
    """Analyze overall cache set usage patterns"""
    print("OVERALL CACHE SET USAGE ANALYSIS")
    print("=" * 60)
    
    # Collect all cache sets from both runs
    all_sets_run1 = set()
    all_sets_run2 = set()
    
    for token_id in range(64):
        token_str = str(token_id)
        if token_str in run1:
            all_sets_run1.update(run1[token_str]['cache_sets'])
        if token_str in run2:
            all_sets_run2.update(run2[token_str]['cache_sets'])
    
    common_sets = all_sets_run1 & all_sets_run2
    only_in_run1 = all_sets_run1 - all_sets_run2
    only_in_run2 = all_sets_run2 - all_sets_run1
    
    print(f"Run 1 used {len(all_sets_run1)} unique cache sets: {sorted(all_sets_run1)}")
    print(f"Run 2 used {len(all_sets_run2)} unique cache sets: {sorted(all_sets_run2)}")
    print()
    print(f"Common cache sets ({len(common_sets)}): {sorted(common_sets)}")
    if only_in_run1:
        print(f"Only in Run 1 ({len(only_in_run1)}): {sorted(only_in_run1)}")
    if only_in_run2:
        print(f"Only in Run 2 ({len(only_in_run2)}): {sorted(only_in_run2)}")
    
    # Cache set consistency score
    if len(all_sets_run1) > 0 or len(all_sets_run2) > 0:
        cache_consistency = len(common_sets) / len(all_sets_run1 | all_sets_run2)
        print(f"\nCache Set Consistency: {cache_consistency:.3f}")
    print()

def find_most_problematic_tokens(detailed_results, n=5):
    """Find tokens with lowest consistency"""
    print(f"MOST INCONSISTENT TOKENS (Bottom {n})")
    print("=" * 50)
    
    # Sort by Jaccard similarity (lowest first)
    sorted_tokens = sorted(detailed_results.items(), 
                          key=lambda x: x[1]['jaccard_similarity'])
    
    for i, (token_id, details) in enumerate(sorted_tokens[:n]):
        print(f"{i+1}. Token {token_id} (Jaccard: {details['jaccard_similarity']:.3f})")
        print(f"   Run 1: {details['run1_sets']}")
        print(f"   Run 2: {details['run2_sets']}")
        print()

def main():
    # Default file names
    run1_file = "token_cache_fingerprints_1.json"
    run2_file = "token_cache_fingerprints_2.json"
    
    # Check if files exist
    if not Path(run1_file).exists():
        print(f"Error: {run1_file} not found!")
        print("Make sure you renamed the first run results to 'token_fingerprints_run1.json'")
        return
    
    if not Path(run2_file).exists():
        print(f"Error: {run2_file} not found!")
        print("Make sure you completed the second run")
        return
    
    # Load data
    print("Loading experimental data...")
    run1 = load_run_data(run1_file)
    run2 = load_run_data(run2_file)
    
    if run1 is None or run2 is None:
        return
    
    print(f"✓ Loaded {run1_file}")
    print(f"✓ Loaded {run2_file}")
    print()
    
    # Perform analysis
    consistency_results, detailed_results = analyze_token_consistency(run1, run2)
    
    # Print results
    print_summary_statistics(consistency_results, run1, run2)
    analyze_overall_cache_usage(run1, run2)
    print_detailed_results(detailed_results, show_first_n=10)
    find_most_problematic_tokens(detailed_results, n=3)
    
    # Final assessment
    reliable_tokens = len(consistency_results['identical']) + len(consistency_results['high_similarity'])
    reliability_percentage = reliable_tokens / 64 * 100
    
    print("FINAL ASSESSMENT")
    print("=" * 40)
    if reliability_percentage >= 90:
        assessment = "EXCELLENT - Very reliable for side-channel attack"
        emoji = "🎯"
    elif reliability_percentage >= 75:
        assessment = "GOOD - Suitable for side-channel attack"
        emoji = "✅"
    elif reliability_percentage >= 50:
        assessment = "MODERATE - May work with noise filtering"
        emoji = "⚠️"
    else:
        assessment = "POOR - Too inconsistent for reliable attack"
        emoji = "❌"
    
    print(f"{emoji} {assessment}")
    print(f"Reliability Score: {reliability_percentage:.1f}%")
    
    # Save detailed results to file
    output_file = "consistency_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': consistency_results,
            'detailed': detailed_results,
            'reliability_score': reliability_percentage
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
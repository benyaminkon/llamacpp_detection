# File: evaluate_cache_similarity.py

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def load_cache_data(csv_file):
    """
    Load the cache measurement data from CSV file
    Expected columns: TokenIndex, DetectionIndex, BurstIteration, CacheSet, Probe
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} measurements")
    
    # Check if we have the expected columns
    expected_cols = ['TokenIndex', 'DetectionIndex', 'BurstIteration', 'CacheSet', 'Probe']
    if not all(col in df.columns for col in expected_cols):
        print(f"Warning: Expected columns {expected_cols}")
        print(f"Found columns: {df.columns.tolist()}")
    
    return df

def plot_token_detections(df, token_id=3, output_file=None):
    """
    Plot all 5 detection samples for a specific token
    Shows how the cache pattern varies across different detections
    """
    print(f"\nPlotting 5 detection samples for Token {token_id}...")
    
    # Filter data for the specific token
    token_data = df[df['TokenIndex'] == token_id]
    
    if token_data.empty:
        print(f"No data found for token {token_id}")
        return
    
    # Get unique detection indices
    detections = sorted(token_data['DetectionIndex'].unique())
    n_detections = len(detections)
    
    if n_detections == 0:
        print(f"No detections found for token {token_id}")
        return
    
    # Create subplots for each detection
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Color map for different detections
    colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
    
    for idx, det_idx in enumerate(detections[:5]):  # Show max 5 detections
        ax = axes[idx]
        
        # Get data for this detection (average across burst iterations)
        det_data = token_data[token_data['DetectionIndex'] == det_idx]
        avg_per_cacheset = det_data.groupby('CacheSet')['Probe'].mean().sort_index()
        
        # Plot bar chart
        cache_sets = avg_per_cacheset.index
        probe_times = avg_per_cacheset.values
        
        bars = ax.bar(cache_sets, probe_times, color=colors[idx % len(colors)], alpha=0.7)
        
        # Highlight significant spikes (e.g., > 1.5x median)
        median_probe = np.median(probe_times)
        threshold = median_probe * 1.5
        
        for i, (cs, pt) in enumerate(zip(cache_sets, probe_times)):
            if pt > threshold:
                bars[i].set_color('red')
                bars[i].set_alpha(1.0)
                ax.text(cs, pt + 1, f'{int(pt)}', ha='center', fontsize=8)
        
        ax.set_title(f'Detection {det_idx} (Token {token_id})')
        ax.set_xlabel('Cache Set')
        ax.set_ylabel('Avg Probe Time (cycles)')
        ax.set_ylim([0, max(probe_times) * 1.2])
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text(0.02, 0.98, f'Max: {max(probe_times):.0f}\nMean: {np.mean(probe_times):.1f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Use the last subplot for combined view
    ax = axes[5]
    for idx, det_idx in enumerate(detections[:5]):
        det_data = token_data[token_data['DetectionIndex'] == det_idx]
        avg_per_cacheset = det_data.groupby('CacheSet')['Probe'].mean().sort_index()
        ax.plot(avg_per_cacheset.index, avg_per_cacheset.values, 
                label=f'Detection {det_idx}', color=colors[idx % len(colors)], 
                alpha=0.7, linewidth=2)
    
    ax.set_title(f'All Detections Overlay (Token {token_id})')
    ax.set_xlabel('Cache Set')
    ax.set_ylabel('Avg Probe Time (cycles)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Token {token_id}: Cache Access Patterns Across 5 Detections', fontsize=16)
    plt.tight_layout()
    
    # Save to file
    if output_file is None:
        output_file = f'token_{token_id}_detections.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved detection plot to {output_file}")
    plt.show()
    
    # Print statistics about variability
    print(f"\nVariability Analysis for Token {token_id}:")
    for cs in range(64):  # Assuming 64 cache sets
        cs_data = token_data[token_data['CacheSet'] == cs]['Probe'].values
        if len(cs_data) > 0:
            std_dev = np.std(cs_data)
            if std_dev > 10:  # High variability threshold
                print(f"  Cache Set {cs}: mean={np.mean(cs_data):.1f}, std={std_dev:.1f} (HIGH VARIABILITY)")

def plot_token_burst_timeline(df, token_id=3, detection_idx=0, output_file=None):
    """
    Plot the timeline of probe measurements within a single detection burst
    Shows all 30 burst iterations for one detection
    """
    print(f"\nPlotting burst timeline for Token {token_id}, Detection {detection_idx}...")
    
    # Filter for specific token and detection
    burst_data = df[(df['TokenIndex'] == token_id) & (df['DetectionIndex'] == detection_idx)]
    
    if burst_data.empty:
        print(f"No data found for token {token_id}, detection {detection_idx}")
        return
    
    # Create heatmap of burst iterations vs cache sets
    pivot_data = burst_data.pivot_table(values='Probe', 
                                        index='BurstIteration', 
                                        columns='CacheSet', 
                                        aggfunc='mean')
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_data, cmap='hot', cbar_kws={'label': 'Probe Time (cycles)'})
    plt.title(f'Token {token_id} - Detection {detection_idx}: Burst Timeline Heatmap')
    plt.xlabel('Cache Set')
    plt.ylabel('Burst Iteration (0-29)')
    plt.tight_layout()
    
    if output_file is None:
        output_file = f'token_{token_id}_burst_timeline.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved burst timeline to {output_file}")
    plt.show()

def create_token_fingerprints(df, method='mean'):
    """
    Create a fingerprint for each token based on cache measurements
    method: 'mean', 'median', or 'max' across all detections and bursts
    """
    print(f"\nCreating token fingerprints using {method} method...")
    
    # Group by token and cache set, aggregate probe times
    if method == 'mean':
        fingerprints = df.groupby(['TokenIndex', 'CacheSet'])['Probe'].mean()
    elif method == 'median':
        fingerprints = df.groupby(['TokenIndex', 'CacheSet'])['Probe'].median()
    elif method == 'max':
        fingerprints = df.groupby(['TokenIndex', 'CacheSet'])['Probe'].max()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reshape to have tokens as rows and cache sets as columns
    fingerprints = fingerprints.unstack(fill_value=0)
    
    print(f"Created fingerprints for {len(fingerprints)} tokens")
    print(f"Each fingerprint has {len(fingerprints.columns)} cache sets")
    
    return fingerprints

def calculate_similarity_matrix(fingerprints, metric='correlation'):
    """
    Calculate similarity between all pairs of tokens
    metric: 'correlation', 'cosine', 'euclidean'
    """
    print(f"\nCalculating similarity matrix using {metric}...")
    
    n_tokens = len(fingerprints)
    similarity_matrix = np.zeros((n_tokens, n_tokens))
    
    for i in range(n_tokens):
        for j in range(n_tokens):
            vec1 = fingerprints.iloc[i].values
            vec2 = fingerprints.iloc[j].values
            
            if metric == 'correlation':
                # Pearson correlation (1 = identical, -1 = opposite)
                if np.std(vec1) > 0 and np.std(vec2) > 0:
                    similarity_matrix[i, j] = pearsonr(vec1, vec2)[0]
                else:
                    similarity_matrix[i, j] = 0
                    
            elif metric == 'cosine':
                # Cosine similarity (1 = identical, 0 = orthogonal)
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    similarity_matrix[i, j] = 1 - cosine(vec1, vec2)
                else:
                    similarity_matrix[i, j] = 0
                    
            elif metric == 'euclidean':
                # Euclidean distance (0 = identical, larger = more different)
                # We'll convert to similarity by using 1/(1+distance)
                dist = euclidean(vec1, vec2)
                similarity_matrix[i, j] = 1 / (1 + dist)
                
    return similarity_matrix

def find_most_similar_tokens(similarity_matrix, token_indices):
    """
    For each token, find the most similar other token
    """
    results = {}
    
    for i, token_id in enumerate(token_indices):
        # Get similarities for this token
        similarities = similarity_matrix[i, :]
        
        # Set self-similarity to -inf to exclude it
        similarities[i] = -np.inf
        
        # Find the most similar token
        most_similar_idx = np.argmax(similarities)
        most_similar_token = token_indices[most_similar_idx]
        similarity_score = similarities[most_similar_idx]
        
        # Find top 5 most similar tokens
        top5_indices = np.argsort(similarities)[-5:][::-1]
        top5_tokens = [(token_indices[idx], similarities[idx]) for idx in top5_indices]
        
        results[token_id] = {
            'most_similar': most_similar_token,
            'similarity_score': similarity_score,
            'top5': top5_tokens
        }
    
    return results

def print_similarity_report(results, n_tokens=None):
    """
    Print a formatted report of token similarities
    """
    print("\n" + "="*80)
    print("TOKEN SIMILARITY ANALYSIS")
    print("="*80)
    
    # If n_tokens specified, only show first n
    tokens_to_show = list(results.keys())[:n_tokens] if n_tokens else list(results.keys())
    
    for token_id in tokens_to_show:
        info = results[token_id]
        print(f"\nToken {token_id:3d}:")
        print(f"  Most similar: Token {info['most_similar']:3d} (similarity: {info['similarity_score']:.4f})")
        print(f"  Top 5 similar tokens:")
        for similar_token, score in info['top5']:
            print(f"    - Token {similar_token:3d}: {score:.4f}")
    
    # Find tokens that are most similar to each other (mutual pairs)
    print("\n" + "-"*80)
    print("MUTUAL SIMILARITY PAIRS (tokens that are most similar to each other):")
    print("-"*80)
    
    mutual_pairs = []
    for token_id, info in results.items():
        most_similar = info['most_similar']
        # Check if the most similar token also considers this token as most similar
        if results[most_similar]['most_similar'] == token_id and token_id < most_similar:
            mutual_pairs.append((token_id, most_similar, info['similarity_score']))
    
    if mutual_pairs:
        for token1, token2, score in sorted(mutual_pairs, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  Tokens {token1:3d} <-> {token2:3d}: similarity = {score:.4f}")
    else:
        print("  No mutual pairs found")

def plot_similarity_heatmap(similarity_matrix, token_indices, output_file='similarity_heatmap.png'):
    """
    Create a heatmap visualization of token similarities
    """
    print(f"\nCreating similarity heatmap...")
    
    # Limit to first 50 tokens for visibility
    n_show = min(50, len(token_indices))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix[:n_show, :n_show], 
                xticklabels=token_indices[:n_show],
                yticklabels=token_indices[:n_show],
                cmap='coolwarm',
                center=0,
                cbar_kws={'label': 'Similarity'})
    
    plt.title(f'Token Similarity Matrix (First {n_show} tokens)')
    plt.xlabel('Token ID')
    plt.ylabel('Token ID')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_file}")
    plt.show()

def main():
    # Configuration - UPDATE THIS PATH
    csv_file = "results/bens_64_tokens_result.csv"  # Change to your file
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found!")
        print("Please update the csv_file path in the script.")
        return
    
    # Load data
    df = load_cache_data(csv_file)
    
    # NEW: Plot detection samples for token 3
    plot_token_detections(df, token_id=18)
    
    # NEW: Plot burst timeline for token 3
    plot_token_burst_timeline(df, token_id=18, detection_idx=0)
    
    # You can also plot other tokens
    # plot_token_detections(df, token_id=5)
    # plot_token_detections(df, token_id=10)
    
    # Continue with original analysis...
    fingerprints = create_token_fingerprints(df, method='mean')
    similarity_matrix = calculate_similarity_matrix(fingerprints, metric='correlation')
    token_indices = fingerprints.index.tolist()
    results = find_most_similar_tokens(similarity_matrix, token_indices)
    print_similarity_report(results, n_tokens=20)
    plot_similarity_heatmap(similarity_matrix, token_indices)
    
    # Save results
    print("\nSaving detailed results to CSV...")
    similarity_df = pd.DataFrame([
        {
            'token_id': token_id,
            'most_similar_token': info['most_similar'],
            'similarity_score': info['similarity_score'],
            'top5_similar': str(info['top5'])
        }
        for token_id, info in results.items()
    ])
    similarity_df.to_csv('token_similarity_results.csv', index=False)
    print("Saved to token_similarity_results.csv")

if __name__ == "__main__":
    main()
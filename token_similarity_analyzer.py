#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json
from scipy.spatial.distance import jaccard, cosine
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class TokenSimilarityAnalyzer:
    def __init__(self, results_dir="token_analysis_results"):
        self.results_dir = Path(results_dir)
        self.token_data = {}
        self.similarity_matrix = None
        self.df = None
        
    def load_data(self):
        """Load token analysis results from CSV and individual files"""
        
        print("📂 Loading token analysis data...")
        
        # Load from CSV if available
        csv_file = self.results_dir / "token_cache_mapping.csv"
        if csv_file.exists():
            self.df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(self.df)} tokens from CSV")
        else:
            print("❌ CSV file not found, loading from individual files...")
            self._load_from_individual_files()
            
        # Process cache sets into binary vectors
        self._process_cache_vectors()
        
    def _load_from_individual_files(self):
        """Fallback: load from individual analysis files"""
        
        token_files = list(self.results_dir.glob("token_*_analysis.txt"))
        data = []
        
        for file_path in token_files:
            token_id = int(file_path.stem.split('_')[1])
            
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract cache sets
            cache_line = [line for line in content.split('\n') if '[' in line and ']' in line]
            if cache_line:
                cache_sets_str = cache_line[0].strip('[]').replace(' ', '')
                if cache_sets_str:
                    cache_sets = [int(x) for x in cache_sets_str.split(',')]
                    cache_sets_str = '|'.join(map(str, cache_sets))
                    data.append({
                        'token_id': token_id,
                        'cache_sets': cache_sets_str,
                        'num_cache_sets': len(cache_sets)
                    })
        
        self.df = pd.DataFrame(data)
        print(f"✓ Loaded {len(self.df)} tokens from individual files")
    
    def _process_cache_vectors(self):
        """Convert cache sets to binary vectors for similarity analysis"""
        
        print("🔄 Processing cache vectors...")
        
        # Create binary vectors (64 cache sets)
        vectors = []
        token_ids = []
        
        for _, row in self.df.iterrows():
            token_id = row['token_id']
            cache_sets_str = row['cache_sets']
            
            if pd.isna(cache_sets_str) or cache_sets_str == '':
                continue
                
            # Parse cache sets
            cache_sets = [int(x) for x in cache_sets_str.split('|') if x.strip()]
            
            # Create binary vector
            vector = np.zeros(64)
            for cache_set in cache_sets:
                if 0 <= cache_set < 64:
                    vector[cache_set] = 1
                    
            vectors.append(vector)
            token_ids.append(token_id)
            self.token_data[token_id] = cache_sets
        
        self.vectors = np.array(vectors)
        self.token_ids = np.array(token_ids)
        
        print(f"✓ Processed {len(self.vectors)} token vectors")
    
    def calculate_similarity_matrix(self):
        """Calculate pairwise similarity between all tokens"""
        
        print("🧮 Calculating similarity matrix...")
        
        n_tokens = len(self.vectors)
        self.similarity_matrix = np.zeros((n_tokens, n_tokens))
        
        for i in range(n_tokens):
            for j in range(n_tokens):
                if i == j:
                    self.similarity_matrix[i, j] = 1.0
                else:
                    # Jaccard similarity for binary vectors
                    # Jaccard = intersection / union
                    intersection = np.sum(self.vectors[i] & self.vectors[j])
                    union = np.sum(self.vectors[i] | self.vectors[j])
                    
                    if union == 0:
                        jaccard_sim = 1.0
                    else:
                        jaccard_sim = intersection / union
                    
                    self.similarity_matrix[i, j] = jaccard_sim
        
        print("✓ Similarity matrix calculated")
    
    def find_top_similar_tokens(self, target_token, top_k=5):
        """Find the top K most similar tokens to a target token"""
        
        if target_token not in self.token_ids:
            print(f"❌ Token {target_token} not found in dataset")
            return []
        
        target_idx = np.where(self.token_ids == target_token)[0][0]
        similarities = self.similarity_matrix[target_idx]
        
        # Get indices of top similar tokens (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in similar_indices:
            similar_token = self.token_ids[idx]
            similarity_score = similarities[idx]
            
            # Calculate detailed metrics
            target_sets = set(self.token_data[target_token])
            similar_sets = set(self.token_data[similar_token])
            
            intersection = target_sets & similar_sets
            union = target_sets | similar_sets
            
            results.append({
                'token_id': similar_token,
                'similarity': similarity_score,
                'intersection': list(intersection),
                'union': list(union),
                'target_sets': list(target_sets),
                'similar_sets': list(similar_sets),
                'overlap_count': len(intersection),
                'target_unique': list(target_sets - similar_sets),
                'similar_unique': list(similar_sets - target_sets)
            })
        
        return results
    
    def analyze_all_similarities(self):
        """Analyze similarity patterns across all tokens"""
        
        print("📊 Analyzing similarity patterns...")
        
        # Find all pairwise similarities
        similarities = []
        
        for i in range(len(self.token_ids)):
            for j in range(i+1, len(self.token_ids)):
                similarity = self.similarity_matrix[i, j]
                similarities.append({
                    'token1': self.token_ids[i],
                    'token2': self.token_ids[j],
                    'similarity': similarity
                })
        
        sim_df = pd.DataFrame(similarities)
        
        # Statistics
        stats = {
            'mean_similarity': sim_df['similarity'].mean(),
            'std_similarity': sim_df['similarity'].std(),
            'max_similarity': sim_df['similarity'].max(),
            'min_similarity': sim_df['similarity'].min(),
            'median_similarity': sim_df['similarity'].median()
        }
        
        # Find most similar pairs
        most_similar = sim_df.nlargest(10, 'similarity')
        
        # Find least similar pairs  
        least_similar = sim_df.nsmallest(10, 'similarity')
        
        return stats, most_similar, least_similar
    
    def plot_similarity_heatmap(self, max_tokens=50):
        """Create a heatmap of token similarities"""
        
        print("🎨 Creating similarity heatmap...")
        
        # Limit to first N tokens for readability
        n_show = min(max_tokens, len(self.token_ids))
        indices = np.argsort(self.token_ids)[:n_show]
        
        subset_matrix = self.similarity_matrix[np.ix_(indices, indices)]
        subset_labels = self.token_ids[indices]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(subset_matrix, 
                   xticklabels=subset_labels,
                   yticklabels=subset_labels,
                   cmap='coolwarm',
                   center=0.5,
                   cbar_kws={'label': 'Jaccard Similarity'})
        
        plt.title(f'Token Similarity Heatmap (First {n_show} tokens)')
        plt.xlabel('Token ID')
        plt.ylabel('Token ID')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'similarity_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_similarity_distribution(self):
        """Plot distribution of similarity scores"""
        
        print("📈 Creating similarity distribution plot...")
        
        # Extract upper triangle of similarity matrix (excluding diagonal)
        upper_triangle = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.hist(upper_triangle, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(upper_triangle), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(upper_triangle):.3f}')
        plt.axvline(np.median(upper_triangle), color='green', linestyle='--',
                   label=f'Median: {np.median(upper_triangle):.3f}')
        
        plt.xlabel('Jaccard Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Token Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'similarity_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_cache_set_usage(self):
        """Plot which cache sets are most commonly used"""
        
        print("📊 Creating cache set usage plot...")
        
        cache_usage = np.zeros(64)
        for token_id, cache_sets in self.token_data.items():
            for cache_set in cache_sets:
                cache_usage[cache_set] += 1
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(64), cache_usage)
        
        # Highlight most/least used
        max_idx = np.argmax(cache_usage)
        bars[max_idx].set_color('red')
        
        plt.xlabel('Cache Set')
        plt.ylabel('Usage Count')
        plt.title('Cache Set Usage Frequency Across All Tokens')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.text(0.02, 0.98, f'Most used: Set {max_idx} ({cache_usage[max_idx]:.0f} times)\n'
                              f'Mean usage: {np.mean(cache_usage):.1f}\n'
                              f'Std usage: {np.std(cache_usage):.1f}',
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'cache_set_usage.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_token_clustering(self):
        """Create t-SNE visualization and clustering"""
        
        print("🔬 Creating token clustering visualization...")
        
        if len(self.vectors) < 10:
            print("❌ Not enough tokens for clustering")
            return
        
        # t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.vectors)-1))
        coords_2d = tsne.fit_transform(self.vectors)
        
        # K-means clustering
        n_clusters = min(8, len(self.vectors) // 3)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.vectors)
        else:
            cluster_labels = np.zeros(len(self.vectors))
        
        plt.figure(figsize=(12, 8))
        
        # Plot with clusters
        scatter = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7)
        
        # Add token labels for a subset
        for i in range(0, len(self.token_ids), max(1, len(self.token_ids) // 20)):
            plt.annotate(str(self.token_ids[i]), 
                        (coords_2d[i, 0], coords_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(f'Token Clustering (t-SNE) - {n_clusters} clusters')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'token_clustering.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, target_tokens=None):
        """Generate a detailed similarity analysis report"""
        
        print("📝 Generating detailed similarity report...")
        
        if target_tokens is None:
            # Analyze first 10 tokens as examples
            target_tokens = sorted(self.token_ids)[:10]
        
        report_file = self.results_dir / 'detailed_similarity_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("TOKEN SIMILARITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            stats, most_similar, least_similar = self.analyze_all_similarities()
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("MOST SIMILAR TOKEN PAIRS:\n")
            f.write("-" * 25 + "\n")
            for _, row in most_similar.iterrows():
                f.write(f"Tokens {row['token1']} <-> {row['token2']}: {row['similarity']:.4f}\n")
            f.write("\n")
            
            f.write("LEAST SIMILAR TOKEN PAIRS:\n")
            f.write("-" * 26 + "\n")
            for _, row in least_similar.iterrows():
                f.write(f"Tokens {row['token1']} <-> {row['token2']}: {row['similarity']:.4f}\n")
            f.write("\n")
            
            # Detailed analysis for target tokens
            f.write("DETAILED ANALYSIS FOR SELECTED TOKENS:\n")
            f.write("-" * 40 + "\n\n")
            
            for token in target_tokens:
                if token in self.token_ids:
                    f.write(f"TOKEN {token} ANALYSIS:\n")
                    f.write("-" * 20 + "\n")
                    
                    similar_tokens = self.find_top_similar_tokens(token, top_k=5)
                    
                    f.write(f"Cache sets: {self.token_data[token]}\n")
                    f.write(f"Number of cache sets: {len(self.token_data[token])}\n\n")
                    
                    f.write("Top 5 most similar tokens:\n")
                    for i, sim_token in enumerate(similar_tokens, 1):
                        f.write(f"  {i}. Token {sim_token['token_id']} "
                               f"(similarity: {sim_token['similarity']:.4f})\n")
                        f.write(f"     Cache sets: {sim_token['similar_sets']}\n")
                        f.write(f"     Overlap: {sim_token['intersection']} "
                               f"({sim_token['overlap_count']} sets)\n")
                        f.write(f"     Unique to {token}: {sim_token['target_unique']}\n")
                        f.write(f"     Unique to {sim_token['token_id']}: {sim_token['similar_unique']}\n\n")
                    
                    f.write("\n")
        
        print(f"✓ Detailed report saved to {report_file}")
    
    def run_complete_analysis(self, target_tokens=None):
        """Run the complete similarity analysis"""
        
        print("🚀 Starting complete token similarity analysis\n")
        
        # Load data
        self.load_data()
        
        if len(self.token_data) == 0:
            print("❌ No token data found!")
            return
        
        # Calculate similarities
        self.calculate_similarity_matrix()
        
        # Generate visualizations
        self.plot_similarity_heatmap()
        self.plot_similarity_distribution()
        self.plot_cache_set_usage()
        self.plot_token_clustering()
        
        # Generate reports
        self.generate_detailed_report(target_tokens)
        
        # Interactive query
        self.interactive_similarity_query()
        
        print("\n🎉 Analysis complete! Check the results directory for all outputs.")
    
    def interactive_similarity_query(self):
        """Interactive mode to query specific token similarities"""
        
        print("\n🔍 INTERACTIVE SIMILARITY QUERY")
        print("Enter a token ID to find its most similar tokens (or 'quit' to exit)")
        
        while True:
            try:
                user_input = input("\nEnter token ID: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                token_id = int(user_input)
                
                if token_id not in self.token_ids:
                    print(f"❌ Token {token_id} not found in dataset")
                    print(f"Available tokens: {sorted(self.token_ids)[:10]}...")
                    continue
                
                similar_tokens = self.find_top_similar_tokens(token_id, top_k=5)
                
                print(f"\n📊 Token {token_id} Analysis:")
                print(f"Cache sets: {self.token_data[token_id]}")
                print(f"\nTop 5 most similar tokens:")
                
                for i, sim_token in enumerate(similar_tokens, 1):
                    print(f"  {i}. Token {sim_token['token_id']} "
                          f"(similarity: {sim_token['similarity']:.4f})")
                    print(f"     Cache sets: {sim_token['similar_sets']}")
                    print(f"     Overlap: {sim_token['intersection']} "
                          f"({sim_token['overlap_count']} sets)")
                
            except ValueError:
                print("❌ Please enter a valid integer token ID")
            except KeyboardInterrupt:
                break

def main():
    analyzer = TokenSimilarityAnalyzer()
    
    # Run complete analysis for ALL 129 tokens
    analyzer.run_complete_analysis(save_plots=True)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Individual Detection Event Visualization
Creates separate plots for each detection event to find signal patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Ground truth cache sets for tokens 0-9
GROUND_TRUTH = {
    0:  [41,39,36,0,18,58,60,60,0,4],
    1:  [41,39,36,0,18,58,60,60,1,9],
    2:  [41,39,36,0,18,58,60,60,2,17],
    3:  [41,39,36,0,18,58,60,60,2,34],
    4:  [41,39,36,0,18,58,60,60,3,17],
    5:  [41,39,36,0,18,58,60,60,3,9],
    6:  [41,39,36,0,18,58,60,60,4,37],
    7:  [41,39,36,0,18,58,60,60,5,58],
    8:  [41,39,36,0,18,58,60,60,5,56],
    9:  [41,39,36,0,18,58,60,60,6,58]
}

def create_individual_detection_plots(csv_file):
    """Create individual heatmaps for each detection event"""
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} measurements")
    print(f"Tokens: {sorted(df['TokenIndex'].unique())}")
    print(f"Detection events per token: {df['DetectionIndex'].nunique()}")
    
    # Create plots directory
    plots_dir = Path("individual_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Track signal strength for each individual detection
    signal_strengths = []
    
    # Process each token and detection event
    for token_idx in sorted(df['TokenIndex'].unique()):
        for det_idx in sorted(df[df['TokenIndex'] == token_idx]['DetectionIndex'].unique()):
            
            # Filter data for this specific detection event
            event_data = df[(df['TokenIndex'] == token_idx) & (df['DetectionIndex'] == det_idx)]
            
            if event_data.empty:
                continue
            
            # Create pivot table for heatmap
            pivot_data = event_data.pivot_table(
                values='Probe',
                index='CacheSet',
                columns='BurstIteration',
                aggfunc='first'  # No averaging - raw measurements
            )
            
            # Calculate signal strength for this individual detection
            if token_idx in GROUND_TRUTH:
                expected_sets = set(GROUND_TRUTH[token_idx])
                expected_data = event_data[event_data['CacheSet'].isin(expected_sets)]
                other_data = event_data[~event_data['CacheSet'].isin(expected_sets)]
                
                if not expected_data.empty and not other_data.empty:
                    signal_strength = expected_data['Probe'].mean() - other_data['Probe'].mean()
                    signal_strengths.append({
                        'token': token_idx,
                        'detection': det_idx,
                        'signal_strength': signal_strength,
                        'expected_mean': expected_data['Probe'].mean(),
                        'other_mean': other_data['Probe'].mean()
                    })
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, 
                       cmap='Blues', 
                       cbar_kws={'label': 'Access Time (cycles)'},
                       fmt='.0f')
            
            plt.title(f'Token {token_idx} (ID {token_idx + 100}) - Detection {det_idx}\n' + 
                     f'Expected sets: {GROUND_TRUTH.get(token_idx, "Unknown")}')
            plt.xlabel('Burst Iteration')
            plt.ylabel('Cache Set')
            
            # Highlight expected cache sets
            if token_idx in GROUND_TRUTH:
                expected_sets = set(GROUND_TRUTH[token_idx])
                ax = plt.gca()
                
                for cache_set in expected_sets:
                    if cache_set in pivot_data.index:
                        y_pos = list(pivot_data.index).index(cache_set)
                        ax.add_patch(plt.Rectangle((0, y_pos), len(pivot_data.columns), 1,
                                                 fill=False, edgecolor='red', linewidth=2))
            
            # Save plot
            filename = f"individual_plots/token_{token_idx}_det_{det_idx}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created: {filename}")
    
    # Analyze and report best detection events
    if signal_strengths:
        signal_df = pd.DataFrame(signal_strengths)
        
        print(f"\n=== Individual Detection Signal Analysis ===")
        print(f"Total detection events: {len(signal_df)}")
        
        # Find best signals
        best_signals = signal_df.nlargest(10, 'signal_strength')
        print("\nTop 10 strongest signals:")
        for _, row in best_signals.iterrows():
            print(f"Token {row['token']}, Det {row['detection']}: {row['signal_strength']:.2f} cycles "
                  f"(exp: {row['expected_mean']:.1f}, other: {row['other_mean']:.1f})")
        
        # Find worst signals (most negative)
        worst_signals = signal_df.nsmallest(5, 'signal_strength')
        print("\nWorst signals (most negative):")
        for _, row in worst_signals.iterrows():
            print(f"Token {row['token']}, Det {row['detection']}: {row['signal_strength']:.2f} cycles")
        
        # Statistics
        print(f"\nSignal Statistics:")
        print(f"Mean signal strength: {signal_df['signal_strength'].mean():.2f} cycles")
        print(f"Std deviation: {signal_df['signal_strength'].std():.2f} cycles")
        print(f"Positive signals: {(signal_df['signal_strength'] > 0).sum()}/{len(signal_df)}")
        print(f"Strong signals (>2 cycles): {(signal_df['signal_strength'] > 2).sum()}/{len(signal_df)}")
        
        # Create summary plot showing signal strength distribution
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Signal strength per detection event
        plt.subplot(1, 2, 1)
        x = range(len(signal_df))
        colors = ['red' if s > 2 else 'orange' if s > 0 else 'blue' for s in signal_df['signal_strength']]
        plt.scatter(x, signal_df['signal_strength'], c=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Strong signal threshold')
        plt.xlabel('Detection Event')
        plt.ylabel('Signal Strength (cycles)')
        plt.title('Signal Strength per Detection Event')
        plt.legend()
        
        # Plot 2: Signal strength histogram
        plt.subplot(1, 2, 2)
        plt.hist(signal_df['signal_strength'], bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='Strong signal')
        plt.xlabel('Signal Strength (cycles)')
        plt.ylabel('Frequency')
        plt.title('Signal Strength Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("individual_plots/signal_strength_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: individual_plots/signal_strength_summary.png")
    
    print(f"\nAll plots saved to: {plots_dir}")
    return signal_df if signal_strengths else None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python individual_detection_viz.py <csv_file>")
        sys.exit(1)
    
    signal_df = create_individual_detection_plots(sys.argv[1])
    
    print("\nRecommendations:")
    if signal_df is not None and len(signal_df) > 0:
        strong_signals = signal_df[signal_df['signal_strength'] > 2]
        if len(strong_signals) > 0:
            print("✅ Found some strong signals! Focus on these detection events for optimization.")
            print("✅ Look for timing patterns in the best detection event plots.")
        else:
            moderate_signals = signal_df[signal_df['signal_strength'] > 0.5]
            if len(moderate_signals) > 0:
                print("⚠️  Found moderate signals. Try:")
                print("   - Adding delay after FR detection (delayloop)")
                print("   - Reducing BURST_WINDOW to focus on peak activity")
                print("   - Verify ground truth cache sets are correct")
            else:
                print("❌ Very weak signals. Consider:")
                print("   - Verify FR is detecting correct function calls")
                print("   - Check if ground truth cache sets are wrong")
                print("   - Try monitoring different function offset")
    
    print("\n🔍 Examine individual plots for patterns - look for detection events with")
    print("   clear dark blue regions (high access times) in expected cache sets!")
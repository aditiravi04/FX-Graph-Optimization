import pandas as pd
import matplotlib.pyplot as plt

def generate_report():
    # Load data saved by the benchmark script
    try:
        df = pd.read_csv('benchmark_results.csv')
    except FileNotFoundError:
        print("Error: benchmark_results.csv not found. Run resnet_benchmark.py first.")
        return

    # CORRECT EXTRACTION: Use .iloc to get the actual scalar value
    eager = df['Eager_ms'].iloc[0]
    fused = df['Fused_ms'].iloc[0]
    speedup = df['Speedup'].iloc[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    # Now [eager, fused] contains two numbers, which Matplotlib can handle
    bars = plt.bar(['Eager Mode', 'Fused Mode'], [eager, fused], color=['#95a5a6', '#2980b9'])
    
    
    # Labeling
    plt.ylabel('Latency (ms)')
    plt.title(f'Phase 4: ResNet18 Optimization Performance (Speedup: {speedup}x)')
    
    # Annotate bars with exact ms values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height}ms', ha='center', va='bottom', fontweight='bold')

    # Save final artifact
    plt.tight_layout()
    plt.savefig('final_performance_report.png')
    print("Report generated: final_performance_report.png")
    plt.show()

if __name__ == "__main__":
    generate_report()
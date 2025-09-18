#!/bin/bash

echo "================================"
echo "Running 10-Token Test Attack"
echo "================================"

# Make sure results directory exists
mkdir -p results

# Clean and rebuild
cd src
echo "[*] Cleaning old build..."
make clean

echo "[*] Building orchestrator and attacker..."
make

if [ $? -ne 0 ]; then
    echo "[!] Build failed!"
    exit 1
fi

cd ../

echo "[*] Starting attack on 10 tokens..."
echo "[*] Output will be saved to: results/bens_10_tokens_result.csv"

# Run the orchestrator on CPU 0
taskset -c 0 ./bin/orchastractor

echo ""
echo "[*] Attack complete!"
echo "[*] Checking output file..."

if [ -f "results/bens_10_tokens_result.csv" ]; then
    lines=$(wc -l < "results/bens_10_tokens_result.csv")
    echo "[✓] Output file created with $lines lines"
    echo ""
    echo "First few lines of output:"
    head -n 10 results/bens_10_tokens_result.csv
else
    echo "[!] Output file not found!"
fi

echo "================================"
#!/bin/bash

# Set the base directory
BASE_DIR="/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot/analysis_results"

# Create necessary directories
mkdir -p "$BASE_DIR/classification"
mkdir -p "$BASE_DIR/generation"

# Move generation files (already done, but keeping for reference)
# mv "$BASE_DIR/"*.csv "$BASE_DIR/generation/" 2>/dev/null
# mv "$BASE_DIR/"*.tex "$BASE_DIR/generation/" 2>/dev/null

# Create a README.md file
cat > "$BASE_DIR/README.md" << 'EOL'
# Analysis Results

This directory contains the results of the federated learning experiments, organized by task type.

## Directory Structure

- `classification/`: Contains results from classification tasks
  - `centralized/`: Centralized training results
  - `federated/`: Federated learning results
  - `comparison/`: Comparison between centralized and federated results

- `generation/`: Contains results from text generation tasks
  - `centralized/`: Centralized training results
  - `federated/`: Federated learning results
  - `comparison/`: Comparison between centralized and federated results

## Notes

- All metrics are stored in CSV format
- LaTeX tables are provided for easy inclusion in papers
- The `comparison` directories contain combined results and analysis
EOL

echo "Directory structure has been organized. Please move your classification results to the classification/ directory."
echo "Generation results are already in the generation/ directory."

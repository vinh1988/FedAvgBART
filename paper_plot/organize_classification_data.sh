#!/bin/bash

# Set the base directory
BASE_DIR="/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot/data_to_plot"

# Create the new directory structure
mkdir -p "$BASE_DIR/classification/centralized"
mkdir -p "$BASE_DIR/classification/federated"
mkdir -p "$BASE_DIR/classification/summary"

# Move files to their new locations
mv "$BASE_DIR/classification/central"/* "$BASE_DIR/classification/centralized/" 2>/dev/null
mv "$BASE_DIR/classification/federated" "$BASE_DIR/classification/federated/raw" 2>/dev/null
mv "$BASE_DIR/classification/centralized_results.csv" "$BASE_DIR/classification/summary/" 2>/dev/null
mv "$BASE_DIR/classification/federated_results_raw.csv" "$BASE_DIR/classification/federated/" 2>/dev/null
mv "$BASE_DIR/classification/federated_results_stats.csv" "$BASE_DIR/classification/summary/" 2>/dev/null

# Create a README.md file
cat > "$BASE_DIR/classification/README.md" << 'EOL'
# Classification Data

This directory contains the data files for classification task visualization.

## Directory Structure

- `centralized/`: Contains centralized training results
  - Raw metrics and model outputs

- `federated/`: Contains federated learning results
  - `raw/`: Raw metrics from each client and round
  - `federated_results_raw.csv`: Combined raw results

- `summary/`: Contains aggregated and processed results
  - `centralized_results.csv`: Summary of centralized training
  - `federated_results_stats.csv`: Aggregated federated results

## Notes

- All metrics are stored in CSV format
- The `summary` directory contains pre-processed data ready for visualization
- Raw data is preserved in their respective directories for reference
EOL

echo "Classification data has been organized successfully."

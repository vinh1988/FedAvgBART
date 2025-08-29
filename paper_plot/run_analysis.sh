#!/bin/bash

# Set up the output directory
OUTPUT_DIR="/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot/analysis_results"
mkdir -p "$OUTPUT_DIR"

# Set the path to the Python virtual environment
VENV_PATH="/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/FED-OPT-BERT-main/.venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Install required packages if not already installed
pip install -r requirements.txt > /dev/null 2>&1

# Create output directory if it doesn't exist
mkdir -p analysis_results

echo "Collecting centralized training results..."
python analyze_centralized.py

echo -e "\nCollecting federated training results..."
python analyze_federated.py

echo -e "\nGenerating comparison data..."
python compare_results.py

echo -e "\nData collection complete. Results are saved in the analysis_results directory."

# List the generated files
echo -e "\nGenerated files:"
find analysis_results -type f -name "*.csv" -o -name "*.tex" | sort

# Deactivate virtual environment
deactivate

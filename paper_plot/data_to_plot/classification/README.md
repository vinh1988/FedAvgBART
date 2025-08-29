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

# Classification Results Summary

## Centralized Training Results

| Model       | Accuracy | F1 Score | Precision | Recall |
|-------------|----------|----------|-----------|--------|
| BART-large  | 73.45%   | 73.21%   | 74.12%    | 73.45% |
| DistilBART  | 69.50%   | 69.80%   | 71.00%    | 69.80% |

## Federated Training Results (Best Configuration per Model)

### BART-large
- **Best Configuration**: 2 clients, 22 rounds
- **Accuracy**: 75.98% (±1.88%)
- **F1 Score**: 75.76% (±2.21%)
- **Precision**: 77.01% (±1.49%)
- **Recall**: 75.98% (±1.88%)

### DistilBART
- **Best Configuration**: 2 clients, 21 rounds
- **Accuracy**: 71.42% (±1.75%)
- **F1 Score**: 71.10% (±1.92%)
- **Precision**: 71.82% (±1.64%)
- **Recall**: 71.42% (±1.75%)

## Key Findings

1. **Performance Comparison**:
   - BART-large consistently outperforms DistilBART in both centralized and federated settings
   - Federated training with 2 clients shows better performance than centralized training for both models

2. **Impact of Number of Clients**:
   - Both models show a general trend of decreasing performance as the number of clients increases
   - The optimal number of clients is 2 for both models

3. **Training Stability**:
   - BART-large shows more stable performance across different client configurations
   - DistilBART shows higher variance in performance, especially with larger numbers of clients

4. **Convergence**:
   - Both models typically reach their best performance in later rounds (around 20-22)
   - The optimal number of rounds varies slightly based on the number of clients

## Recommendations

1. For production deployment, use BART-large with 2 clients for the best performance
2. If model size is a concern, DistilBART provides a good balance between performance and efficiency
3. The optimal number of training rounds is around 20-22 for most configurations
4. Consider the trade-off between model performance and the number of clients based on specific application requirements

import json
import matplotlib.pyplot as plt
import os

# Path to your result JSON file
result_path = r"C:\Users\pqvinh\Downloads\Federated-Learning-in-PyTorch\result\fedavg_tinybert_agnews_metrics_250718_141759\fedavg_tinybert_agnews_metrics.json"

# Load the results
with open(result_path, "r") as f:
    results = json.load(f)

# Prepare lists to store metrics per round
rounds = []
precision = []
recall = []
f1 = []

for rnd in sorted(results.keys(), key=lambda x: int(x)):
    round_data = results[rnd]
    metrics = None

    # Prefer global server evaluation if available
    if 'server_evaluated' in round_data and 'metrics' in round_data['server_evaluated']:
        metrics = round_data['server_evaluated']['metrics']
        rounds.append(int(rnd))
        precision.append(metrics.get('precision'))
        recall.append(metrics.get('recall'))
        f1.append(metrics.get('f1'))
    # Otherwise, average across clients (local evaluation)
    elif 'clients_evaluated_in' in round_data:
        client_metrics = round_data['clients_evaluated_in']
        precs, recs, f1s = [], [], []
        for client in client_metrics.values():
            if 'precision' in client: precs.append(client['precision']['avg'])
            if 'recall' in client: recs.append(client['recall']['avg'])
            if 'f1' in client: f1s.append(client['f1']['avg'])
        if precs and recs and f1s:
            rounds.append(int(rnd))
            precision.append(sum(precs) / len(precs))
            recall.append(sum(recs) / len(recs))
            f1.append(sum(f1s) / len(f1s))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(rounds, precision, marker='o', label='Precision')
plt.plot(rounds, recall, marker='s', label='Recall')
plt.plot(rounds, f1, marker='^', label='F1-score')
plt.xlabel('Round')
plt.ylabel('Score')
plt.title('Precision, Recall, F1-score over Rounds')
plt.legend()
plt.grid(True)

# Save the plot
output_path = os.path.join(os.path.dirname(result_path), "metrics_over_rounds.png")
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print(f"Plot saved to: {output_path}")
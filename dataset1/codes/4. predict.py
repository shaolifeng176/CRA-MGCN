import torch
import numpy as np
import pandas as pd
from model import GCN

def load_features(features_path):
    df = pd.read_csv(features_path, header=None, sep='\t')
    features = torch.FloatTensor(np.array(df.iloc[:, 1:], dtype=np.float32))
    return features

def load_adjacency(adj_path, num_nodes):
    edges = pd.read_csv(adj_path, header=None, sep='\t', dtype=np.int64)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges.values:
        adj[i, j] = 1
        adj[j, i] = 1
    adj = torch.FloatTensor(adj)
    return adj

def predict(model_path, features, adj, num_classes, threshold=0.5, temperature=1.0):
    nfeat = features.shape[1]
    nhid = 256
    dropout = 0.1
    model = GCN(nfeat=nfeat, nhid=nhid, nclass=num_classes, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(features, adj)
        probabilities = torch.softmax(output / temperature, dim=1)  # Apply temperature scaling
        predictions = (probabilities[:, 1] >= threshold).long()

    return predictions.numpy(), probabilities.numpy()

def save_predictions(predictions, probabilities, output_path):
    results = pd.DataFrame({
        'Index': range(len(predictions)),
        'Predicted Class': predictions,
        'Probability': probabilities.max(axis=1)
    })
    results.to_csv(output_path, index=False)

if __name__ == "__main__":
    for i in range(1, 644):
        features_path = f'D:\\Project\\pythonProject\\re\\keti\\dataset1\\dataset\\HCGCN-prediction set\\prediction set-{i}-.content'
        adj_path = f'D:\\Project\\pythonProject\\re\\keti\\dataset1\\dataset\\HCGCN-prediction set\\prediction set-{i}.cites'
        model_path = f'D:\\Project\\pythonProject\\re\\keti\\dataset1\\codes\\model.pth'
        output_path = f'D:\\Project\\pythonProject\\re\\keti\\dataset1\\codes\\predictions{i}.csv'

        features = load_features(features_path)
        num_nodes = features.shape[0]
        adj = load_adjacency(adj_path, num_nodes)

        num_classes = 2
        threshold = 0.8
        temperature = 30.0  # Set the temperature parameter

        predictions, probabilities = predict(model_path, features, adj, num_classes, threshold, temperature)

        save_predictions(predictions, probabilities, output_path)

        print(f"Predictions and probabilities for cora-zong-sum{i} have been saved to '{output_path}'.")

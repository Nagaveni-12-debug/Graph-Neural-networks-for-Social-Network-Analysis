# Graph Neural Network for Social Network Analysis

This project implements a Heterogeneous Graph Neural Network (GNN) for social network analysis, specifically focused on **Fake News Detection** and **User-Post Interaction Analysis**.

## Project Architecture
Following the proposed system architecture, the project includes:
1.  **Data Preprocessing**: Loading the **FakeNewsNet** dataset (Politifact/GossipCop), cleaning news titles, and extracting `tweet_ids` to form graph edges.
2.  **Heterogeneous Graph Construction**: Creating a graph with `News` nodes and `Tweet` nodes.
3.  **GNN Implementation**: A two-layer heterogeneous GNN using:
    - **SAGEConv** (representing GCN for bipartite graphs)
    - **GATConv** (Graph Attention Network)
4.  **Model Evaluation**: Precision, Recall, F1-Score, and Confusion Matrix.
5.  **Visualization**: Training/Validation metrics and Network structure plots.

## Files
- `preprocessing.py`: NLP and feature extraction.
- `graph_builder.py`: Constructs the `HeteroData` object.
- `gnn_model.py`: Defines the GNN architecture.
- `engine.py`: Main script for training and plotting.
- `visualize.py`: Graph visualization logic.
- `evaluation.py`: Performance metrics logic.

## Generated Graphs
- `training_validation_graph.png`: Accuracy and Loss over epochs.
- `network_graph.png`: Visualization of the User-Post interaction network.
- `confusion_matrix.png`: Detailed performance on the test set.

## How to Run
Run the entire pipeline using:
```bash
python engine.py
```

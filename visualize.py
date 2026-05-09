import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def visualize_graph_structure(data, num_nodes=100):
    # Convert a subset of the graph to NetworkX for visualization
    G = nx.Graph()
    
    # Add nodes from 'news' and 'tweet'
    news_nodes = [f'n{i}' for i in range(min(num_nodes // 2, data['news'].num_nodes))]
    tweet_nodes = [f't{i}' for i in range(min(num_nodes // 2, data['tweet'].num_nodes))]
    
    G.add_nodes_from(news_nodes, type='news')
    G.add_nodes_from(tweet_nodes, type='tweet')
    
    # Add edges
    edge_index = data['news', 'shared_by', 'tweet'].edge_index
    for i in range(edge_index.shape[1]):
        n_idx = edge_index[0, i].item()
        t_idx = edge_index[1, i].item()
        if f'n{n_idx}' in G and f't{t_idx}' in G:
            G.add_edge(f'n{n_idx}', f't{t_idx}')
            
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Color coding
    node_colors = ['skyblue' if G.nodes[n]['type'] == 'news' else 'orange' for n in G.nodes]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8)
    plt.title("Heterogeneous News-Tweet Network Graph")
    plt.savefig('network_graph.png')
    print("Network graph visualization saved as 'network_graph.png'")
    plt.show()

if __name__ == "__main__":
    from preprocessing import load_fakenewsnet
    from graph_builder import build_hetero_graph
    import os
    
    # Load a small sample of real data for visualization
    data_path = r'c:\Users\Rohan\Desktop\GNN\dataset.csv\FakeNewsNet-master\FakeNewsNet-master\dataset'
    if os.path.exists(data_path):
        print("Loading real data sample for visualization...")
        news_features, labels, edge_index, num_tweets = load_fakenewsnet(data_path, subset='politifact', sample_size=20)
        data = build_hetero_graph(news_features, labels, edge_index, num_tweets)
        visualize_graph_structure(data, num_nodes=50)
    else:
        print("Dataset not found, using mock data...")
        news_feats = torch.randn((10, 128))
        labels = torch.randint(0, 2, (10,))
        mock_edges = torch.tensor([[0, 1, 2], [0, 1, 1]], dtype=torch.long)
        data = build_hetero_graph(news_feats, labels, mock_edges, num_tweets=2)
        visualize_graph_structure(data)

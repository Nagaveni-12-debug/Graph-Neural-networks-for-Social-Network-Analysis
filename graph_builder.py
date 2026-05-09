import torch
from torch_geometric.data import HeteroData
import numpy as np

def build_hetero_graph(news_features, labels, news_tweet_edge_index, num_tweets):
    data = HeteroData()
    
    # 1. Add News Nodes
    data['news'].x = torch.tensor(news_features, dtype=torch.float)
    data['news'].y = torch.tensor(labels, dtype=torch.long)
    
    # 2. Add Tweet Nodes (using constant features as we don't have tweet content)
    # We use 64-dim features for tweets
    data['tweet'].x = torch.ones((num_tweets, 64))
    
    # 3. Add Edges (News -> Tweet)
    # The edge_index from preprocessing is [News_idx, Tweet_idx]
    data['news', 'shared_by', 'tweet'].edge_index = news_tweet_edge_index
    
    # Add reverse edges (Tweet -> News)
    data['tweet', 'rev_shared_by', 'news'].edge_index = news_tweet_edge_index.flip([0])
    
    return data

if __name__ == "__main__":
    # Test
    news_feats = np.random.randn(10, 128)
    labels = np.random.randint(0, 2, 10)
    edges = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    graph = build_hetero_graph(news_feats, labels, edges, 2)
    print(graph)

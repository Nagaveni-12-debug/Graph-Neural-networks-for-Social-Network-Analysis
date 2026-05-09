import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, HeteroConv, Linear

class SNA_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        # 1. GCN Layer (Wrapped in HeteroConv)
        self.conv1 = HeteroConv({
            ('news', 'shared_by', 'tweet'): SAGEConv((-1, -1), hidden_channels),
            ('tweet', 'rev_shared_by', 'news'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')
        
        # 2. GAT Layer (Wrapped in HeteroConv)
        self.conv2 = HeteroConv({
            ('news', 'shared_by', 'tweet'): GATConv(-1, hidden_channels, heads=4, concat=False, add_self_loops=False),
            ('tweet', 'rev_shared_by', 'news'): GATConv(-1, hidden_channels, heads=4, concat=False, add_self_loops=False),
        }, aggr='sum')
        
        # 3. Aggregation & Fully Connected Layers
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Apply first convolution (GCN/SAGE)
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Apply second convolution (GAT)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Final classification on 'news' nodes
        out = self.lin(x_dict['news'])
        
        return out

import torch
import matplotlib.pyplot as plt
from preprocessing import load_fakenewsnet
from graph_builder import build_hetero_graph
from gnn_model import SNA_GNN
from visualize import visualize_graph_structure
from evaluation import evaluate_model
from torch_geometric.transforms import RandomNodeSplit
import os

def train():
    # 1. Load and Preprocess Data (Real FakeNewsNet Dataset)
    data_path = r'c:\Users\Rohan\Desktop\GNN\dataset.csv\FakeNewsNet-master\FakeNewsNet-master\dataset'
    print("Loading FakeNewsNet data...")
    news_features, labels, news_tweet_edge_index, num_tweets = load_fakenewsnet(data_path, subset='politifact', sample_size=200)
    
    # 2. Build Graph
    print(f"Building heterogeneous graph with {len(news_features)} news and {num_tweets} tweets...")
    data = build_hetero_graph(news_features, labels, news_tweet_edge_index, num_tweets)
    # visualize_graph_structure(data) # Removed to focus on accuracy graph
    
    # 3. Split Data (Train/Val/Test)
    transform = RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.1)
    data = transform(data)
    
    # 4. Initialize Model
    model = SNA_GNN(hidden_channels=64, out_channels=2, metadata=data.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 5. Training Loop
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    
    print("Starting training...")
    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(out[data['news'].train_mask], data['news'].y[data['news'].train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            pred = out.argmax(dim=-1)
            
            # Train metrics
            train_mask = data['news'].train_mask
            train_correct = (pred[train_mask] == data['news'].y[train_mask]).sum()
            train_acc = int(train_correct) / int(train_mask.sum())
            
            # Val metrics
            val_mask = data['news'].val_mask
            val_loss = criterion(out[val_mask], data['news'].y[val_mask])
            val_correct = (pred[val_mask] == data['news'].y[val_mask]).sum()
            val_acc = int(val_correct) / int(val_mask.sum())
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
        print(f'Epoch: {epoch:02d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # 6. Generate Styled Accuracy Graph
    print("Generating styled accuracy graph...")
    plt.figure(figsize=(14, 9))
    plt.rcParams['font.family'] = 'sans-serif'
    
    epochs = range(1, len(train_accs) + 1)
    t_accs = [a * 100 for a in train_accs]
    v_accs = [a * 100 for a in val_accs]
    
    plt.plot(epochs, t_accs, 'o-', color='#0033CC', label='Training Accuracy', markersize=8, linewidth=2)
    plt.plot(epochs, v_accs, 's-', color='#108010', label='Validation Accuracy', markersize=8, linewidth=2)
    
    # Add labels on points with offset
    for i, (t, v) in enumerate(zip(t_accs, v_accs)):
        plt.text(i + 1, t + 1.5, f'{t:.1f}', color='#0033CC', fontsize=10, ha='center', fontweight='bold')
        plt.text(i + 1, v - 3.5 if v > 55 else v + 1.5, f'{v:.1f}', color='#108010', fontsize=10, ha='center', fontweight='bold')

    plt.title('Training and Validation Accuracy', fontsize=22, fontweight='bold', pad=30)
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.ylim(50, 100)
    plt.xlim(0.5, 20.5)
    plt.xticks(epochs)
    plt.yticks(range(50, 101, 5))
    
    # Grid styling
    plt.grid(True, which='both', linestyle='--', alpha=0.5, color='lightgray')
    plt.gca().set_axisbelow(True)
    
    # Legend
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    
    # Final Accuracy Annotation Box (Bottom Right)
    final_t = t_accs[-1]
    final_v = v_accs[-1]
    textstr = f"Final Accuracy (Epoch 20)\n"
    textstr += f"Training Accuracy: {final_t:.1f}%\n"
    textstr += f"Validation Accuracy: {final_v:.1f}%"
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#F8F8F8', alpha=1.0, edgecolor='#CCCCCC')
    plt.text(0.97, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=props,
             color='black', linespacing=1.5)

    plt.tight_layout()
    plt.savefig('training_validation_graph.png', dpi=300)
    print("Graph saved as 'training_validation_graph.png'")
    plt.show()
    
    # 7. Evaluate Model
    evaluate_model(model, data)
    
    return model

if __name__ == "__main__":
    train()
